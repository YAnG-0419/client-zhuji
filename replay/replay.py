"""EEF-derived dual-arm and Wuji replay; default offline planning, --check reads hardware.

Cartesian targets are converted to continuous IK and certified joint-linear MoveJ
segments. The direct ServoP/ServoJ backends remain disabled after prior faults.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

import findings
from episode import ROBOT_NAMES, SIDES, hand_names, load_episode, pose_error, sample
from clearance import certify_path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from robot_home import Hands, process_commands
from home_policy import require_status, stage_guard
from transport import replay_backend as backend

DEFAULT_EPISODE = ROOT/'record/data/20261104_133350_628817_combined_record_episode_003'
# Operator-selected thresholds; first-frame FK consistency checks remain separate.
from replay_thresholds import (IK_POSITION_TOL, IK_ROTATION_TOL, CHORD_ROTATION_TOL,
                               MODEL_CLEARANCE_M, LIVE_CLEARANCE_M, INTERNAL_WRIST_CLEARANCE_M)
HAND_FIRST_FRAME_TOL = math.radians(6.)
# MoveJ planning limits. Raised to 1 rad/s by operator decision on 2026-11-04:
# at 0.25 rad/s arm / 0.4 rad/s finger the 84 s episode planned to 772 s.
ARM_SPEED_LIMIT = 1.         # rad/s
ARM_ACCEL_LIMIT = .5         # rad/s^2
HAND_SPEED_LIMIT = 1.        # rad/s, finger targets
HAND_SETTLE_ALLOWANCE_S = 8.
# A MoveJ the firmware refuses is still acknowledged with result=success, so
# absent motion is reported directly instead of as a late settle timeout.
# Only commands whose travel far exceeds the detection threshold are judged,
# so a slow-but-moving short segment is never called stalled.
NO_MOTION_GRACE_S = 1.5
NO_MOTION_PROGRESS_RAD = .002
NO_MOTION_MIN_TRAVEL_RAD = .05
# Extra time allowed past a segment's nominal duration for it to come to rest.
# Measured on 2026-11-04 (runs/20261104_195451_209950): 246 of 304 segments
# never met the rest tolerance and spent the whole allowance waiting, which
# made execution 32.8x the source duration against a planned 9.2x. Zero by
# operator decision: advance as soon as the nominal duration has elapsed.
SETTLE_ALLOWANCE_S = 0.


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def timing(q0, q1, source_delta, hand_speed, speed):
    travel = float(np.max(abs(q1[:14]-q0[:14])))
    return max(.25, source_delta/speed, 1.875*travel/ARM_SPEED_LIMIT,
               math.sqrt(5.774*travel/ARM_ACCEL_LIMIT), 1.875*source_delta*hand_speed/HAND_SPEED_LIMIT)


def guard_values(q, dq, start, goal, indices):
    """stage_guard's measurements, without its abort thresholds."""
    active = list(indices)
    span = sum((goal[i]-start[i])**2 for i in active)
    alpha = min(1., max(0., sum((q[i]-start[i])*(goal[i]-start[i]) for i in active)/max(span, 1e-12)))
    return {'progress': alpha,
            'path_deviation_rad': max(abs(q[i]-(start[i]+alpha*(goal[i]-start[i]))) for i in active),
            'held_error_rad': max([abs(q[i]-start[i]) for i in range(16) if i not in active] or [0.]),
            'max_speed_rad_s': max(abs(dq[i]) for i in active)}


def chord_deviation(model, q0, q1, pose0, pose1, samples=21):
    """Worst pose deviation inside a straight joint chord against the source.

    The endpoints are exact IK solutions, so only the interior is informative.
    Orientation is compared against the slerp of the two source poses, matching
    how the recording itself is resampled.
    """
    from scipy.spatial.transform import Rotation, Slerp
    knots = np.array([0., 1.])
    slerps = [Slerp(knots, Rotation.from_quat(np.array([pose0[i, 3:], pose1[i, 3:]])))
              for i in range(pose0.shape[0])]
    worst_p = worst_r = 0.
    for alpha in np.linspace(0., 1., samples)[1:-1]:
        model.set_q(q0+alpha*(q1-q0))
        target = np.empty_like(pose0)
        target[:, :3] = pose0[:, :3]+alpha*(pose1[:, :3]-pose0[:, :3])
        for i in range(pose0.shape[0]):
            target[i, 3:] = slerps[i](alpha).as_quat()
        p, r = pose_error(model.poses(), target)
        worst_p, worst_r = max(worst_p, float(max(p))), max(worst_r, float(max(r)))
    return worst_p, worst_r


def solve_eef(model, seed, target):
    """Bounded EEF IK, retaining the measured branch where MoveJ permits it."""
    seed = np.clip(seed, model.lower+2e-5, model.upper-2e-5)
    try:
        return model.solve(seed, target)
    except ValueError:
        # Near the wrist bounds the damped local solver can stall. Use a
        # bounded numerical solve, with identical pose acceptance criteria.
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        q = seed.copy()
        for arm in range(2):
            sl = slice(7*arm, 7*arm+7)
            target_rotation = Rotation.from_quat(target[arm, 3:])
            def residual(values):
                q[sl] = values
                model.set_q(q)
                actual = model.poses()[arm]
                return np.r_[actual[:3]-target[arm, :3],
                             .3*(target_rotation*Rotation.from_quat(actual[3:]).inv()).as_rotvec()]
            answer = least_squares(residual, q[sl].copy(),
                                   bounds=(model.lower[sl]+2e-5, model.upper[sl]-2e-5),
                                   max_nfev=500, ftol=1e-11, xtol=1e-11, gtol=1e-11)
            q[sl] = answer.x
        return q


def seed_matches_target(model, seed, target):
    """Cheap FK acceptance for an already solved cross-embodiment seed."""
    # Reuse the executable-target margin, not the numerical solver seed margin.
    if (np.shape(seed) != (16,) or not np.isfinite(seed).all()
            or np.any(seed < model.lower+1e-5) or np.any(seed > model.upper-1e-5)):
        return False
    model.set_q(seed)
    p, r = pose_error(model.poses(), target)
    return (np.isfinite(p).all() and np.isfinite(r).all()
            and max(p) <= IK_POSITION_TOL and max(r) <= IK_ROTATION_TOL)


def make_plan(model, t, eef, hands, measured_q, meta, speed, out, reuse_valid_seed=False):
    if not np.isfinite(speed) or not 0 < speed <= 1:
        raise ValueError('speed must be within (0,1]')
    urdf = ROOT/'record/config/model/robot.urdf'
    if hashlib.sha256(urdf.read_bytes()).hexdigest() != meta['fk']['urdf_sha256']:
        raise ValueError('Recording FK URDF differs from installed model provenance')
    if meta['fk']['frame_id'] != 'base_link' or meta['fk']['quaternion_order'] != 'xyzw':
        raise ValueError('Unsupported recording EEF convention')
    if not np.allclose(meta['fk']['sdk_eef_offset_from_wrist_roll_m'], [-.0338, 0., -.2125], atol=1e-10, rtol=0):
        raise ValueError('Recorded SDK EEF offset differs from replay model')
    limits = json.loads((ROOT/'teleop/wuji_control/config/joint_limits.json').read_text())
    for i, s in enumerate(SIDES):
        bounds = np.array(limits[s])
        if np.any(hands[:, i] < bounds[:, 0]-1e-6) or np.any(hands[:, i] > bounds[:, 1]+1e-6):
            excess = max(float(np.max(bounds[:, 0]-hands[:, i])), float(np.max(hands[:, i]-bounds[:, 1])))
            detail = '%s hand commands exceed the configured limits by up to %.5f rad; clamping' % (s, excess)
            if not findings.permit('hand_command_outside_limits', detail, value=excess, side=s):
                raise ValueError('Recorded hand commands exceed limits: '+s)
        # Only float32 roundoff <=1e-6 is tolerated.
        hands[:, i] = np.clip(hands[:, i], bounds[:, 0], bounds[:, 1])
    branch_switches = []
    seed = measured_q[0].copy()
    model.set_q(seed)
    p, r = pose_error(model.poses(), eef[0])
    if max(p) > .001 or max(r) > .0035:
        detail = ('Frame-zero recorded joint seed and EEF FK disagree by %.5f m / %.5f rad'
                  % (max(p), max(r)))
        if not findings.permit('frame_zero_seed_disagrees', detail, value=max(p)):
            raise ValueError('Frame-zero recorded joint seed and EEF FK disagree')
    qs, max_p, max_r = [], 0., 0.
    reused_seed_frames = 0
    for index, target in enumerate(eef):
        # This is a same-robot demonstration: measured joints identify its
        # redundant branch, while IK solves the interpolated EEF pose exactly.
        # Previous-solution-only seeding drifts along the 7-DOF nullspace.
        branch_seed = measured_q[index].copy()
        branch_seed[14:] = measured_q[0, 14:]
        raw_first_frame_valid = np.all(branch_seed >= model.lower+2e-5) and np.all(branch_seed <= model.upper-2e-5)
        if index == 0 and raw_first_frame_valid:
            # Approach uses recorded joints directly when valid MoveJ targets.
            seed = branch_seed.copy()
        elif reuse_valid_seed and seed_matches_target(model, branch_seed, target):
            seed = branch_seed.copy()
            reused_seed_frames += 1
        else:
            try:
                seed = solve_eef(model, branch_seed, target)
            except ValueError as exc:
                raise ValueError('EEF IK at source %.3fs: %s' % (t[index], exc)) from exc
        if qs and np.max(abs(seed[:14]-qs[-1][:14])) > .1:
            # A fresh measured seed can switch redundant branches at a wrist
            # bound. Retry from the preceding accepted solution before rejecting.
            continuous = solve_eef(model, qs[-1], target)
            model.set_q(continuous)
            cp, cr = pose_error(model.poses(), target)
            if max(cp) > IK_POSITION_TOL or max(cr) > IK_ROTATION_TOL:
                detail = ('Discontinuous IK branch at source %.3fs (frame %d): residual %.5f m / %.5f rad'
                          % (t[index], index, max(cp), max(cr)))
                if not findings.permit('discontinuous_ik_branch', detail, value=max(cr),
                                       frame=index, source_time_s=float(t[index])):
                    raise ValueError('Discontinuous IK branch at source %.3fs (frame %d)' % (t[index], index))
            step = float(np.max(abs(continuous[:14]-qs[-1][:14])))
            if step > .1:
                # Wrists clipped to the executable envelope sit at a bound for
                # several frames and rejoin the interior in one step. Accept it
                # only when the joint chord itself still tracks the source pose.
                deviation = chord_deviation(model, qs[-1], continuous, eef[index-1], target)
                if deviation[0] > IK_POSITION_TOL or deviation[1] > CHORD_ROTATION_TOL:
                    detail = ('IK branch step of %.4f rad at source %.3fs (frame %d) deviates '
                              '%.4f m / %.4f rad from the source'
                              % (step, t[index], index, deviation[0], deviation[1]))
                    if not findings.permit('uncertified_branch_step', detail, value=deviation[1],
                                           frame=index, source_time_s=float(t[index]),
                                           joint_step_rad=step):
                        raise ValueError('Discontinuous IK branch at source %.3fs (frame %d): step %.4f rad '
                                         'deviates %.4f m / %.4f rad from the source'
                                         % (t[index], index, step, deviation[0], deviation[1]))
                branch_switches.append({'frame': index, 'source_time_s': float(t[index]),
                                        'joint_step_rad': step,
                                        'chord_position_deviation_m': deviation[0],
                                        'chord_rotation_deviation_rad': deviation[1]})
            seed = continuous
        model.set_q(seed)
        p, r = pose_error(model.poses(), target)
        max_p, max_r = max(max_p, float(max(p))), max(max_r, float(max(r)))
        if max(p) > IK_POSITION_TOL or max(r) > IK_ROTATION_TOL:
            save(out/'ik_failure.json', {'source_time_s': float(t[index]),
                 'target_eef_xyz_xyzw': target.tolist(), 'nearest_eef_xyz_xyzw': model.poses().tolist(),
                 'position_error_m': p.tolist(), 'rotation_error_rad': r.tolist(),
                 'recorded_q16': measured_q[index].tolist(), 'bounded_solution_q16': seed.tolist(),
                 'joint_lower': model.lower.tolist(), 'joint_upper': model.upper.tolist(),
                 'note': 'Local bounded IK failed the stated tolerance; this is not a global unreachability proof.'})
            detail = ('EEF IK at source %.3fs (frame %d): residual %.6f m / %.6f rad exceeds '
                      '%.1f mm / %.6f rad; using the nearest reachable solution'
                      % (t[index], index, max(p), max(r), IK_POSITION_TOL*1000, IK_ROTATION_TOL))
            if not findings.permit('ik_residual_above_tolerance', detail, value=max(p),
                                   frame=index, source_time_s=float(t[index]),
                                   position_error_m=float(max(p)), rotation_error_rad=float(max(r))):
                raise ValueError('EEF IK at source %.3fs: residual %.6fm / %.6frad exceeds %.1fmm / %.6frad; target may be unreachable within MoveJ limits' %
                                 (t[index], max(p), max(r), IK_POSITION_TOL*1000, IK_ROTATION_TOL))
        invalid = np.flatnonzero((seed < model.lower+1e-5) | (seed > model.upper-1e-5))
        if len(invalid):
            names = [ROBOT_NAMES[j] for j in invalid]
            # Clipping keeps the solution inside the envelope the firmware will
            # execute; the pose error this introduces is recorded above.
            seed = np.clip(seed, model.lower+1e-5, model.upper-1e-5)
            detail = 'IK solution at source %.3fs reached the joint limit: %s' % (t[index], names)
            if not findings.permit('ik_solution_at_joint_limit', detail,
                                   frame=index, source_time_s=float(t[index]), joints=names):
                raise ValueError('IK joint limit at %.3fs: %s' % (t[index], names))
        qs.append(seed.copy())
        if index % 200 == 0:
            print('IK %d/%d' % (index+1, len(t)), flush=True)
    qs = np.array(qs)
    if reuse_valid_seed:
        print('IK REUSED %d/%d existing solutions; path checks follow' %
              (reused_seed_frames, len(t)-1), flush=True)
    # Adapt waypoints until the FK of each joint chord follows the EEF source.
    pending, indices = [(0, len(t)-1)], {0, len(t)-1}
    max_chord_p, max_chord_r = 0., 0.
    while pending:
        a, b = pending.pop()
        bad = t[b]-t[a] > 1.0
        cp, cr = 0., 0.
        if not bad:
            for j in range(a, b+1):
                alpha = (t[j]-t[a])/(t[b]-t[a])
                model.set_q(qs[a]+alpha*(qs[b]-qs[a]))
                p, r = pose_error(model.poses(), eef[j])
                cp, cr = max(cp, float(max(p))), max(cr, float(max(r)))
            # Source frames alone leave the interior of a one-frame segment
            # unexamined, because both ends are exact IK solutions.
            ip, ir = chord_deviation(model, qs[a], qs[b], eef[a], eef[b])
            cp, cr = max(cp, ip), max(cr, ir)
            bad = cp > IK_POSITION_TOL or cr > CHORD_ROTATION_TOL
        if bad:
            if b-a <= 1:
                detail = ('Chord between frames %d and %d deviates %.5f m / %.5f rad; '
                          'the sample rate cannot be refined further' % (a, b, cp, cr))
                if not findings.permit('chord_accuracy_unrefinable', detail, value=cp,
                                       first_frame=a, last_frame=b):
                    raise ValueError('EEF chord accuracy needs a finer sample rate')
                max_chord_p, max_chord_r = max(max_chord_p, cp), max(max_chord_r, cr)
                continue
            mid = (a+b)//2
            indices.add(mid)
            pending.extend([(a, mid), (mid, b)])
        else:
            max_chord_p, max_chord_r = max(max_chord_p, cp), max(max_chord_r, cr)
    idx = np.array(sorted(indices))
    durations = []
    for a, b in zip(idx[:-1], idx[1:]):
        hs = float(np.max(abs(np.diff(hands[a:b+1], axis=0))/np.diff(t[a:b+1])[:, None, None]))
        durations.append(timing(qs[a], qs[b], t[b]-t[a], hs, speed))
    print('Checking swept arm/body/head clearance for %d waypoints' % len(idx), flush=True)
    collision = certify_or_record(model, qs[idx], 'offline waypoint path')
    report = dict(meta, backend='EEF -> continuous seeded IK -> supervised stop-at-waypoint MoveJ',
                  source_frames=len(t), waypoints=len(idx), segments=len(durations),
                  reused_seed_frames=reused_seed_frames,
                  first_frame_arm_source=('recorded /joint_states arm positions' if np.max(abs(qs[0,:14]-measured_q[0,:14])) < 1e-8 else
                                          'bounded IK adjustment of recorded first-frame joints; reached with MoveJ'),
                  first_frame_joint_adjustment_rad=(qs[0,:14]-measured_q[0,:14]).tolist(),
                  arm_joint_limits='URDF mechanical ranges intersected with the SDK MoveJ envelope; '
                                   'targets outside it are acknowledged but not executed by the firmware',
                  requested_speed=speed, nominal_motion_duration_s=float(sum(durations)),
                  waypoint_settle_allowance_s=SETTLE_ALLOWANCE_S*len(durations),
                  ik_max_position_residual_m=max_p, ik_max_rotation_residual_rad=max_r,
                  ik_position_tolerance_m=IK_POSITION_TOL, ik_rotation_tolerance_rad=IK_ROTATION_TOL,
                  hand_first_frame_tolerance_rad=HAND_FIRST_FRAME_TOL,
                  sampled_chord_position_error_m=max_chord_p, sampled_chord_rotation_error_rad=max_chord_r,
                  chord_rotation_tolerance_rad=CHORD_ROTATION_TOL,
                  certified_branch_switches=branch_switches,
                  cartesian_accuracy_note='Chord deviation checked at source resampling points and in the chord interior; collision bounds are continuous.',
                  collision=collision, model_dependencies=model.dependencies,
                  clearance_metric='pair distance minus required margin plus 5 mm; not raw geometric distance',
                  required_pair_clearance_m={'same_wrist_yaw_roll': INTERNAL_WRIST_CLEARANCE_M, 'other_pairs': MODEL_CLEARANCE_M},
                  collision_scope='Existing arm/body/head geometry; excludes Wuji/adapter/cables and external obstacles',
                  head_policy='Offline uses recorded initial head; live preserves current head and recertifies the entire path; no head commands',
                  timing_policy='Arms stop at waypoints; hands use same monotonic source phase; not original wall-clock timing')
    np.savez_compressed(out/'plan.npz', t=t, eef=eef, hands=hands, q16=qs,
                        waypoint_indices=idx, durations_s=durations)
    save(out/'plan.json', report)
    return qs, idx, durations, report


def certify_or_record(model, qs, where):
    """Certify swept clearance, or record the shortfall and continue."""
    try:
        return certify_path(model, qs)
    except ValueError as exc:
        if not findings.permit('clearance_below_requirement', '%s: %s' % (where, exc)):
            raise
        return {'certified': False, 'where': where, 'error': str(exc),
                'required_clearance_m': MODEL_CLEARANCE_M,
                'note': 'Recorded by --permissive; the path was kept as planned'}


def stop_stack(stack):
    # Inspect the entire group even if its launcher has already exited.
    for sig, seconds in ((signal.SIGINT, 10), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
        try:
            os.killpg(stack.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            stack.poll()  # Reap the launcher before checking group existence.
            try:
                os.killpg(stack.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(.1)
    stack.wait(timeout=2)


class Session:
    def __init__(self, robot, hands, model, t, eef, hand_data, out):
        self.robot, self.hands, self.model = robot, hands, model
        self.t, self.eef, self.hand_data = t, eef, hand_data
        self.last_hand = np.array([hands.feedback()[s] for s in SIDES])
        self.last_tick = time.monotonic()
        self.feedback_log = (out/'feedback.jsonl').open('w')
        self.log_at = 0.
        self.stop_file = out/'stop.request'

    def tick(self, desired, start, goal, source_time=None):
        now = time.monotonic()
        dt = now-self.last_tick
        # stop.request and Ctrl-C are the operator's stop and always abort.
        if self.stop_file.exists():
            raise RuntimeError('stop.request received')
        if not 0 < dt <= .1:
            detail = 'Supervision loop interval was %.3f s' % dt
            if not findings.permit('scheduler_stall', detail, value=dt):
                raise RuntimeError('Scheduler stalled >100 ms or stop.request received')
        self.last_tick = now
        q, dq = self.robot.healthy()
        try:
            require_status(self.robot.info)
        except RuntimeError as exc:
            if not findings.permit('robot_status_unexpected', str(exc)):
                raise
        guard = guard_values(q.tolist(), dq.tolist(), start.tolist(), goal.tolist(), range(14))
        try:
            stage_guard(q.tolist(), dq.tolist(), start.tolist(), goal.tolist(), range(14))
        except RuntimeError as exc:
            if not findings.permit('tracking_guard', str(exc),
                                   value=guard['path_deviation_rad'],
                                   source_time_s=source_time):
                raise
        self.model.set_q(q)
        margin, pair = self.model.clearance()
        if margin < LIVE_CLEARANCE_M:
            detail = 'Model clearance %.5f m between %s' % (margin, pair)
            if not findings.permit('live_clearance_low', detail, value=LIVE_CLEARANCE_M-margin, bodies=pair):
                raise RuntimeError('Live joint limit or collision guard: '+str(pair))
        if np.any(q < self.model.lower) or np.any(q > self.model.upper):
            outside = [ROBOT_NAMES[j] for j in
                       np.flatnonzero((q < self.model.lower) | (q > self.model.upper))]
            detail = 'Measured joints outside the model range: %s' % outside
            if not findings.permit('measured_joint_outside_range', detail, joints=outside):
                raise RuntimeError('Live joint limit or collision guard: '+str(pair))
        measured = np.array([self.hands.feedback()[s] for s in SIDES])
        hand_error = float(np.max(abs(measured-self.last_hand)))
        if hand_error > .2:
            detail = 'Hand tracking error %.4f rad exceeds 0.2 rad' % hand_error
            if not findings.permit('hand_tracking_error', detail, value=hand_error,
                                   source_time_s=source_time):
                raise RuntimeError('Hand tracking error exceeds 0.2 rad')
        for s in SIDES:
            for topic, expected, message in (
                    ('/target/program', 1, 'Conflicting hand program publisher'),
                    ('/joint_states', 1, 'Missing/conflicting hand feedback owner'),
                    ('/command', 1, 'Missing/conflicting hand control output owner')):
                actual = self.hands.node.count_publishers('/wuji/'+s+topic)
                if actual != expected:
                    detail = '%s: /wuji/%s%s has %d publishers' % (message, s, topic, actual)
                    if not findings.permit('hand_topic_ownership', detail, publishers=actual):
                        raise RuntimeError(message)
        self.last_hand += np.clip(desired-self.last_hand, -HAND_SPEED_LIMIT*dt, HAND_SPEED_LIMIT*dt)
        self.hands.set_targets({s: self.last_hand[i].tolist() for i, s in enumerate(SIDES)})
        self.robot.lease_at = now
        if now-self.log_at >= .1:
            record = dict(guard, q16=q.tolist(), source_time_s=source_time,
                          clearance_score_m=margin, hand_error_rad=float(np.max(abs(measured-self.last_hand))))
            self.feedback_log.write(json.dumps(record)+'\n')
            self.feedback_log.flush()
            self.log_at = now
        return q, dq, guard['progress'], measured

    def approach_hands(self, duration, stationary_q):
        initial = self.last_hand.copy()
        goal = self.hand_data[0]
        began, settled = time.monotonic(), None
        while True:
            elapsed = time.monotonic()-began
            u = min(1., elapsed/duration)
            alpha = 10*u**3-15*u**4+6*u**5
            q, dq, _, h = self.tick(initial+alpha*(goal-initial), stationary_q, stationary_q)
            drift = float(np.max(abs(q-stationary_q)))
            if drift > .01 or np.max(abs(dq)) > .02:
                detail = 'Arms drifted %.5f rad during the hand approach' % drift
                if not findings.permit('arm_drift_during_hand_approach', detail, value=drift):
                    raise RuntimeError('Arms moved during hand approach')
            error = float(np.max(abs(h-goal)))
            ready = elapsed >= duration and error < HAND_FIRST_FRAME_TOL
            settled = (time.monotonic() if settled is None else settled) if ready else None
            if settled is not None and time.monotonic()-settled > .3:
                return
            if elapsed > duration+HAND_SETTLE_ALLOWANCE_S:
                detail = ('Hands stopped %.4f rad from the first frame after %.1f s; continuing'
                          % (error, elapsed))
                if not findings.permit('hands_missed_first_frame', detail, value=error):
                    raise TimeoutError('Hands did not reach first frame')
                return
            time.sleep(.01)

    def segment(self, planned_start, goal, duration, a, b):
        current, dq = self.robot.healthy()
        offset = float(np.max(abs(current-planned_start)))
        if offset > .02 or np.max(abs(dq)) > .02:
            detail = 'MoveJ start is %.5f rad from the planned waypoint' % offset
            if not findings.permit('segment_start_off_plan', detail, value=offset, source_time_s=a):
                raise RuntimeError('MoveJ start is not stationary at the planned waypoint')
        # The live start is certified exactly, rather than assuming nominal arrival.
        from replay_movej_path import check_current_segment
        def pump():
            self.tick(self.last_hand, current, current)
        try:
            start, dq, bound, attempts = check_current_segment(self.model, self.robot.healthy, goal, pump)
        except (ValueError, RuntimeError) as exc:
            if not findings.permit('live_start_uncertified', str(exc), source_time_s=a):
                raise
            start, dq = self.robot.healthy()
            start, bound, attempts = start.copy(), None, 0
        moved = float(np.max(abs(start-current)))
        if moved > .002 or np.max(abs(dq)) > .02:
            detail = 'Robot moved %.5f rad during the live-start check' % moved
            if not findings.permit('moved_during_certification', detail, value=moved, source_time_s=a):
                raise RuntimeError('Robot moved during live-start collision check')
        duration = max(duration, timing(start, goal, 0., 0., 1.))
        self.robot.arm(backend='supervised_movej')
        self.robot.commands_sent += 1
        target = findings.clip_to_envelope(self.model.lower[:14], self.model.upper[:14],
                                           goal[:14], 'segment ending at source %.3fs' % b,
                                           ROBOT_NAMES)
        self.robot.query('request_movej', timeout=.08,
                         data={'joint': target.tolist(), 'time': float(duration)})
        began, phase, settled = time.monotonic(), 0., None
        while True:
            elapsed = time.monotonic()-began
            q, dq = self.robot.healthy()
            delta = goal[:14]-start[:14]
            progress = float(np.clip(np.dot(q[:14]-start[:14], delta)/max(np.dot(delta, delta), 1e-12), 0., 1.))
            u = min(1., elapsed/duration)
            nominal = 10*u**3-15*u**4+6*u**5
            resting = np.max(abs(q-goal)) < .01 and np.max(abs(dq)) < .02
            phase = max(phase, min(nominal, progress+.02/max(np.max(abs(delta)), .02)))
            if elapsed >= duration and resting:
                phase = 1.
            source = a+phase*(b-a)
            desired_eef, desired_hands = sample(self.t, self.eef, self.hand_data, source)
            q, dq, _, measured_hands = self.tick(desired_hands, start, goal, source)
            # Runtime EEF supervision checks the SDK feedback against the source.
            feedback = self.robot.pose
            measured_eef = np.array([feedback[s+'_position']+feedback[s+'_quat'][1:]+feedback[s+'_quat'][:1] for s in SIDES])
            ep, er = pose_error(measured_eef, desired_eef)
            if max(ep) > .05 or max(er) > .25:
                detail = ('EEF tracking error %.4f m / %.4f rad exceeds 5 cm / 0.25 rad'
                          % (max(ep), max(er)))
                if not findings.permit('eef_tracking_error', detail, value=max(ep), source_time_s=source):
                    raise RuntimeError('EEF tracking error exceeds 5 cm / 0.25 rad')
            ready = (elapsed >= duration and resting and phase == 1. and
                     np.max(abs(measured_hands-desired_hands)) < .05 and max(ep) < .01 and max(er) < .05)
            settled = (time.monotonic() if settled is None else settled) if ready else None
            if settled is not None and time.monotonic()-settled >= .15:
                self.robot.armed = False
                return dict(duration_s=elapsed, source_end_s=b, clearance_bound_m=bound,
                            certification_attempts=attempts, eef_position_error_m=ep.tolist(), eef_rotation_error_rad=er.tolist())
            stalled = (elapsed > NO_MOTION_GRACE_S and
                       np.max(abs(delta)) > NO_MOTION_MIN_TRAVEL_RAD and
                       np.max(abs(q[:14]-start[:14])) < NO_MOTION_PROGRESS_RAD)
            if stalled:
                detail = ('Controller acknowledged MoveJ but the arms did not move in %.1f s '
                          '(commanded travel %.4f rad)' % (elapsed, float(np.max(abs(delta)))))
                if not findings.permit('movej_not_executed', detail, source_time_s=source):
                    raise RuntimeError('Controller acknowledged MoveJ but the arms did not move; '
                                       'check the target against the executable joint range')
            if elapsed > duration+SETTLE_ALLOWANCE_S or (stalled and elapsed > NO_MOTION_GRACE_S+1.):
                detail = ('Segment ending at source %.3fs did not settle after %.1f s; '
                          'advancing to the next waypoint' % (b, elapsed))
                if not findings.permit('segment_did_not_settle', detail, source_time_s=source):
                    raise TimeoutError('Synchronized arm/hand segment did not settle')
                self.robot.armed = False
                return dict(duration_s=elapsed, source_end_s=b, clearance_bound_m=bound,
                            certification_attempts=attempts, settled=False,
                            eef_position_error_m=ep.tolist(), eef_rotation_error_rad=er.tolist())
            time.sleep(.01)


def run(args, out, result):
    robot = hands = stack = stack_log = session = None
    ros_initialized = False
    try:
        t, eef, hand_data, measured_q, meta = load_episode(args.episode, args.sample_hz)
        end = t[-1] if args.end_seconds is None else args.end_seconds
        if not np.isfinite(args.start_seconds) or not np.isfinite(end) or not 0 <= args.start_seconds < end <= t[-1]:
            raise ValueError('Requested source window must be finite and inside the recording')
        mask = (t >= args.start_seconds) & (t <= end)
        if np.count_nonzero(mask) < 2:
            raise ValueError('Source window contains fewer than two EEF samples')
        origin = float(t[mask][0])
        t, eef, hand_data, measured_q = t[mask]-origin, eef[mask], hand_data[mask], measured_q[mask]
        meta['selected_source_window_s'] = [origin, float(origin+t[-1])]
        result['source'] = meta
        _, Model, Transport = backend()
        model = Model()
        qs, idx, durations, plan = make_plan(model, t, eef, hand_data, measured_q, meta, args.speed, out)
        result['plan'] = plan
        print('PLAN READY source=%.2fs segments=%d motion=%.1fs' % (t[-1], len(durations), sum(durations)), flush=True)
        if not (args.check or args.execute):
            result['offline_validated'] = True
            return
        commands = process_commands()
        # axis_replay.py also drives this robot and shares the execution lock.
        owners = ('wuji_teleop.switch', 'wuji-teleop-switch', 'tools/manus_teleop.sh',
                  'robot_home.py', 'axis_replay.py')
        if any(owner in c for owner in owners for c in commands):
            raise RuntimeError('Exit teleop/home before replay; another controller owns the robot')
        robot = Transport(model.lower[:14], model.upper[:14])
        state = robot.snapshot()
        save(out/'start_state.json', state)
        try:
            require_status(state['robot_info'])
        except RuntimeError as exc:
            if not findings.permit('robot_status_unexpected', str(exc)):
                raise
        q0 = np.array(state['q16'])
        # EEF is already in base coordinates; preserve the live head and
        # recertify the entire arm path against its actual geometry.
        qs[:, 14:] = q0[14:]
        result['live_head_collision'] = certify_or_record(model, qs[idx], 'live head recertification')
        plan['live_head_q_rad'] = q0[14:].tolist()
        plan['collision'] = result['live_head_collision']
        save(out/'plan.json', plan)
        np.savez_compressed(out/'plan.npz', t=t, eef=eef, hands=hand_data, q16=qs,
                            waypoint_indices=idx, durations_s=durations)
        model.set_q(q0)
        ep, er = pose_error(model.poses(), np.array(state['eef_base_xyz_xyzw']))
        if max(ep) > .001 or max(er) > .0035:
            detail = ('Live model FK and SDK EEF feedback differ by %.5f m / %.5f rad'
                      % (max(ep), max(er)))
            if not findings.permit('live_fk_disagrees', detail, value=max(ep)):
                raise RuntimeError('Live model FK and SDK EEF feedback disagree')
        approach = certify_or_record(model, np.array([q0, qs[0]]), 'first-frame approach')
        result['arm_approach'] = approach
        # A checked MoveJ approach reaches the recorded first-frame arm joints.
        approach_seconds = max(20., timing(q0, qs[0], 0., 0., 1.))
        result['arm_approach_duration_s'] = approach_seconds
        import rclpy
        rclpy.init()
        ros_initialized = True
        target = {s+'_hand': dict(zip(hand_names(s), hand_data[0, i])) for i, s in enumerate(SIDES)}
        hands = Hands(target)
        existing = any('wuji-hand-2-driver' in c or 'wuji-control' in c for c in commands)
        # Check mode never starts drivers or changes control/motor state.
        if args.execute and not existing:
            stack_log = (out/'stack.log').open('w')
            stack = subprocess.Popen(['ros2', 'launch', str(ROOT/'tools/home_bimanual.launch.py')],
                                     stdout=stack_log, stderr=subprocess.STDOUT, start_new_session=True)
        current_hands = np.array([hands.feedback(timeout=20 if stack else 5)[s] for s in SIDES])
        hands.validate_owner()
        for s in SIDES:
            if hands.node.count_publishers('/wuji/'+s+'/command') != 1:
                raise RuntimeError('Expected exactly one wuji_control output publisher for '+s)
        # ROS initialization/discovery can block Python while no robot motion
        # is active. Begin the strict feedback deadlines after that startup.
        robot.start_feedback()
        result['live_preflight_passed'] = True
        if not args.execute:
            return
        q, dq = robot.healthy()
        shift = float(np.max(abs(q-q0)))
        if shift > .002 or np.max(abs(dq)) > .02:
            detail = 'Robot moved %.5f rad during preflight' % shift
            if not findings.permit('moved_during_preflight', detail, value=shift):
                raise RuntimeError('Robot moved during preflight')
        hands.start()
        session = Session(robot, hands, model, t, eef, hand_data, out)
        hand_approach_seconds = max(10., 1.875*float(np.max(abs(hand_data[0]-current_hands)))/.3)
        print('MOVING FIRST FRAME hands duration=%.1fs' % hand_approach_seconds, flush=True)
        session.approach_hands(hand_approach_seconds, q0)
        # Constant source time keeps the hands at frame zero during arm approach.
        # Skip only if all arm joints are already within 2 mrad of the target.
        if np.max(abs(q0[:14]-qs[0, :14])) > .002:
            # Approach EEF follows FK of the joint chord, not source phase zero.
            print('MOVING FIRST FRAME arms MoveJ duration=%.1fs' % approach_seconds, flush=True)
            result['approach_execution'] = execute_approach(session, q0, qs[0], approach_seconds)
        print('REACHED FIRST FRAME', flush=True)
        print('REPLAY START', flush=True)
        result['segments'] = []
        for number, (a, b, duration) in enumerate(zip(idx[:-1], idx[1:], durations), 1):
            try:
                entry = session.segment(qs[a], qs[b], duration, t[a], t[b])
            except (RuntimeError, ValueError, TimeoutError) as exc:
                # stop.request is the operator's stop and is never absorbed.
                if 'stop.request' in str(exc) or not findings.permit(
                        'segment_failed', 'Segment %d of %d (source %.3fs): %s'
                        % (number, len(durations), t[b], exc), source_time_s=float(t[b])):
                    raise
                entry = {'source_end_s': float(t[b]), 'completed': False, 'error': str(exc)}
                robot.armed = False
            result['segments'].append(entry)
            print('REACHED %d/%d source=%.2fs' % (number, len(durations), t[b]), flush=True)
        result['replayed'] = True
    finally:
        cleanup_errors = []
        if robot and robot.armed:
            try:
                robot.emergency_stop()
                result['arm_stop'] = 'sent'
            except Exception as exc:
                cleanup_errors.append('arm stop: '+str(exc))
        if hands:
            if hands.touched:
                try:
                    hands.stop()
                    result['hands_stop'] = 'confirmed'
                except Exception as exc:
                    cleanup_errors.append('hand stop: '+str(exc))
            try:
                hands.close()
            except Exception as exc:
                cleanup_errors.append('hand close: '+str(exc))
        if session:
            session.feedback_log.close()
        if stack:
            try:
                stop_stack(stack)
            except Exception as exc:
                cleanup_errors.append('stack shutdown: '+str(exc))
        if stack_log:
            stack_log.close()
        if robot:
            result['motion_commands_sent'] = robot.commands_sent
            robot.close()
        if ros_initialized:
            import rclpy
            rclpy.shutdown()
        if cleanup_errors:
            result['cleanup_errors'] = cleanup_errors
            raise RuntimeError('; '.join(cleanup_errors))


def execute_approach(session, start, goal, duration):
    robot = session.robot
    # Recheck the actual measured start after the hand enable/approach interval.
    q, dq = robot.healthy()
    drift = float(np.max(abs(q-start)))
    if drift > .002 or np.max(abs(dq)) > .02:
        detail = 'Approach start moved %.5f rad after the hand approach' % drift
        if not findings.permit('approach_start_moved', detail, value=drift):
            raise RuntimeError('Arm approach start moved')
    from replay_movej_path import check_current_segment
    def pump():
        session.tick(session.hand_data[0], start, start)
    try:
        actual, dq, bound, attempts = check_current_segment(session.model, robot.healthy, goal, pump)
    except (ValueError, RuntimeError) as exc:
        if not findings.permit('approach_uncertified', str(exc)):
            raise
        actual, dq = robot.healthy()
        actual, bound, attempts = actual.copy(), None, 0
    moved = float(np.max(abs(actual-start)))
    if moved > .002 or np.max(abs(dq)) > .02:
        detail = 'Approach start changed %.5f rad during certification' % moved
        if not findings.permit('approach_start_changed', detail, value=moved):
            raise RuntimeError('Arm approach start changed during certification')
    robot.arm(backend='supervised_movej')
    robot.commands_sent += 1
    target = findings.clip_to_envelope(session.model.lower[:14], session.model.upper[:14],
                                       goal[:14], 'first-frame approach', ROBOT_NAMES)
    robot.query('request_movej', timeout=.08, data={'joint': target.tolist(), 'time': duration})
    began, settled = time.monotonic(), None
    while True:
        q, dq, _, _ = session.tick(session.hand_data[0], actual, goal)
        ready = np.max(abs(q-goal)) < .01 and np.max(abs(dq)) < .02
        settled = (time.monotonic() if settled is None else settled) if ready else None
        if time.monotonic()-began >= duration and settled is not None and time.monotonic()-settled > .3:
            robot.armed = False
            return {'clearance_bound_m': bound, 'certification_attempts': attempts, 'duration_s': time.monotonic()-began}
        elapsed = time.monotonic()-began
        stalled = (elapsed > NO_MOTION_GRACE_S and
                   np.max(abs(goal[:14]-actual[:14])) > NO_MOTION_MIN_TRAVEL_RAD and
                   np.max(abs(q[:14]-actual[:14])) < NO_MOTION_PROGRESS_RAD)
        if stalled:
            detail = ('Controller acknowledged the first-frame MoveJ but the arms did not move '
                      'in %.1f s' % elapsed)
            if not findings.permit('movej_not_executed', detail):
                raise RuntimeError('Controller acknowledged the first-frame MoveJ but the arms did '
                                   'not move; check the target against the executable joint range')
        if elapsed > duration+SETTLE_ALLOWANCE_S or (stalled and elapsed > NO_MOTION_GRACE_S+1.):
            error = float(np.max(abs(q[:14]-goal[:14])))
            detail = ('First-frame approach stopped %.4f rad short after %.1f s; starting the replay'
                      % (error, elapsed))
            if not findings.permit('approach_did_not_settle', detail, value=error):
                raise TimeoutError('Arm first-frame approach did not settle')
            robot.armed = False
            return {'clearance_bound_m': bound, 'certification_attempts': attempts,
                    'duration_s': elapsed, 'settled': False, 'final_error_rad': error}
        time.sleep(.01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, default=DEFAULT_EPISODE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Offline plan + read-only live preflight')
    mode.add_argument('--execute', action='store_true', help='Approach first frame and execute on hardware')
    parser.add_argument('--speed', type=float, default=.5, help='Maximum source-time speed; stops/limits can slow it further')
    parser.add_argument('--sample-hz', type=float, default=50.)
    parser.add_argument('--start-seconds', type=float, default=0., help='Optional source window start')
    parser.add_argument('--end-seconds', type=float, help='Optional source window end; full episode by default')
    parser.add_argument('--permissive', action='store_true',
                        help='Record accuracy/tracking/state findings instead of aborting, so a '
                             'demonstration completes. Out-of-envelope MoveJ targets move to the '
                             'joint limit. stop.request and Ctrl-C still stop the run.')
    args = parser.parse_args()
    out = ROOT/'replay/runs'/(time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
    out.mkdir(parents=True)
    result = {'completed': False, 'execute_requested': args.execute,
              'permissive': args.permissive, 'motion_commands_sent': 0}
    def interrupt(signum, frame):
        raise KeyboardInterrupt()
    old = signal.signal(signal.SIGTERM, interrupt)
    lock = open('/home/guest/wanchi/replay_tip_align_test_1030/.hands_replay.lock', 'a')
    if args.permissive:
        findings.begin(out/'violations.jsonl')
        print('PERMISSIVE: findings recorded to %s; out-of-envelope MoveJ targets move to the limit'
              % (out/'violations.jsonl'), flush=True)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, out, result)
        result['completed'] = True
    except (Exception, KeyboardInterrupt) as exc:
        result['error'] = type(exc).__name__+': '+str(exc)
        print('REPLAY FAILED:', result['error'], flush=True)
    finally:
        lock.close()
        signal.signal(signal.SIGTERM, old)
        record = findings.active()
        if record is not None:
            result['findings'] = record.summary()
            findings.end()
            counts = result['findings']['counts']
            print('FINDINGS: %d recorded across %d kinds'
                  % (sum(counts.values()), len(counts)), flush=True)
            for kind in sorted(counts, key=lambda k: -counts[k]):
                print('   %-34s %d' % (kind, counts[kind]), flush=True)
        save(out/'result.json', result)
        print('REPORT', out/'result.json', flush=True)
    return 0 if result['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
