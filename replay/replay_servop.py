"""EEF dual-arm + Wuji replay executed as one continuous ServoP stream.

Planning, collision certification, hand handling and the first-frame approach are
reused from replay.py unchanged. Only execution differs:

  replay.py         : waypoints -> stop-at-waypoint MoveJ segments
  replay_servop.py  : every planned 50 Hz joint solution -> joint-linear interpolation
                      -> FK -> request_servop at --rate Hz, no stops

What is streamed is FK of the certified dense joint path (not the raw recording), so
every pose sent is reachable, inside the executable joint range and collision-checked.
Joint speed is capped at SERVO_SPEED_LIMIT so the existing tracking guard (0.35 rad/s)
stays valid; the stream eases in and out over RAMP_S seconds.

  bash replay/replay_servop.sh                 # offline plan only
  bash replay/replay_servop.sh --check         # + read-only live preflight
  bash replay/replay_servop.sh --execute       # real robot
"""
import argparse
import fcntl
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np

import findings
import replay as base
from episode import SIDES, hand_names, load_episode, pose_error, sample
from robot_home import Hands, process_commands
from home_policy import require_status
from transport import replay_backend as backend

ROOT = base.ROOT
SERVO_SPEED_LIMIT = .30  # rad/s; stage_guard aborts above 0.35 rad/s
RAMP_S = 1.0             # wall-clock ease-in / ease-out
HOLD_END_S = .5          # keep streaming the final pose before releasing
TICK_PERIOD_S = .02      # supervision (guards, clearance, hands) period
EEF_TRACK_POS = .05      # m, same as replay.py segments
EEF_TRACK_ROT = .25      # rad


def make_timeline(t, qs, speed):
    """Nominal time of each planned frame: source pace / speed, slowed where joints would exceed the cap."""
    dsrc = np.diff(t)/speed
    dq = np.max(np.abs(np.diff(qs[:, :14], axis=0)), axis=1)/SERVO_SPEED_LIMIT
    return np.r_[0., np.cumsum(np.maximum(dsrc, dq))]


def warp(w, total):
    """Wall time -> nominal time with smoothstep velocity ramps at both ends.

    Each ramp adds RAMP_S/2 of wall time; wall duration = total + RAMP_S.
    """
    wall = total+RAMP_S
    def ramp(x):
        x = min(max(x, 0.), RAMP_S)/RAMP_S
        return RAMP_S*(x**3-x**4/2)
    if w <= RAMP_S:
        return ramp(w)
    if w >= wall-RAMP_S:
        return total-ramp(wall-w)
    return w-RAMP_S/2


def interp_rows(T, rows, tau):
    i = int(np.clip(np.searchsorted(T, tau, side='right')-1, 0, len(T)-2))
    a = (tau-T[i])/max(T[i+1]-T[i], 1e-12)
    a = min(max(a, 0.), 1.)
    return rows[i]+a*(rows[i+1]-rows[i])


class QuatTrack:
    """xyzw (model) -> wxyz (SDK), keeping each arm's sign continuous."""
    def __init__(self):
        self.last = [None, None]

    def __call__(self, poses):
        out = []
        for i in range(2):
            x, y, z, w = poses[i, 3:]
            q = np.array([w, x, y, z])
            if self.last[i] is not None and np.dot(q, self.last[i]) < 0:
                q = -q
            self.last[i] = q
            out.append(np.r_[poses[i, :3], q])
        return out


def servo_transport(Transport):
    class ServoTransport(Transport):
        def arm_servop(self):
            self.healthy()
            require_status(self.info)
            self.backend = 'supervised_servop'
            self.lease_at = time.monotonic()
            self.armed = True

        def send(self, title, data, guid=None):
            if title != 'request_servop':
                return super().send(title, data, guid)
            import json
            import uuid
            for key in ('left_pos', 'right_pos'):
                v = np.asarray(data[key], dtype=float)
                if v.shape != (7,) or not np.isfinite(v).all() or abs(np.linalg.norm(v[3:])-1) > 1e-3:
                    raise ValueError('ServoP %s must be finite xyz + unit wxyz' % key)
            with self.write_lock:
                if not self.armed or self.backend != 'supervised_servop' or self.fault or self.closed.is_set():
                    raise RuntimeError('ServoP transport is not healthy and armed')
                self.ws.send(json.dumps({'accid': self.info['accid'], 'title': title,
                                         'timestamp': int(time.time()*1000), 'guid': guid or uuid.uuid4().hex,
                                         'data': {'left_pos': list(map(float, data['left_pos'])),
                                                  'right_pos': list(map(float, data['right_pos']))}},
                                        allow_nan=False))
    return ServoTransport


def stream(session, T, t, qs, rate):
    """Continuous ServoP execution of the planned dense path."""
    robot, model = session.robot, session.model
    total = float(T[-1])
    wall = total+RAMP_S
    quat = QuatTrack()
    model.set_q(qs[0])
    quat(model.poses())  # seed quaternion signs
    q, dq = robot.healthy()
    start_offset = float(np.max(abs(q[:14]-qs[0, :14])))
    if start_offset > .02 or np.max(abs(dq)) > .02:
        detail = 'ServoP start is %.5f rad from frame zero' % start_offset
        if not findings.permit('servop_start_off_plan', detail, value=start_offset):
            raise RuntimeError('Robot is not stationary at frame zero')
    stats = dict(commands_sent=0, max_eef_error_m=0., max_eef_error_rad=0., max_joint_dev_rad=0.,
                 max_joint_speed_rad_s=0., late_ticks=0, takeover_drop_m=None)
    takeover = []
    period = 1./rate
    robot.arm_servop()
    t0 = time.monotonic()
    last_tick, n = -1., 0
    try:
        while True:
            now = time.monotonic()
            w = now-t0
            if w >= wall+HOLD_END_S:
                break
            tau = warp(min(w, wall), total)
            q_ref = interp_rows(T, qs, tau)
            source = float(np.interp(tau, T, t))
            model.set_q(q_ref)
            left, right = quat(model.poses())
            if now-last_tick >= TICK_PERIOD_S:
                _, desired_hands = sample(session.t, session.eef, session.hand_data, source)
                q, dq, _, _ = session.tick(desired_hands, q_ref, q_ref, source)
                last_tick = now
                fb = robot.pose
                measured = np.array([fb[s+'_position']+fb[s+'_quat'][1:]+fb[s+'_quat'][:1] for s in SIDES])
                model.set_q(q_ref)
                ep, er = pose_error(measured, model.poses())
                if w < 1.:
                    takeover.append(float(max(ep)))
                stats['max_eef_error_m'] = max(stats['max_eef_error_m'], float(max(ep)))
                stats['max_eef_error_rad'] = max(stats['max_eef_error_rad'], float(max(er)))
                stats['max_joint_dev_rad'] = max(stats['max_joint_dev_rad'], float(np.max(abs(q[:14]-q_ref[:14]))))
                stats['max_joint_speed_rad_s'] = max(stats['max_joint_speed_rad_s'], float(np.max(abs(dq[:14]))))
                if max(ep) > EEF_TRACK_POS or max(er) > EEF_TRACK_ROT:
                    detail = ('ServoP EEF tracking error %.4f m / %.4f rad at source %.3fs'
                              % (max(ep), max(er), source))
                    if not findings.permit('eef_tracking_error', detail, value=max(ep), source_time_s=source):
                        raise RuntimeError(detail)
            robot.send('request_servop', {'left_pos': left, 'right_pos': right})
            robot.lease_at = time.monotonic()
            stats['commands_sent'] += 1
            robot.commands_sent += 1
            n += 1
            nxt = t0+n*period
            if time.monotonic() > nxt+period:
                stats['late_ticks'] += 1
                n = int((time.monotonic()-t0)/period)  # no catch-up burst
                continue
            while time.monotonic() < nxt-.001:
                time.sleep(.0005)
            while time.monotonic() < nxt:
                pass
    finally:
        robot.armed = False
        stats['wall_duration_s'] = time.monotonic()-t0
        stats['takeover_drop_m'] = max(takeover) if takeover else None
    return stats


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
        qs, idx, durations, plan = base.make_plan(model, t, eef, hand_data, measured_q, meta, args.speed, out)
        # ServoP follows the dense joint path, not the waypoint chords: certify it directly.
        print('Checking swept clearance of the dense %d-frame ServoP path' % len(qs), flush=True)
        result['dense_collision'] = base.certify_or_record(model, qs, 'dense ServoP path')
        T = make_timeline(t, qs, args.speed)
        plan.update(backend='EEF -> continuous seeded IK -> dense joint path -> FK -> continuous ServoP',
                    servo_rate_hz=args.rate, servo_speed_limit_rad_s=SERVO_SPEED_LIMIT,
                    servo_motion_duration_s=float(T[-1]+RAMP_S), servo_ramp_s=RAMP_S,
                    dense_collision=result['dense_collision'])
        base.save(out/'plan.json', plan)
        np.savez_compressed(out/'plan_servop.npz', t=t, q16=qs, nominal_time_s=T)
        result['plan'] = plan
        print('PLAN READY source=%.2fs frames=%d servop motion=%.1fs (MoveJ version: %.1fs)'
              % (t[-1], len(t), T[-1]+RAMP_S, sum(durations)), flush=True)
        if not (args.check or args.execute):
            result['offline_validated'] = True
            return
        commands = process_commands()
        owners = ('wuji_teleop.switch', 'wuji-teleop-switch', 'tools/manus_teleop.sh',
                  'robot_home.py', 'axis_replay.py', 'replay.py')
        if any(owner in c for owner in owners for c in commands if 'replay_servop.py' not in c):
            raise RuntimeError('Exit teleop/home/replay before ServoP replay; another controller owns the robot')
        robot = servo_transport(Transport)(model.lower[:14], model.upper[:14])
        state = robot.snapshot()
        base.save(out/'start_state.json', state)
        try:
            require_status(state['robot_info'])
        except RuntimeError as exc:
            if not findings.permit('robot_status_unexpected', str(exc)):
                raise
        q0 = np.array(state['q16'])
        qs[:, 14:] = q0[14:]
        result['live_head_collision'] = base.certify_or_record(model, qs, 'live head recertification (dense)')
        model.set_q(q0)
        ep, er = pose_error(model.poses(), np.array(state['eef_base_xyz_xyzw']))
        if max(ep) > .001 or max(er) > .0035:
            detail = 'Live model FK and SDK EEF feedback differ by %.5f m / %.5f rad' % (max(ep), max(er))
            if not findings.permit('live_fk_disagrees', detail, value=max(ep)):
                raise RuntimeError('Live model FK and SDK EEF feedback disagree')
        result['arm_approach'] = base.certify_or_record(model, np.array([q0, qs[0]]), 'first-frame approach')
        approach_seconds = max(20., base.timing(q0, qs[0], 0., 0., 1.))
        result['arm_approach_duration_s'] = approach_seconds
        import rclpy
        rclpy.init()
        ros_initialized = True
        target = {s+'_hand': dict(zip(hand_names(s), hand_data[0, i])) for i, s in enumerate(SIDES)}
        hands = Hands(target)
        existing = any('wuji-hand-2-driver' in c or 'wuji-control' in c for c in commands)
        if args.execute and not existing:
            stack_log = (out/'stack.log').open('w')
            stack = subprocess.Popen(['ros2', 'launch', str(ROOT/'tools/home_bimanual.launch.py')],
                                     stdout=stack_log, stderr=subprocess.STDOUT, start_new_session=True)
        current_hands = np.array([hands.feedback(timeout=20 if stack else 5)[s] for s in SIDES])
        hands.validate_owner()
        for s in SIDES:
            if hands.node.count_publishers('/wuji/'+s+'/command') != 1:
                raise RuntimeError('Expected exactly one wuji_control output publisher for '+s)
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
        session = base.Session(robot, hands, model, t, eef, hand_data, out)
        hand_approach_seconds = max(10., 1.875*float(np.max(abs(hand_data[0]-current_hands)))/.3)
        print('MOVING FIRST FRAME hands duration=%.1fs' % hand_approach_seconds, flush=True)
        session.approach_hands(hand_approach_seconds, q0)
        if np.max(abs(q0[:14]-qs[0, :14])) > .002:
            print('MOVING FIRST FRAME arms MoveJ duration=%.1fs' % approach_seconds, flush=True)
            result['approach_execution'] = base.execute_approach(session, q0, qs[0], approach_seconds)
        print('REACHED FIRST FRAME', flush=True)
        print('SERVOP REPLAY START %.1fs at %.0f Hz' % (T[-1]+RAMP_S, args.rate), flush=True)
        result['servop'] = stream(session, T, t, qs, args.rate)
        print('SERVOP REPLAY DONE', result['servop'], flush=True)
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
                base.stop_stack(stack)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--episode', type=Path, default=base.DEFAULT_EPISODE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Offline plan + read-only live preflight')
    mode.add_argument('--execute', action='store_true', help='Approach first frame and execute on hardware')
    parser.add_argument('--speed', type=float, default=.5, help='Maximum source-time speed; joint cap can slow it further')
    parser.add_argument('--rate', type=float, default=300., help='ServoP command rate Hz')
    parser.add_argument('--sample-hz', type=float, default=50.)
    parser.add_argument('--start-seconds', type=float, default=0.)
    parser.add_argument('--end-seconds', type=float)
    parser.add_argument('--permissive', action='store_true',
                        help='Record findings instead of aborting (same meaning as replay.py)')
    args = parser.parse_args()
    if not 50 <= args.rate <= 500:
        parser.error('--rate must be within [50, 500]')
    out = ROOT/'replay/runs'/('servop_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
    out.mkdir(parents=True)
    result = {'completed': False, 'execute_requested': args.execute, 'backend': 'servop',
              'permissive': args.permissive, 'motion_commands_sent': 0}
    def interrupt(signum, frame):
        raise KeyboardInterrupt()
    old = signal.signal(signal.SIGTERM, interrupt)
    lock = open('/home/guest/wanchi/replay_tip_align_test_1030/.hands_replay.lock', 'a')
    if args.permissive:
        findings.begin(out/'violations.jsonl')
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
        base.save(out/'result.json', result)
        print('REPORT', out/'result.json', flush=True)
    return 0 if result['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
