"""Bounded, model-checked IK recovery for a mapped dual-arm replay file."""
from dataclasses import dataclass
import math
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class Limits:
    position_m: float
    rotation_rad: float
    gap_s: float

    def __post_init__(self):
        if not all(math.isfinite(x) and x > 0 for x in
                   (self.position_m, self.rotation_rad, self.gap_s)):
            raise ValueError('Recovery limits must be finite and positive')
        if self.rotation_rad > math.pi:
            raise ValueError('Recovery rotation cannot exceed 180 degrees')


class Reference:
    def __init__(self, t, eef):
        self.t = t
        self.eef = eef
        self.rotation = [Slerp(t, Rotation.from_quat(eef[:, side, 3:])) for side in (0, 1)]

    def interval(self, start, end):
        knots = self.t[(self.t > start) & (self.t < end)]
        times = np.unique(np.r_[np.linspace(start, end, 11), knots])
        poses = np.empty((len(times), 2, 7))
        for side in (0, 1):
            for axis in range(3):
                poses[:, side, axis] = np.interp(times, self.t, self.eef[:, side, axis])
            poses[:, side, 3:] = self.rotation[side](times).as_quat()
        return (times - start) / (end - start), poses


class RecoveryPlanner:
    # make_plan retries jumps above 0.10 rad and must preserve this path.
    max_joint_step = .10

    def __init__(self, t, goals, hands, reference, model, solve, correct,
                 error, workspace, sweep, nominal, recovery):
        self.t, self.goals, self.hands = t, goals, hands
        self.model, self.solve, self.correct = model, solve, correct
        self.error, self.workspace, self.sweep = error, workspace, sweep
        self.nominal, self.recovery = nominal, recovery
        self.intervals = [reference.interval(a, b) for a, b in zip(t[:-1], t[1:])]

    def fk(self, q):
        if (q.shape != (16,) or not np.isfinite(q).all()
                or np.any(q < self.model.lower + 1e-5)
                or np.any(q > self.model.upper - 1e-5)):
            raise ValueError('Joint solution exceeds executable bounds')
        self.model.set_q(q)
        actual = self.model.poses().copy()
        self.workspace(actual)
        return actual

    def check_frame(self, q, goal, position_m, rotation_rad):
        p, r = self.error(self.fk(q), goal)
        if (not np.isfinite(p).all() or not np.isfinite(r).all()
                or max(p) > position_m or max(r) > rotation_rad):
            raise ValueError('Frame residual %.2f mm / %.2f deg' %
                             (1000 * max(p), math.degrees(max(r))))

    def check_section(self, start, path, position_m, rotation_rad):
        if np.max(np.abs(np.diff(path, axis=0))) > self.max_joint_step:
            raise ValueError('Joint step exceeds %.2f rad' % self.max_joint_step)
        peak_p = np.zeros(2)
        peak_r = np.zeros(2)
        for index, (a, b) in enumerate(zip(path[:-1], path[1:]), start):
            alphas, targets = self.intervals[index]
            for alpha, target in zip(alphas, targets):
                p, r = self.error(self.fk(a + alpha * (b - a)), target)
                if not np.isfinite(p).all() or not np.isfinite(r).all():
                    raise ValueError('Nonfinite interpolated pose error')
                peak_p = np.maximum(peak_p, p)
                peak_r = np.maximum(peak_r, r)
        if max(peak_p) > position_m or max(peak_r) > rotation_rad:
            raise ValueError('Interpolated residual %.2f mm / %.2f deg' %
                             (1000 * max(peak_p), math.degrees(max(peak_r))))
        return dict(position_mm=(1000 * peak_p).tolist(),
                    rotation_deg=np.rad2deg(peak_r).tolist())

    def solve_at(self, seed, index):
        q = self.solve(seed.copy(), self.goals[index])
        q[14:] = seed[14:]
        self.check_frame(q, self.goals[index], *self.nominal[:2])
        return q

    def bridge(self, start, q_start):
        stop = np.searchsorted(self.t, self.t[start] + self.recovery.gap_s + 1e-9, side='right')
        last_error = 'no subsequent nominal IK frame in the configured window'
        for end in range(start + 1, stop):
            try:
                q_end = self.solve_at(q_start, end)
            except ValueError:
                continue
            alpha = (self.t[start:end + 1] - self.t[start]) / (self.t[end] - self.t[start])
            chord = q_start + alpha[:, None] * (q_end - q_start)
            chord[-1] = q_end
            for method in ('joint_chord', 'pose_adjusted_chord'):
                try:
                    candidate = chord.copy()
                    if method == 'pose_adjusted_chord':
                        if end - start < 2:
                            continue
                        for index in range(start + 1, end):
                            row = index - start
                            candidate[row] = self.correct(chord[row].copy(), self.goals[index])
                            candidate[row, 14:] = q_start[14:]
                    peak = self.check_section(start, candidate, self.recovery.position_m,
                                              self.recovery.rotation_rad)
                    clearance = self.sweep(candidate, self.hands[start:end + 1])
                except ValueError as exc:
                    last_error = str(exc)
                    continue
                return end, candidate, dict(method=method, start_s=float(self.t[start]),
                                            end_s=float(self.t[end]), peak_error=peak,
                                            clearance=clearance)
        raise ValueError('No recovery from %.3f s within %.3f s: %s' %
                         (self.t[start], self.recovery.gap_s, last_error))

    def sections(self, first):
        """Yield pose-checked spans. Certify the whole path before execution."""
        index, q = 0, first.copy()
        while index + 1 < len(self.t):
            try:
                following = self.solve_at(q, index + 1)
                path = np.stack([q, following])
                peak = self.check_section(index, path, self.nominal[0], self.nominal[2])
                end, detail = index + 1, None
            except ValueError as exc:
                end, path, detail = self.bridge(index, q)
                detail['trigger'] = str(exc)
                peak = detail['peak_error']
            yield index, end, path, peak, detail
            index, q = end, path[-1].copy()


def load_recovered_episode(args, root, out, result):
    """Plan before device connection. The caller still runs full replay preflight."""
    import replay as base
    from clearance import certify_path
    from episode import pose_error
    from episode_cross import load_cross_episode
    from hand_body_clearance import HandBodyClearance
    from movej_prepare import check_servop_box
    from Pinocchio.pin_model import PinArms
    from planner_cross import candidates
    from transport import replay_backend

    began = time.monotonic()
    limits = Limits(args.recover_position_mm / 1000., math.radians(args.recover_rotation_deg),
                    args.recover_max_gap_s)
    report = dict(status='planning', limits=dict(position_mm=args.recover_position_mm,
                  rotation_deg=args.recover_rotation_deg, max_gap_s=limits.gap_s), bridges=[])
    result['ik_recovery'] = report
    t, goals, hands, meta = load_cross_episode(args.episode, 50.)
    if meta.get('placement_policy') != 'preserve_common_frame_v1':
        raise ValueError('IK recovery expects a common-frame TRON2 replay input')
    with np.load(args.episode, allow_pickle=False) as original:
        reference = Reference(original['t'].astype(float), original['eef'].astype(float))
    _, Model, _ = replay_backend()
    model = Model()
    pin = PinArms()
    pin.lower = np.maximum(pin.lower, model.lower)
    pin.upper = np.minimum(pin.upper, model.upper)
    hand_model = HandBodyClearance(root/'replay_robot/models/tron2_wuji_board_v1_20261110/robot.urdf')

    def solve(seed, target):
        q, _ = pin.solve(seed, target, max_iter=45)
        return q

    def correct(seed, target):
        q = seed.copy()
        for side in (0, 1):
            sl = slice(7 * side, 7 * side + 7)
            q[sl] = pin._bounded(q, side, target, rot_weight=limits.position_m / limits.rotation_rad)
        return q

    def sweep(path, fingers):
        return dict(arms=certify_path(model, path), hands=hand_model.check_path(path, fingers))

    planner = RecoveryPlanner(
        t, goals, hands, reference, model, solve, correct, pose_error, check_servop_box, sweep,
        (base.IK_POSITION_TOL, base.IK_ROTATION_TOL, base.CHORD_ROTATION_TOL), limits)
    starts = []
    report['first_frame_attempts'] = []
    for name, seed in candidates(root, model):
        try:
            q = base.solve_eef(model, seed.copy(), goals[0])
            q[14:] = seed[14:]
            planner.check_frame(q, goals[0], .001, .0035)
            sweep(q[None], hands[:1])
            starts.append((name, q))
        except ValueError as exc:
            report['first_frame_attempts'].append(dict(seed=name, error=str(exc)))
    if not starts:
        report['status'] = 'failed'
        raise ValueError('IK recovery requires a valid first anchor at 0.000 s')

    for name, first in starts:
        qs = np.empty((len(t), 16))
        qs[0] = first
        peak_p, peak_r = np.zeros(2), np.zeros(2)
        bridges = []
        try:
            for start, end, path, peak, detail in planner.sections(first):
                qs[start:end + 1] = path
                peak_p = np.maximum(peak_p, peak['position_mm'])
                peak_r = np.maximum(peak_r, peak['rotation_deg'])
                if detail:
                    bridges.append(detail)
            clearance = sweep(qs, hands)
        except ValueError as exc:
            report['first_frame_attempts'].append(dict(seed=name, error=str(exc), bridges=bridges))
            continue
        commanded = np.asarray([planner.fk(q) for q in qs])
        report.update(status='recovered', first_seed=name, bridges=bridges, frames=len(t),
                      peak_error=dict(position_mm=peak_p.tolist(), rotation_deg=peak_r.tolist()),
                      clearance=clearance, planning_seconds=time.monotonic() - began)
        np.savez_compressed(out/'ik_recovery_path.npz', t=t, original_eef=goals,
                            planned_eef=commanded, q16=qs, hands=hands)
        meta['ik_recovery'] = report
        return t, commanded, hands, qs, meta
    report['status'] = 'failed'
    raise ValueError('IK recovery failed: ' + report['first_frame_attempts'][-1]['error'])
