"""File-only IK experiment for the episode616 right-arm 0.2 second gap.

Reuses PinArms without loading any robot control/transport entry point. This
retains the earlier URDF-only hypothesis, not a verified ServoP envelope.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def intervals(t, mask):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    chunks = np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1)
    return [dict(first_frame=int(c[0]), last_frame=int(c[-1]),
                 start_s=float(t[c[0]]), end_s=float(t[c[-1]]), frames=len(c))
            for c in chunks]


def smoothstep(x):
    return x ** 3 * (10 - 15 * x + 6 * x ** 2)


def main(root):
    sys.path.insert(0, str(root / 'Pinocchio'))
    from pin_model import PinArms, Q16

    src = root / 'replay_robot/task_transfer_20261110/episodes/episode616/replay_input.npz'
    old = root / 'replay_robot/limit_policy_comparison_20261110/episode616/diagnostics_not_for_execution.npz'
    out = root / 'replay_robot/episode616_gap_optimization_20261110'
    if out.exists():
        raise FileExistsError('Refusing to overwrite ' + str(out))
    sources = [src, old, root / 'Pinocchio/pin_model.py',
               root / 'record/config/model/robot.urdf', Path(__file__).resolve()]
    hashes = {str(p.relative_to(root)): sha(p) for p in sources}
    with np.load(src, allow_pickle=False) as z:
        t, goals = z['t'].copy(), z['eef'].copy()
    with np.load(old, allow_pickle=False) as z:
        baseline = z['diagnostic_q16'].copy()
        old_pass = z['pose_pass'][:, 1].copy()
        old_pos = z['position_error_m'][:, 1].copy()
        old_rot = z['rotation_error_rad'][:, 1].copy()
    solver = PinArms()
    movej_lower, movej_upper = solver.lower.copy(), solver.upper.copy()
    solver.lower = solver.model.lowerPositionLimit[solver.iq].copy()
    solver.upper = solver.model.upperPositionLimit[solver.iq].copy()
    target_rot = Rotation.from_quat(goals[:, 1, 3:])

    def errors(q, xyz, rot):
        p, r = solver.fk(q)
        return (float(np.linalg.norm(p[1] - xyz)),
                float((Rotation.from_matrix(r[1]).inv() * rot).magnitude()))

    # Keep both window boundaries exactly unchanged. Taper the objective weight
    # over neighboring valid frames so that the local solution joins smoothly.
    a, b = 1077, 1092
    weights = np.full(len(t), .3)
    for k in range(a + 1, b):
        amount = min(1., (k - a) / 4., (b - k) / 5.)
        weights[k] = .3 + .2 * smoothstep(amount)
    qs = baseline.copy()
    for k in range(a + 1, b):
        seed = qs[k - 1].copy()
        qs[k, 7:14] = solver._bounded(seed, 1, goals[k], rot_weight=weights[k])
    np.testing.assert_array_equal(qs[:a + 1], baseline[:a + 1])
    np.testing.assert_array_equal(qs[b:], baseline[b:])
    np.testing.assert_array_equal(qs[:, :7], baseline[:, :7])
    np.testing.assert_array_equal(qs[:, 14:], baseline[:, 14:])
    assert np.all(qs[:, 7:14] >= solver.lower[7:14] - 1e-10)
    assert np.all(qs[:, 7:14] <= solver.upper[7:14] + 1e-10)
    pos, rot = np.asarray([errors(q, g[1, :3], r)
                           for q, g, r in zip(qs, goals, target_rot)]).T
    valid = (pos <= .010) & (rot <= .015)
    assert np.all(valid[old_pass])
    assert np.all(valid[1081:1088])
    merged = next(c for c in intervals(t, valid)
                  if c['first_frame'] <= 1081 and c['last_frame'] >= 1087)
    first, last = merged['first_frame'], merged['last_frame']

    # Densely evaluate linearly interpolated model joints against xyz+SLERP
    # original targets. This verifies sampled FK tracking, not ServoP dynamics.
    dense_t = np.concatenate([np.linspace(t[k], t[k + 1], 11)[:-1]
                              for k in range(first, last)] + [t[last:last + 1]])
    dense_q = np.column_stack([np.interp(dense_t, t, qs[:, j]) for j in range(16)])
    dense_xyz = np.column_stack([np.interp(dense_t, t, goals[:, 1, j]) for j in range(3)])
    dense_rot = Slerp(t, target_rot)(dense_t)
    dense_pos_err, dense_rot_err = np.asarray([
        errors(q, xyz, r) for q, xyz, r in zip(dense_q, dense_xyz, dense_rot)]).T
    dense_valid = (dense_pos_err <= .010) & (dense_rot_err <= .015)
    # Report, rather than hide, any interpolation failure outside the gap.
    gap_dense = (dense_t >= t[a]) & (dense_t <= t[b])
    assert np.all(dense_valid[gap_dense])
    speeds = np.diff(qs[first:last + 1, 7:14], axis=0) / np.diff(t[first:last + 1])[:, None]
    local_speed = np.diff(qs[a - 1:b + 2, 7:14], axis=0) / np.diff(t[a - 1:b + 2])[:, None]
    delta = goals[1087, 1, :3] - goals[1081, 1, :3]
    target_delta_rot = (target_rot[1081].inv() * target_rot[1087]).magnitude()
    outside_movej = np.any((qs[:, 7:14] < movej_lower[7:14]) |
                          (qs[:, 7:14] > movej_upper[7:14]), axis=1)
    frame_rows = [dict(frame=k, time_s=float(t[k]), weight=float(weights[k]),
                       before_position_mm=float(old_pos[k] * 1000),
                       before_rotation_deg=float(np.rad2deg(old_rot[k])),
                       after_position_mm=float(pos[k] * 1000),
                       after_rotation_deg=float(np.rad2deg(rot[k])),
                       passing=bool(valid[k])) for k in range(a, b + 1)]
    report = dict(
        episode='episode616', arm='right', motion_commands_sent=0,
        execution_ready=False, servop_limits_verified=False,
        live_limits_changed=False, source_targets_changed=False,
        source_timestamps_changed=False, thresholds_changed=False,
        scope='URDF-only offline FK/IK feasibility; not a hardware execution plan',
        method='Original goals; orientation weight .3 to .5 with quintic taper, forward warm starts',
        error_thresholds=dict(position_m=.010, rotation_rad=.015),
        window=dict(first_frame=a, last_frame=b, start_s=float(t[a]), end_s=float(t[b]),
                    boundary_joints_exactly_preserved=True),
        failed_original_frames=(np.flatnonzero(~old_pass[1081:1088]) + 1081).tolist(),
        original_target_motion_over_gap=dict(position_mm=float(np.linalg.norm(delta) * 1000),
                                              rotation_deg=float(np.rad2deg(target_delta_rot))),
        gap_max_before=dict(position_mm=float(old_pos[1081:1088].max() * 1000),
                            rotation_deg=float(np.rad2deg(old_rot[1081:1088].max()))),
        gap_max_after=dict(position_mm=float(pos[1081:1088].max() * 1000),
                           rotation_deg=float(np.rad2deg(rot[1081:1088].max()))),
        original_right_intervals=intervals(t, old_pass),
        optimized_right_intervals=intervals(t, valid), merged_interval=merged,
        original_passing_samples=int(old_pass.sum()), optimized_passing_samples=int(valid.sum()),
        dense_check=dict(samples=len(dense_t), nominal_hz=300,
                         passing_samples=int(dense_valid.sum()), all_pass=bool(dense_valid.all()),
                         max_position_mm=float(dense_pos_err.max() * 1000),
                         max_rotation_deg=float(np.rad2deg(dense_rot_err.max())),
                         window_all_pass=bool(dense_valid[gap_dense].all()),
                         window_max_position_mm=float(dense_pos_err[gap_dense].max() * 1000),
                         window_max_rotation_deg=float(np.rad2deg(dense_rot_err[gap_dense].max())),
                         interpretation='Linear joint interpolation vs original xyz+SLERP targets; sampled, not continuous certification'),
        continuity=dict(max_changed_joint_deg=float(np.rad2deg(abs(qs[:, 7:14] - baseline[:, 7:14]).max())),
                        max_local_joint_step_deg=float(np.rad2deg(abs(np.diff(qs[a - 1:b + 2, 7:14], axis=0)).max())),
                        max_local_joint_speed_rad_s=float(abs(local_speed).max()),
                        max_merged_interval_joint_speed_rad_s=float(abs(speeds).max())),
        optimized_window_outside_movej_frames=int(outside_movej[a:b + 1].sum()),
        joint_names=Q16[7:14],
        source_sha256=hashes, frame_details=frame_rows,
        not_checked=['ServoP controller joint envelope and closed-loop tracking',
                     'MoveJ preparation and transition', 'collisions and environment',
                     'torque/acceleration feasibility', 'left-arm feasibility'])
    assert all(sha(root / name) == digest for name, digest in hashes.items())
    out.mkdir(parents=True)
    np.savez_compressed(out / 'right_arm_diagnostics_not_for_execution.npz',
                        t=t, diagnostic_right_q7=qs[:, 7:14], target_right_eef=goals[:, 1],
                        position_error_m=pos, rotation_error_rad=rot, pose_pass=valid,
                        rotation_weight=weights, outside_movej=outside_movej)
    np.savez_compressed(out / 'dense_fk_check.npz', t=dense_t,
                        position_error_m=dense_pos_err, rotation_error_rad=dense_rot_err,
                        pose_pass=dense_valid)
    (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'frame_details'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    main(parser.parse_args().root)
