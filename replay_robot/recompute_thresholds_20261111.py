"""Offline re-evaluation of original task-mapped episodes with current policy.

Reuses saved solutions; retries failed frames with position/orientation residuals
weighted by the new tolerances. Computes FK, checked model distances and chord
errors without creating a robot transport, ROS node, or device command.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def intervals(t, mask):
    ids = np.flatnonzero(mask)
    if not len(ids):
        return []
    return [dict(first_frame=int(c[0]), last_frame=int(c[-1]), start_s=float(t[c[0]]),
                 end_s=float(t[c[-1]]), frames=len(c))
            for c in np.split(ids, np.flatnonzero(np.diff(ids) > 1) + 1)]


def run(root, episode, policy):
    began = time.monotonic()
    sys.path[:0] = [str(root/'replay'), str(root/'replay_hand'), str(root/'tools'), str(root/'Pinocchio')]
    from pin_model import PinArms
    import replay_thresholds as thresholds
    from transport import replay_backend
    import mujoco
    pos_tol, rot_tol = thresholds.IK_POSITION_TOL, thresholds.IK_ROTATION_TOL
    chord_tol, clearance_tol = thresholds.CHORD_ROTATION_TOL, thresholds.MODEL_CLEARANCE_M
    source = root/'replay_robot/task_transfer_20261110/episodes'/episode/'replay_input.npz'
    baseline = (root/'replay_robot/task_transfer_20261110/episodes'/episode/'ik_diagnostics_not_for_execution.npz'
                if policy == 'movej' else root/'replay_robot/limit_policy_comparison_20261110'/episode/'diagnostics_not_for_execution.npz')
    out = root/'replay_robot/threshold_recalculation_20261111'/episode/policy
    if out.exists():
        raise FileExistsError('Refusing to overwrite '+str(out))
    out.mkdir(parents=True)
    sources = [source, baseline, Path(__file__).resolve(), Path(thresholds.__file__),
               root/'Pinocchio/pin_model.py', root/'record/config/model/robot.urdf',
               root/'replay/transport.py', root/'replay_hand/hand_body_clearance.py']
    hashes = {str(p.relative_to(root)): sha(p) for p in sources}
    with np.load(source, allow_pickle=False) as z:
        t, goals, hands = z['t'].copy(), z['eef'].copy(), z['hands'].copy()
    with np.load(baseline, allow_pickle=False) as z:
        qs = z['diagnostic_q16'].copy()
        previous_pass = z['pose_pass'].copy()
    solver = PinArms()
    if policy == 'urdf':
        solver.lower = solver.model.lowerPositionLimit[solver.iq].copy()
        solver.upper = solver.model.upperPositionLimit[solver.iq].copy()
    goal_rot = [Rotation.from_quat(goals[:, arm, 3:]) for arm in range(2)]

    def errors(q, k):
        p, r = solver.fk(q)
        return (np.linalg.norm(p-goals[k, :, :3], axis=1),
                np.array([(Rotation.from_matrix(r[a]).inv()*goal_rot[a][k]).magnitude() for a in range(2)]))

    def quality(p, r):
        return (p > pos_tol or r > rot_tol, max(p/pos_tol, r/rot_tol))

    before_error = np.asarray([errors(q, k) for k, q in enumerate(qs)])
    before_pass = (before_error[:, 0] <= pos_tol) & (before_error[:, 1] <= rot_tol)
    pos, rot = np.zeros((len(t), 2)), np.zeros((len(t), 2))
    retried = 0
    for k in range(len(t)):
        pe, re = errors(qs[k], k)
        for a in range(2):
            sl = slice(7*a, 7*a+7)
            needs_retry = pe[a] > pos_tol or re[a] > rot_tol
            discontinuous = k > 0 and np.max(abs(qs[k, sl]-qs[k-1, sl])) > .20
            if needs_retry or discontinuous:
                seeds = [qs[k].copy()] if needs_retry else []
                if k and (pos[k-1, a] <= pos_tol and rot[k-1, a] <= rot_tol):
                    seed = qs[k].copy(); seed[sl] = qs[k-1, sl]; seeds.append(seed)
                for seed in seeds:
                    trial = np.clip(seed, solver.lower+1e-5, solver.upper-1e-5)
                    trial[sl] = solver._bounded(trial, a, goals[k], rot_weight=pos_tol/rot_tol)
                    tp, tr = errors(trial, k); retried += 1
                    better = quality(tp[a], tr[a]) < quality(pe[a], re[a])
                    continuous_alternative = (k > 0 and not quality(tp[a], tr[a])[0]
                        and np.max(abs(qs[k, sl]-qs[k-1, sl])) > .20
                        and np.max(abs(trial[sl]-qs[k-1, sl])) <= .20)
                    if better or continuous_alternative:
                        qs[k, sl] = trial[sl]; pe, re = errors(qs[k], k)
        pos[k], rot[k] = pe, re
        if k % 200 == 0:
            print(episode, policy, 'IK', k, '/', len(t), flush=True)
    valid = (pos <= pos_tol) & (rot <= rot_tol)
    assert np.all(valid[before_pass])
    assert np.all(qs >= solver.lower-1e-9) and np.all(qs <= solver.upper+1e-9)

    # All checked arm/body/head distances use the SAME 20 mm requirement.
    _, Model, _ = replay_backend(); model = Model()
    wrist_pairs = []
    for arm in 'LR':
        wanted = frozenset(['wrist_yaw_'+arm+'_Link', 'wrist_roll_'+arm+'_Link'])
        wrist_pairs.append([pair for pair in model.pairs if frozenset(
            model.body_name(model.model.geom_bodyid[g]) for g in pair) == wanted])
    assert all(wrist_pairs)
    clearance = np.empty(len(t)); wrist_distance = np.empty((len(t), 2)); closest = []
    hand_clearance = np.full(len(t), np.nan); hand_checked = np.zeros(len(t), bool)
    checker = None
    for k, q in enumerate(qs):
        model.set_q(q)
        clearance[k], pair = model.clearance(); closest.append(pair)
        for a, pairs in enumerate(wrist_pairs):
            wrist_distance[k, a] = min(float(mujoco.mj_geomDistance(model.model, model.data, g, h, .5, None))
                                       for g, h in pairs)
        if clearance[k] >= clearance_tol:
            if checker is None:
                from hand_body_clearance import HandBodyClearance
                checker = HandBodyClearance(root/'replay_robot/models/tron2_wuji_board_v1_20261110/robot.urdf')
            checker.set_q(q, hands[k]); hand_clearance[k], _ = checker.clearance(); hand_checked[k] = True
        if k % 300 == 0:
            print(episode, policy, 'collision', k, '/', len(t), flush=True)
    collision_valid = (clearance >= clearance_tol) & hand_checked & (hand_clearance >= clearance_tol)
    combined = valid.all(axis=1) & collision_valid

    # Nine interior samples per original 30 Hz interval: nominally 300 Hz.
    # Retain separate masks: pose intervals are not certified collision paths.
    chord_pos, chord_rot = np.full((len(t)-1, 2), np.nan), np.full((len(t)-1, 2), np.nan)
    for k in range(len(t)-1):
        arms = valid[k] & valid[k+1]
        if not np.any(arms):
            continue
        alphas = np.linspace(0., 1., 11)[1:-1]
        interp_rot = [Slerp([0., 1.], Rotation.from_quat(goals[k:k+2, a, 3:]))(alphas) for a in range(2)]
        pe, re = [], []
        for j, alpha in enumerate(alphas):
            p, r = solver.fk(qs[k]+alpha*(qs[k+1]-qs[k]))
            xyz = goals[k, :, :3]+alpha*(goals[k+1, :, :3]-goals[k, :, :3])
            pe.append(np.linalg.norm(p-xyz, axis=1))
            re.append([(Rotation.from_matrix(r[a]).inv()*interp_rot[a][j]).magnitude() for a in range(2)])
        chord_pos[k, arms] = np.max(pe, axis=0)[arms]
        chord_rot[k, arms] = np.max(re, axis=0)[arms]
    eligible_chord = valid[:-1] & valid[1:]
    chord_fail = eligible_chord & ((chord_pos > pos_tol) | (chord_rot > chord_tol))
    steps = np.max(abs(np.diff(qs[:, :14], axis=0)).reshape(-1, 2, 7), axis=2)
    # A sampled collision failure already rejects a full path. Only a fully
    # eligible path would warrant swept certification; don't certify bad IK.
    worst = int(np.argmin(clearance))
    report = dict(episode=episode, policy=policy, execution_ready=False, motion_commands_sent=0,
        thresholds=dict(position_m=pos_tol, orientation_deg=float(np.rad2deg(rot_tol)),
                        chord_orientation_deg=float(np.rad2deg(chord_tol)), model_clearance_m=clearance_tol,
                        wrist_internal_exception=False),
        source_frames=len(t), duration_s=float(t[-1]),
        method='Retain known passing baseline solutions; retry failures with tolerance-normalized bounded IK; retry branch steps from previous passing frame',
        unchanged_original_targets=True, unchanged_original_times=True,
        prior_policy_passing_frames=previous_pass.sum(axis=0).tolist(),
        new_thresholds_on_previous_solutions=before_pass.sum(axis=0).tolist(),
        recomputed_passing_frames=valid.sum(axis=0).tolist(),
        both_pose_passing_frames=int(valid.all(axis=1).sum()),
        left_pose_intervals=intervals(t, valid[:, 0]), right_pose_intervals=intervals(t, valid[:, 1]),
        both_pose_intervals=intervals(t, valid.all(axis=1)),
        first_position_error_mm=(pos[0]*1000).tolist(), first_orientation_error_deg=np.rad2deg(rot[0]).tolist(),
        bounded_retries=retried,
        collision=dict(arm_body_passing_frames=int((clearance >= clearance_tol).sum()),
                       hand_body_checked_frames=int(hand_checked.sum()),
                       skipped_hand_body_reason='Already rejected by arm/body/head clearance',
                       all_model_passing_frames=int(collision_valid.sum()),
                       both_pose_and_model_passing_frames=int(combined.sum()),
                       both_pose_and_model_intervals=intervals(t, combined),
                       minimum_arm_body_mm=float(clearance[worst]*1000), minimum_pair=closest[worst],
                       minimum_time_s=float(t[worst]),
                       left_internal_wrist_distance_range_mm=(wrist_distance[:, 0][[int(np.argmin(wrist_distance[:, 0])), int(np.argmax(wrist_distance[:, 0]))]]*1000).tolist(),
                       right_internal_wrist_distance_range_mm=(wrist_distance[:, 1][[int(np.argmin(wrist_distance[:, 1])), int(np.argmax(wrist_distance[:, 1]))]]*1000).tolist(),
                       internal_wrist_under_20mm_frames=(wrist_distance < clearance_tol).sum(axis=0).tolist(),
                       scope='Sampled all arm/body/head checked pairs; hand/body checked only if arm/body passes. No environment/cables/adapter/within-hand certification'),
        chord=dict(nominal_hz=300, tested_intervals_per_arm=eligible_chord.sum(axis=0).tolist(),
                   failed_intervals_per_arm=chord_fail.sum(axis=0).tolist(),
                   joint_step_over_0p2rad_in_pose_passing_intervals=(eligible_chord & (steps > .2)).sum(axis=0).tolist(),
                   max_position_mm=[float(np.nanmax(chord_pos[:, a])*1000) if eligible_chord[:, a].any() else None for a in range(2)],
                   max_rotation_deg=[float(np.rad2deg(np.nanmax(chord_rot[:, a]))) if eligible_chord[:, a].any() else None for a in range(2)]),
        source_sha256=hashes, elapsed_seconds=time.monotonic()-began,
        not_checked=['Hardware/ServoP joint envelope', 'Actual MoveJ entry and takeover',
                     'Full swept collision certification', 'Dynamics and final retiming'],
        solver_failure_is_not_global_unreachability_proof=True)
    assert all(sha(root/name) == digest for name, digest in hashes.items())
    np.savez_compressed(out/'diagnostics_not_for_execution.npz', t=t, diagnostic_q16=qs,
                        position_error_m=pos, rotation_error_rad=rot, pose_pass=valid,
                        arm_body_clearance_m=clearance, internal_wrist_distance_m=wrist_distance,
                        hand_body_clearance_m=hand_clearance, hand_body_checked=hand_checked,
                        collision_pass=collision_valid, combined_pass=combined,
                        chord_position_error_m=chord_pos, chord_rotation_error_rad=chord_rot)
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print('RESULT', episode, policy, json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--episode', required=True, choices=['episode594', 'episode615', 'episode616'])
    p.add_argument('--policy', required=True, choices=['movej', 'urdf'])
    args = p.parse_args(); run(args.root, args.episode, args.policy)
