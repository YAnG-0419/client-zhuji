"""File/model-only episode102 diagnostics with current replay thresholds."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'replay_robot/episode102_review_20261111'
sys.path[:0] = [str(ROOT/'replay'), str(ROOT/'replay_hand'), str(ROOT/'tools'), str(ROOT/'Pinocchio'), str(ROOT/'replay_robot')]
from pin_model import PinArms
from planner_cross import saved_home
from replay_thresholds import (IK_POSITION_TOL, IK_ROTATION_TOL, CHORD_ROTATION_TOL,
                               MODEL_CLEARANCE_M, INTERNAL_WRIST_CLEARANCE_M)
from transport import replay_backend
from clearance import certify_path
from recompute_thresholds_20261111 import intervals


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def run(policy):
    began = time.monotonic(); out = WORK/policy
    assert not out.exists(), 'Refusing to overwrite diagnostic results'
    out.mkdir()
    source = WORK/'replay_input.npz'
    with np.load(source, allow_pickle=False) as z:
        t, targets, hands = z['t'].copy(), z['eef'].copy(), z['hands'].copy()
        meta = json.loads(str(z['metadata_json']))
    dependencies = [source, Path(__file__).resolve(), ROOT/'Pinocchio/pin_model.py',
                    ROOT/'replay/replay_thresholds.py', ROOT/'replay/transport.py',
                    ROOT/'replay/clearance.py', ROOT/'replay/replay_movej_path.py',
                    ROOT/'replay_hand/hand_body_clearance.py', ROOT/'record/config/model/robot.urdf',
                    ROOT/'replay_robot/models/tron2_wuji_board_v1_20261110/robot.urdf']
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in dependencies}
    solver = PinArms(); movej_lower, movej_upper = solver.lower.copy(), solver.upper.copy()
    if policy == 'urdf':
        solver.lower = solver.model.lowerPositionLimit[solver.iq].copy()
        solver.upper = solver.model.upperPositionLimit[solver.iq].copy()
    head = np.asarray(meta['target_head_pitch_yaw_rad'])
    home = saved_home(ROOT); home[14:] = head
    ready = np.r_[[0., 0., 0., -1.6, 0., 0., 0.]*2, head]
    middle = np.r_[(solver.lower[:14]+solver.upper[:14])/2, head]
    rng = np.random.default_rng(10220261111)
    seeds = [home, ready, middle] + [np.r_[rng.uniform(solver.lower[:14]+.001, solver.upper[:14]-.001), head] for _ in range(9)]
    target_rotation = [Rotation.from_quat(targets[:, a, 3:]) for a in range(2)]
    normalized_weight = IK_POSITION_TOL/IK_ROTATION_TOL

    def errors(q, k):
        p, r = solver.fk(q)
        return (np.linalg.norm(p-targets[k, :, :3], axis=1),
                np.array([(Rotation.from_matrix(r[a]).inv()*target_rotation[a][k]).magnitude() for a in range(2)]))

    def quality(p, r):
        return (p > IK_POSITION_TOL or r > IK_ROTATION_TOL, max(p/IK_POSITION_TOL, r/IK_ROTATION_TOL))

    candidates = []; attempts = []
    for j, seed in enumerate(seeds):
        for weight in ([normalized_weight, .3] if j < 3 else [normalized_weight]):
            q = np.clip(seed.copy(), solver.lower+1e-5, solver.upper-1e-5)
            for a in range(2):
                q[7*a:7*a+7] = solver._bounded(q, a, targets[0], rot_weight=weight)
            pe, re = errors(q, 0); candidates.append((q, pe, re))
            attempts.append(dict(seed=j, rotation_weight=weight, q16=q.tolist(), position_m=pe.tolist(), rotation_rad=re.tolist()))
    q = home.copy()
    for a in range(2):
        best = min(candidates, key=lambda row:quality(row[1][a], row[2][a]))
        q[7*a:7*a+7] = best[0][7*a:7*a+7]
    qs, positions, rotations = [], [], []
    for k in range(len(t)):
        if k:
            for a in range(2):
                previous = q.copy(); sl = slice(7*a, 7*a+7)
                q[sl] = solver._bounded(q, a, targets[k], rot_weight=normalized_weight)
                pe, re = errors(q, k)
                if quality(pe[a], re[a])[0]:
                    trial = previous.copy(); trial[sl] = solver._bounded(trial, a, targets[k], rot_weight=.3)
                    tp, tr = errors(trial, k)
                    if quality(tp[a], tr[a]) < quality(pe[a], re[a]):
                        q[sl] = trial[sl]
        pe, re = errors(q, k); qs.append(q.copy()); positions.append(pe); rotations.append(re)
        if k % 150 == 0:
            print(policy, 'IK', k, '/', len(t), flush=True)
    qs, positions, rotations = map(np.asarray, (qs, positions, rotations))
    valid = (positions <= IK_POSITION_TOL) & (rotations <= IK_ROTATION_TOL)
    assert np.all(qs >= solver.lower-1e-10) and np.all(qs <= solver.upper+1e-10)
    np.savez_compressed(out/'ik_diagnostics_not_for_execution.npz', t=t, diagnostic_q16=qs,
                        position_error_m=positions, rotation_error_rad=rotations, pose_pass=valid)

    _, Model, _ = replay_backend(); model = Model()
    if policy == 'urdf':
        model.lower, model.upper = model.model.jnt_range[model.jids].T.copy()
    score = np.empty(len(t)); raw_at_pair = np.empty(len(t)); closest = []
    hand_distance = np.full(len(t), np.nan); hand_checked = np.zeros(len(t), bool)
    checker = None; fk_difference = 0.
    for k, q in enumerate(qs):
        model.set_q(q); actual = model.poses()
        ep = np.linalg.norm(actual[:, :3]-targets[k, :, :3], axis=1)
        er = (Rotation.from_quat(actual[:, 3:]).inv()*Rotation.from_quat(targets[k, :, 3:])).magnitude()
        fk_difference = max(fk_difference, float(abs(ep-positions[k]).max()), float(abs(er-rotations[k]).max()))
        score[k], pair = model.clearance(); closest.append(pair)
        required = INTERNAL_WRIST_CLEARANCE_M if frozenset(pair) in model.internal_wrist_pairs else MODEL_CLEARANCE_M
        raw_at_pair[k] = score[k]-MODEL_CLEARANCE_M+required
        if score[k] >= MODEL_CLEARANCE_M:
            if checker is None:
                from hand_body_clearance import HandBodyClearance
                checker = HandBodyClearance(ROOT/'replay_robot/models/tron2_wuji_board_v1_20261110/robot.urdf')
            checker.set_q(q, hands[k]); hand_distance[k], _ = checker.clearance(); hand_checked[k] = True
            hp = checker.poses()
            assert np.max(abs(hp[:, :3]-actual[:, :3])) < 1e-6
            assert np.max((Rotation.from_quat(hp[:, 3:]).inv()*Rotation.from_quat(actual[:, 3:])).magnitude()) < 1e-6
        if k % 200 == 0:
            print(policy, 'collision', k, '/', len(t), flush=True)
    assert fk_difference < 1e-7, fk_difference
    collision_ok = (score >= MODEL_CLEARANCE_M) & hand_checked & (hand_distance >= MODEL_CLEARANCE_M)
    combined = valid.all(axis=1) & collision_ok

    chord_pos = np.full((len(t)-1, 2), np.nan); chord_rot = chord_pos.copy()
    eligible = valid[:-1] & valid[1:]
    for k in np.flatnonzero(eligible.any(axis=1)):
        alphas = np.linspace(0., 1., 11)[1:-1]
        rr = [Slerp([0., 1.], Rotation.from_quat(targets[k:k+2, a, 3:]))(alphas) for a in range(2)]
        pp, aa = [], []
        for j, alpha in enumerate(alphas):
            p, r = solver.fk(qs[k]+alpha*(qs[k+1]-qs[k]))
            desired = targets[k, :, :3]+alpha*(targets[k+1, :, :3]-targets[k, :, :3])
            pp.append(np.linalg.norm(p-desired, axis=1))
            aa.append([(Rotation.from_matrix(r[a]).inv()*rr[a][j]).magnitude() for a in range(2)])
        chord_pos[k, eligible[k]] = np.max(pp, axis=0)[eligible[k]]
        chord_rot[k, eligible[k]] = np.max(aa, axis=0)[eligible[k]]
    chord_fail = eligible & ((chord_pos > IK_POSITION_TOL) | (chord_rot > CHORD_ROTATION_TOL))
    steps = np.max(abs(np.diff(qs[:, :14], axis=0)).reshape(-1, 2, 7), axis=2)
    certificates = []
    for span in intervals(t, combined):
        a, b = span['first_frame'], span['last_frame']; row = dict(span)
        if a == b:
            row.update(swept_certified=False, reason='Single isolated sample')
        elif np.any(chord_fail[a:b]) or np.any(steps[a:b] > .2):
            row.update(swept_certified=False, reason='Chord accuracy or joint-step continuity failed')
        else:
            try:
                row['arm_body'] = certify_path(model, qs[a:b+1])
                row['hand_body'] = checker.check_path(qs[a:b+1], hands[a:b+1])
                row['swept_certified'] = True
            except ValueError as exc:
                row.update(swept_certified=False, reason=str(exc))
        certificates.append(row)
    worst = int(np.argmin(score)); failures = Counter(tuple(closest[k]) for k in np.flatnonzero(score < MODEL_CLEARANCE_M))
    report = dict(episode='episode102', policy=policy, frames=len(t), duration_s=float(t[-1]),
        execution_ready=False, motion_commands_sent=0,
        thresholds=dict(position_m=IK_POSITION_TOL, orientation_deg=2., chord_orientation_deg=3.,
                        model_clearance_m=MODEL_CLEARANCE_M, internal_wrist_clearance_m=INTERNAL_WRIST_CLEARANCE_M),
        pose_passing_frames=valid.sum(axis=0).tolist(), both_pose_passing_frames=int(valid.all(axis=1).sum()),
        left_pose_intervals=intervals(t, valid[:, 0]), right_pose_intervals=intervals(t, valid[:, 1]),
        both_pose_intervals=intervals(t, valid.all(axis=1)),
        first_position_error_mm=(positions[0]*1000).tolist(), first_orientation_error_deg=np.rad2deg(rotations[0]).tolist(),
        first_frame_attempts=attempts,
        collision=dict(arm_body_passing_frames=int((score >= MODEL_CLEARANCE_M).sum()),
                       hand_body_checked_frames=int(hand_checked.sum()),
                       all_model_passing_frames=int(collision_ok.sum()),
                       both_pose_and_collision_passing_frames=int(combined.sum()),
                       both_pose_and_collision_intervals=intervals(t, combined),
                       minimum_arm_clearance_score_mm=float(score[worst]*1000),
                       minimum_score_pair=closest[worst], raw_distance_at_minimum_score_mm=float(raw_at_pair[worst]*1000),
                       frequent_arm_failures=[dict(pair=list(pair), frames=n) for pair,n in failures.most_common()],
                       minimum_checked_hand_body_mm=float(np.nanmin(hand_distance)*1000) if hand_checked.any() else None),
        chord=dict(nominal_hz=300, tested_intervals_per_arm=eligible.sum(axis=0).tolist(),
                   failed_intervals_per_arm=chord_fail.sum(axis=0).tolist(),
                   large_joint_steps_in_passing_intervals=(eligible & (steps > .2)).sum(axis=0).tolist()),
        swept_certificates=certificates, independent_pinocchio_mujoco_residual_difference_max=fk_difference,
        outside_movej_frames=int(np.any((qs[:, :14] < movej_lower[:14]) | (qs[:, :14] > movej_upper[:14]), axis=1).sum()),
        method='12 first-frame seeds, normalized and legacy orientation weights; continuous previous-frame bounded IK',
        source_sha256=hashes, elapsed_seconds=time.monotonic()-began,
        limitations=['Source assembly/hand-eye applicability assumed from declared export convention; no raw source arm q',
                     'URDF-only range is not a verified ServoP envelope', 'Environment, cables, adapter and within-hand collision excluded',
                     'No live start/MoveJ entry, controller tracking or hardware validation', 'Solver failure is not global infeasibility proof'])
    assert all(sha(ROOT/name) == digest for name,digest in hashes.items())
    np.savez_compressed(out/'collision_and_chord_diagnostics.npz', t=t, arm_clearance_score_m=score,
                        raw_distance_at_limiting_pair_m=raw_at_pair, hand_body_distance_m=hand_distance,
                        hand_body_checked=hand_checked, collision_pass=collision_ok, combined_pass=combined,
                        chord_position_error_m=chord_pos, chord_rotation_error_rad=chord_rot)
    save(out/'report.json', report)
    print('RESULT', json.dumps({k:v for k,v in report.items() if k not in ['first_frame_attempts', 'source_sha256']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=['movej', 'urdf'], required=True)
    run(parser.parse_args().policy)
