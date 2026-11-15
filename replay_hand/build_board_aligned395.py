"""Offline episode395 bridge to the confirmed Wuji base/board tool convention.

The old free-root solver returns wrist-centred FK, not base_link FK. Recover
base_link first, then apply the package's once-only local Z orientation policy.
Never mix the free-root pose with the unrelated deployed-adaptive joint solution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

SIDES = ('left', 'right')


def wrist_to_base(actions, wrist_in_base):
    actions = np.asarray(actions)
    r = Rotation.from_rotvec(actions[:, 3:6]).as_matrix()
    t = np.tile(np.eye(4), (len(actions), 1, 1))
    t[:, :3, :3] = r
    t[:, :3, 3] = actions[:, :3] - np.einsum('nij,j->ni', r, wrist_in_base)
    return t


def resample_pose(stamps, poses, query):
    origin = int(stamps[0])
    ts = (stamps-origin)/1e9
    tq = (query-origin)/1e9
    if tq[0] < ts[0] or tq[-1] > ts[-1]:
        raise ValueError('Pose interpolation would extrapolate')
    out = np.tile(np.eye(4), (len(query), 1, 1))
    out[:, :3, :3] = Slerp(ts, Rotation.from_matrix(poses[:, :3, :3]))(tq).as_matrix()
    for j in range(3):
        out[:, j, 3] = np.interp(tq, ts, poses[:, j, 3])
    return out


def tool_frames(calibration, axes, center):
    e_c = np.asarray(calibration['T_eef_board'], dtype=float).copy()
    e_c[:3, 3] += e_c[:3, :3] @ np.asarray(center)
    c_h = np.eye(4)
    c_h[:3, :3] = axes
    for t in (e_c, c_h):
        if not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-8) or np.linalg.det(t[:3, :3]) < .999999:
            raise ValueError('Invalid rigid tool transform')
    return e_c, c_h


def build(root, converter, config_dir, output):
    import sys
    sys.path.insert(0, str(converter))
    from sharpa21_to_wuji2 import Wuji2Skeleton, control_joint_names
    config = json.loads((config_dir/'alignment_config.json').read_text())
    arm_path = root/'data/episode395_tron2_camera_eef.npz'
    hand_path = root/'data/episode395_wuji2.npz'
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (arm_path, hand_path, config_dir/'alignment_config.json')}
    with np.load(arm_path, allow_pickle=False) as arm, np.load(hand_path, allow_pickle=False) as hand:
        begin = max(int(arm['timestamp_ns'][0]), *(int(hand[s+'_timestamp_ns'][0]) for s in SIDES))
        end = min(int(arm['timestamp_ns'][-1]), *(int(hand[s+'_timestamp_ns'][-1]) for s in SIDES))
        stamps = np.unique(np.r_[np.arange(begin, end, 20000000, dtype=np.int64), end])
        t = (stamps-stamps[0])/1e9
        eef, hands = np.empty((len(t), 2, 7)), np.empty((len(t), 2, 20))
        roots, centers, details = [], [], {}
        for i, side in enumerate(SIDES):
            model = Wuji2Skeleton(side)
            model.fk(model.neutral)
            wrist = model.data.site_xpos[model.site_ids[0]].copy()
            action = hand[side+'_wuji2_free_root_local_actions']
            hs = hand[side+'_timestamp_ns']
            local = wrist_to_base(action, wrist)
            # Verify the bridge against the OLD stored points before correcting axes.
            error = 0.
            for k, a in enumerate(action):
                model.fk(a[6:])
                points = model.data.site_xpos[model.site_ids].copy()
                recovered = points @ local[k, :3, :3].T + local[k, :3, 3]
                error = max(error, float(abs(recovered-hand[side+'_wuji2_free_root_fk_points'][k]).max()))
            if error > 1e-9:
                raise ValueError('Wrist/base bridge failed stored FK reconstruction')
            angle = 90 if side == 'left' else -90
            local[:, :3, :3] = local[:, :3, :3] @ Rotation.from_euler('z', angle, degrees=True).as_matrix()
            # One existing, explicitly assumed camera mapping for BOTH arms.
            source = resample_pose(arm['timestamp_ns'], arm[side+'_sharpa_root_T_tron_base'], stamps)
            target_root = source @ resample_pose(hs, local, stamps)
            cal_path = config_dir/'calibration'/(side+'.json')
            provenance[str(cal_path)] = hashlib.sha256(cal_path.read_bytes()).hexdigest()
            e_c, c_h = tool_frames(json.loads(cal_path.read_text()),
                                  config['sides'][side]['R_board_center_hand'], config['board_center_in_board_m'])
            board = target_root @ np.linalg.inv(c_h)
            target_eef = board @ np.linalg.inv(e_c)
            roundtrip = float(abs(target_eef @ e_c @ c_h-target_root).max())
            if roundtrip > 1e-9:
                raise ValueError('Tool-chain roundtrip failed')
            eef[:, i, :3] = target_eef[:, :3, 3]
            eef[:, i, 3:] = Rotation.from_matrix(target_eef[:, :3, :3]).as_quat()
            for j in range(20):
                hands[:, i, j] = np.interp((stamps-begin)/1e9, (hs-begin)/1e9, action[:, 6+j])
            if np.any(hands[:, i] < model.lower-1e-8) or np.any(hands[:, i] > model.upper+1e-8):
                raise ValueError('Hand targets outside installed limits')
            roots.append(target_root); centers.append(board)
            details[side] = dict(wrist_in_base_m=wrist.tolist(), stored_fk_bridge_max_m=error,
                                 local_z_correction_deg=angle, tool_roundtrip_max=roundtrip,
                                 T_eef_board_center=e_c.tolist(), T_board_center_hand=c_h.tolist())
        metadata = dict(schema='replay_hand_cross_v1', frame='tron2_base_link',
                        placement_policy='preserve_common_frame_v1',
                        alignment_version='episode395_free_root_to_confirmed_board_v1',
                        execution_ready=False, motion_commands_sent=0,
                        camera_alignment='Existing operator-assumed common optical frames; 41.5 degree kinematic extrinsic; not measured cross-cell calibration',
                        hand_source='Paired Wuji2 free-root q20, not deployed-adaptive q20',
                        free_root_adapter_applied=True, per_arm_reanchoring_allowed=False,
                        source_files=provenance, sides=details)
        roots, centers = np.stack(roots, axis=1), np.stack(centers, axis=1)
        separation = np.linalg.norm(roots[:, 0, :3, 3]-roots[:, 1, :3, 3], axis=1)
        metadata['root_separation_m'] = dict(first=float(separation[0]), min=float(separation.min()),
                                           min_source_s=float(t[separation.argmin()]))
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, timestamp_ns=stamps, t=t, eef=eef, hands=hands,
                            hand_root_T_tron_base=roots, board_center_T_tron_base=centers,
                            left_hand_joint_names=np.asarray(control_joint_names('left')),
                            right_hand_joint_names=np.asarray(control_joint_names('right')),
                            metadata_json=np.asarray(json.dumps(metadata)))
        output.with_suffix('.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print(json.dumps(dict(output=str(output), frames=len(t), duration_s=t[-1],
                             root_separation_m=metadata['root_separation_m'], sides=details), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.root, args.root/'keypoints21-action-converter-20261105',
          Path(__file__).resolve().parent/'alignment_20261110', args.output)
