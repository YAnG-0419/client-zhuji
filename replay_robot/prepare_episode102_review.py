"""Apply the previously confirmed camera task mapping to episode102; file I/O only."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT/'replay_robot/episode102_review_20261111'
sys.path[:0] = [str(ROOT/'replay_robot'), str(ROOT/'replay_hand'), str(ROOT/'replay')]
from prepare_task_replay import fixed_scene_base, transfer
from align_board_frames import rigid
from episode_cross import load_cross_episode
from movej_prepare import EEF_BOX


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    target_file = WORK/'replay_input.npz'
    assert not target_file.exists(), 'Refusing to overwrite converted input'
    old = ROOT/'replay_robot/task_transfer_20261110'
    mapping_path = old/'inputs/task_mapping.json'
    assert sha(mapping_path) == '38e68d7d55cfab12fdcc818db11346c3a15ba706d716e88d36f4477f60572393'
    mapping = json.loads(mapping_path.read_text())
    source_cal_path = old/'inputs/handeye/franka-sharpa/handeye_calibration.json'
    target_cal_path = old/'inputs/handeye/zhuji/head_41p5/left_head_41p5_kinematic_estimate.json'
    source_cal = json.loads(source_cal_path.read_text()); target_cal = json.loads(target_cal_path.read_text())
    assert mapping['source_camera_id'] == 'franka-sharpa' and mapping['target_camera_id'] == 'zhuji'
    M = rigid(mapping['target_camera_T_source_camera'], 'task mapping')
    G = rigid(target_cal['T_base_camera'], 'target base camera')
    urdf = ROOT/'replay_robot/traj/sharpa_to_wuji2_beta/assets/motion/source_assembly.urdf'
    prior = json.loads((old/'episodes/episode594/conversion_report.json').read_text())
    with np.load(WORK/'source_scene.npz', allow_pickle=False) as z:
        payload = {k:z[k].copy() for k in z.files}
    meta = json.loads(str(payload['metadata_json']))
    assert meta['frame'] == 'lychee_root' and meta['placement_policy'] == 'source_scene_unplaced'
    t, hands = payload['t'], payload['hands']
    eef = np.empty_like(payload['eef']); roots, boards, eef_matrices, details = [], [], [], {}
    original = ROOT/'replay_robot/traj/episode102/episode102_replay_trajectory.npz'
    with np.load(original, allow_pickle=False) as source:
        np.testing.assert_array_equal(t, source['time_s'])
        np.testing.assert_array_equal(payload['timestamp_ns'], source['timestamp_ns'])
        for a, side in enumerate(['left', 'right']):
            np.testing.assert_array_equal(hands[:, a], source[side+'_joint_positions_rad'])
            target, chain = transfer(source[side+'_root_pose'], fixed_scene_base(urdf, side),
                                     source_cal['transforms']['camera_T_'+side+'_base'], M, G)
            np.testing.assert_allclose(chain, prior['sides'][side]['T_tron_base_source_scene'], atol=1e-12, rtol=0)
            eb = np.asarray(meta['sides'][side]['T_eef_B']); bh = np.asarray(meta['sides'][side]['T_B_hand'])
            board = rigid(target @ np.linalg.inv(bh), 'target B')
            arm = rigid(board @ np.linalg.inv(eb), 'target SDK EEF')
            np.testing.assert_allclose(arm @ eb @ bh, target, atol=1e-9, rtol=0)
            old_arm = np.tile(np.eye(4), (len(t), 1, 1))
            old_arm[:, :3, 3] = payload['eef'][:, a, :3]
            old_arm[:, :3, :3] = Rotation.from_quat(payload['eef'][:, a, 3:]).as_matrix()
            np.testing.assert_allclose(chain @ old_arm, arm, atol=1e-9, rtol=0)
            eef[:, a, :3] = arm[:, :3, 3]
            eef[:, a, 3:] = Rotation.from_matrix(arm[:, :3, :3]).as_quat()
            for k in range(1, len(t)):
                if np.dot(eef[k-1, a, 3:], eef[k, a, 3:]) < 0:
                    eef[k, a, 3:] *= -1
            roots.append(target); boards.append(board); eef_matrices.append(arm)
            details[side] = dict(T_tron_base_source_scene=chain.tolist(), T_sdk_eef_board_center=eb.tolist(),
                                 T_board_center_hand=bh.tolist(),
                                 target_xyz_min_m=arm[:, :3, 3].min(axis=0).tolist(),
                                 target_xyz_max_m=arm[:, :3, 3].max(axis=0).tolist())
            payload[side+'_T_sdk_eef_board_center'] = eb
            payload[side+'_T_board_center_hand'] = bh
    speed = abs(np.diff(hands, axis=0))/np.diff(t)[:, None, None]
    outside = np.any((eef[:, :, :3] < EEF_BOX[None, :, :, 0]) |
                     (eef[:, :, :3] > EEF_BOX[None, :, :, 1]), axis=2)
    paths = [original, mapping_path, source_cal_path, target_cal_path, urdf, WORK/'source_scene.npz',
             ROOT/'replay_robot/models/tron2_wuji_board_v1_20261110/alignment_config.json',
             ROOT/'replay_hand/build_exported_trajectory.py', Path(__file__).resolve()]
    files = dict(meta['source_files']); files.update({str(p):sha(p) for p in paths})
    meta.update(episode='episode102', frame='tron2_base_link', placement_policy='preserve_common_frame_v1',
                execution_ready=False, motion_commands_sent=0, frames=len(t), duration_s=float(t[-1]),
                controlled_hand_origin='B: board center = Wuji l_base_link/r_base_link origin',
                per_arm_reanchoring_allowed=False, source_files=files,
                source_workcell_match='Assumes exported lychee_root uses the same nominal assembly and measured source hand-eye as previous episodes; raw source arm q unavailable for independent source FK',
                target_extrinsic_independently_validated=False,
                target_head_pitch_yaw_rad=target_cal['new_head_pitch_yaw_rad'],
                target_camera_extrinsic_policy='Same common left-derived 41.5 degree T_base_camera for both arms',
                task_placement=dict(method='previously confirmed camera task mapping', mapping_sha256=sha(mapping_path),
                                    physical_validation_complete=False),
                camera_alignment='Existing source per-arm camera extrinsics and measured task mapping',
                conversion=details, source_values_unchanged=True, original_time_unchanged=True,
                source_arm_fk_available=False,
                workspace_outside_frames_per_arm=outside.sum(axis=0).tolist(),
                hand_peak_speed_rad_s=speed.max(axis=(0, 2)).tolist(),
                hand_only_retimed_lower_bound_s=float(np.maximum(np.diff(t), abs(np.diff(hands, axis=0)).max(axis=(1, 2))).sum()))
    payload.update(eef=eef, tron_base_T_hand=np.stack(roots, axis=1),
                   tron_base_T_board_center=np.stack(boards, axis=1),
                   tron_base_T_sdk_eef=np.stack(eef_matrices, axis=1),
                   metadata_json=np.asarray(json.dumps(meta)))
    np.savez_compressed(target_file, **payload)
    loaded = load_cross_episode(target_file, 50.)
    meta['output_sha256'] = sha(target_file); meta['loader_50hz_frames'] = len(loaded[0])
    (WORK/'conversion_report.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:meta[k] for k in ['frames', 'duration_s', 'workspace_outside_frames_per_arm',
                                        'hand_peak_speed_rad_s', 'hand_only_retimed_lower_bound_s', 'conversion']}), flush=True)


if __name__ == '__main__':
    main()
