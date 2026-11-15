"""Offline frame conversion only: no ROS, robot SDK, or device transport.

H: exported hand root; C: centered calibration-board axes; E: SDK EEF;
S: source assembly scene. T_A_B maps B coordinates into A.
T_S_C = T_S_H @ inverse(T_C_H)
T_E_C = T_E_board @ Translation(board_center)
T_S_E = T_S_C @ inverse(T_E_C)
The missing T_tron_base_S is deliberately neither guessed nor applied.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SIDES = ('left', 'right')


def rigid(value, label):
    a = np.asarray(value, dtype=float)
    if a.shape[-2:] != (4, 4) or not np.isfinite(a).all():
        raise ValueError(label + ': expected finite 4x4 transforms')
    if not np.allclose(a[..., 3, :], [0, 0, 0, 1], atol=1e-10, rtol=0):
        raise ValueError(label + ': invalid homogeneous row')
    r = a[..., :3, :3]
    if not np.allclose(np.swapaxes(r, -1, -2) @ r, np.eye(3), atol=1e-8, rtol=0):
        raise ValueError(label + ': invalid rotation')
    if not np.allclose(np.linalg.det(r), 1, atol=1e-8, rtol=0):
        raise ValueError(label + ': reflected frame')
    return a


def tool_frames(calibration, rotation, center):
    e_board = rigid(calibration['T_eef_board'], 'T_eef_board')
    board_c = np.eye(4)
    board_c[:3, 3] = center
    e_c = e_board @ board_c
    c_h = np.eye(4)
    c_h[:3, :3] = rotation
    rigid(c_h, 'T_board_center_hand')
    return e_c, c_h


def convert_poses(scene_hand, e_c, c_h):
    scene_hand = rigid(scene_hand, 'T_scene_hand')
    e_c = rigid(e_c, 'T_eef_board_center')
    c_h = rigid(c_h, 'T_board_center_hand')
    scene_center = scene_hand @ np.linalg.inv(c_h)
    scene_eef = scene_center @ np.linalg.inv(e_c)
    rigid(scene_center, 'T_scene_board_center')
    rigid(scene_eef, 'T_scene_sdk_eef')
    if not np.allclose(scene_eef @ e_c @ c_h, scene_hand, atol=1e-9, rtol=0):
        raise AssertionError('SDK-to-hand reconstruction failed')
    return scene_center, scene_eef


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root, config_path, episode, output):
    config = json.loads(config_path.read_text())
    if config.get('schema') != 'pengyang_board_frame_alignment_v1':
        raise ValueError('Unsupported alignment configuration')
    if config.get('source_scene_to_tron_base') is not None:
        raise ValueError('This converter only exports source-scene poses')
    if output.exists():
        raise FileExistsError('Output exists; choose a new directory: ' + str(output))
    trajectory = episode / 'wuji_trajectory.npz'
    scene_path = episode / 'wuji_motion_scene.npz'
    provenance = {str(p.resolve()): sha(p) for p in (config_path, trajectory, scene_path)}
    payload = {}
    report = dict(schema='board_center_source_scene_v1', episode=episode.name,
                  execution_ready=False, motion_commands_sent=0,
                  reference_frame='source_assembly_lychee_root',
                  source_scene_to_tron_base_applied=False,
                  frame_alignment_authority='operator-confirmed, not independently measured',
                  origin_policy=config['origin_policy'], sides={})
    with np.load(trajectory, allow_pickle=False) as hand, np.load(scene_path, allow_pickle=False) as scene:
        stamps, t = hand['timestamp_ns'].copy(), hand['time_s'].copy()
        if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or t[0] != 0 or np.any(np.diff(t) <= 0):
            raise ValueError('Invalid source time axis')
        if not np.array_equal(stamps, scene['timestamp_ns']) or not np.allclose(t, scene['time_s'], atol=1e-12, rtol=0):
            raise ValueError('Hand and scene time axes differ')
        if not np.allclose(t, (stamps - stamps[0]) / 1e9, atol=1e-9, rtol=0):
            raise ValueError('Timestamp and source time mismatch')
        alignment = str(hand['alignment_version'])
        if str(scene['alignment_version']) != alignment:
            raise ValueError('Hand and scene alignment versions differ')
        payload.update(timestamp_ns=stamps, time_s=t, source_alignment_version=np.asarray(alignment))
        report.update(frames=len(t), duration_s=float(t[-1]), source_alignment_version=alignment,
                      package_alignment_reapplied=False)
        for side in SIDES:
            entry = config['sides'][side]
            cal_path = root / entry['calibration']
            cal = json.loads(cal_path.read_text())
            provenance[str(cal_path.resolve())] = sha(cal_path)
            e_c, c_h = tool_frames(cal, entry['R_board_center_hand'], config['board_center_in_board_m'])
            source = scene[side + '_scene_T_wuji_base']
            if source.shape != (len(t), 4, 4):
                raise ValueError('Unexpected source pose shape')
            # Independently verify that the scene root is the trajectory's exported root.
            local = np.tile(np.eye(4), (len(t), 1, 1))
            local[:, :3, :3] = Rotation.from_rotvec(hand[side + '_root_rotvec_in_sharpa_rad']).as_matrix()
            local[:, :3, 3] = hand[side + '_root_xyz_in_sharpa_m']
            if not np.allclose(scene[side + '_scene_T_sharpa_root'] @ local, source, atol=1e-9, rtol=0):
                raise ValueError('Scene does not compose from trajectory root')
            centered, eef = convert_poses(source, e_c, c_h)
            q = hand[side + '_wuji_joints_rad'].copy()
            if q.shape != (len(t), 20) or not np.isfinite(q).all():
                raise ValueError('Invalid source hand joints')
            payload.update({side + '_scene_T_board_center': centered,
                            side + '_scene_T_sdk_eef': eef,
                            side + '_T_sdk_eef_board_center': e_c,
                            side + '_T_board_center_hand': c_h,
                            side + '_wuji_joints_rad': q,
                            side + '_wuji_joint_names': hand[side + '_wuji_joint_names'].copy()})
            reconstructed = eef @ e_c @ c_h
            report['sides'][side] = dict(
                axis_relation=entry['axis_relation'],
                R_board_center_hand=c_h[:3, :3].tolist(),
                T_sdk_eef_board_center=e_c.tolist(),
                board_center_position_preserved_max_m=float(np.linalg.norm(centered[:, :3, 3]-source[:, :3, 3], axis=1).max()),
                hand_pose_roundtrip_max_abs=float(np.max(np.abs(reconstructed-source))),
                hand_joint_values_unchanged=bool(np.array_equal(q, hand[side+'_wuji_joints_rad'])),
                calibration_validation_passed=cal.get('validation_passed'))
    report['source_sha256'] = provenance
    payload['metadata_json'] = np.asarray(json.dumps(report, ensure_ascii=False))
    output.mkdir(parents=True, exist_ok=False)
    result = output / 'board_center_source_scene.npz'
    np.savez_compressed(result, **payload)
    report['output'] = str(result.resolve())
    report['output_sha256'] = sha(result)
    (output / 'alignment_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('board_frame_alignment.json'))
    parser.add_argument('--episode', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = build(args.root, args.config, args.episode, args.output)
    print(json.dumps({k: report[k] for k in ['episode','frames','output','execution_ready','motion_commands_sent']},ensure_ascii=False))


if __name__ == '__main__':
    main()
