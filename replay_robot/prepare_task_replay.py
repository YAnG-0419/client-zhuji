"""File-only task transfer for episodes 594/615/616; never opens robot devices.

T_A_B maps B into A, metres and column vectors. Each nominal source arm base
is removed before applying its measured source-camera extrinsic. One common
target camera-to-base transform is used for both hands. Tool origin remains B.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from align_board_frames import rigid, tool_frames, convert_poses

SIDES = ('left', 'right')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def origin_transform(origin):
    t = np.eye(4)
    if origin is not None:
        t[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
        t[:3, :3] = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return t


def fixed_scene_base(urdf, side):
    joints = {j.find('child').get('link'): j for j in ET.parse(urdf).getroot().findall('joint')}
    child, t = side+'_fr3v2_link0', np.eye(4)
    seen = set()
    while child in joints:
        if child in seen:
            raise ValueError('Cyclic source assembly')
        seen.add(child)
        joint = joints[child]
        if joint.get('type') != 'fixed':
            raise ValueError('Moving joint in source scene-to-base chain')
        t = origin_transform(joint.find('origin')) @ t
        child = joint.find('parent').get('link')
    if child != 'lychee_root':
        raise ValueError('Unexpected source scene root: '+child)
    return rigid(t, 'scene_T_source_base')


def transfer(scene_pose, scene_base, camera_base, mapping, target_base_camera):
    """Map source pose through the actual per-arm base, then common cameras."""
    chain = (rigid(target_base_camera, 'target_base_T_camera') @ rigid(mapping, 'target_camera_T_source_camera')
             @ rigid(camera_base, 'source_camera_T_base') @ np.linalg.inv(rigid(scene_base, 'scene_T_base')))
    return chain @ rigid(scene_pose, 'scene_T_pose'), chain


def hand_order(root, side, names):
    path = root/'teleop/wuji_teleop/models/hand2_beta/mjcf'/(side+'.xml')
    actuator = ET.parse(path).getroot().find('actuator')
    names_in_actuator = [a.get('joint') for a in actuator]
    if len(names_in_actuator) != 20 or len(set(names_in_actuator)) != 20:
        raise ValueError('Unexpected deployed hand actuator order')
    if list(names) != names_in_actuator:
        raise ValueError('Source hand order differs from deployed actuator order')
    return np.array([side+'_finger%d_joint%d' % (finger, joint) for finger in range(1,6) for joint in range(1,5)]), path


def build(root, work, episode_name):
    inputs, output = work/'inputs', work/'episodes'/episode_name
    if output.exists():
        raise FileExistsError('Refusing to overwrite: '+str(output))
    config = json.loads((work/'config.json').read_text())
    if config['schema'] != 'replay_task_transfer_v1' or config['operator_confirmed_target_head_pitch_deg'] != 41.5:
        raise ValueError('Unexpected task transfer or head configuration')
    mp = inputs/'task_mapping.json'
    mapping = json.loads(mp.read_text())
    if mapping['source_camera_id'] != 'franka-sharpa' or mapping['target_camera_id'] != 'zhuji':
        raise ValueError('Wrong mapping direction/camera identities')
    M = rigid(mapping['target_camera_T_source_camera'], 'mapping')
    if not np.allclose(M @ rigid(mapping['source_camera_T_target_camera'], 'mapping inverse'), np.eye(4), atol=1e-12):
        raise ValueError('Mapping inverse failed')
    observations = [json.loads((inputs/'observations'/s/'observation.json').read_text()) for s in ('franka-sharpa','zhuji')]
    if any(o['board_fingerprint'] != mapping['board_fingerprint'] for o in observations):
        raise ValueError('Board definitions differ')
    closure = float(np.max(np.abs(M @ rigid(observations[0]['camera_T_board'], 'source board') - rigid(observations[1]['camera_T_board'], 'target board'))))
    if closure > 1e-10:
        raise ValueError('Observation mapping closure failed')
    source_cal_path = inputs/'handeye/franka-sharpa/handeye_calibration.json'
    source_cal = json.loads(source_cal_path.read_text())
    if source_cal['conventions']['translation_unit'] != 'metre':
        raise ValueError('Source calibration units')
    target_paths = [inputs/'handeye/zhuji/head_41p5'/(s+'_head_41p5_kinematic_estimate.json') for s in SIDES]
    target_cals = [json.loads(p.read_text()) for p in target_paths]
    G, alt = [rigid(d['T_base_camera'], 'target base camera') for d in target_cals]
    # This selection is fixed in the archived config, never selected per hand.
    if config['target_camera_extrinsic'] != 'left_head_41p5_kinematic_estimate':
        raise ValueError('Unsupported primary target extrinsic')
    frozen = root/'replay_robot/models/tron2_wuji_board_v1_20261110'
    align_path = frozen/'alignment_config.json'
    align = json.loads(align_path.read_text())
    episode = root/'replay_robot/traj/sharpa_to_wuji2_beta'/episode_name
    urdf = episode.parent/'assets/motion/source_assembly.urdf'
    prior = root/'replay_robot/derived/board_aligned_20261110'/episode_name/'board_center_source_scene.npz'
    paths = [work/'config.json', mp, source_cal_path, align_path, urdf, prior,
             episode/'wuji_motion_scene.npz', episode/'wuji_trajectory.npz']+target_paths
    details, payload = {}, {}
    with np.load(episode/'wuji_motion_scene.npz', allow_pickle=False) as scene, np.load(episode/'wuji_trajectory.npz', allow_pickle=False) as hand, np.load(prior, allow_pickle=False) as old:
        t, stamps = hand['time_s'].copy(), hand['timestamp_ns'].copy()
        if t.ndim != 1 or len(t)<2 or t[0]!=0 or np.any(np.diff(t)<=0) or not np.isfinite(t).all():
            raise ValueError('Invalid timeline')
        for z in (scene, old):
            if not np.array_equal(stamps, z['timestamp_ns']) or not np.array_equal(t, z['time_s']):
                raise ValueError('Timeline mismatch')
        if not np.allclose(t, (stamps-stamps[0])/1e9, atol=1e-9, rtol=0):
            raise ValueError('Timestamp mismatch')
        if str(hand['alignment_version']) != 'wuji_beta_local_z_left_p90_right_m90_v1':
            raise ValueError('Unexpected package alignment')
        roots, centers, eefs, values, alt_roots = [], [], [], [], []
        limits = json.loads((root/'teleop/wuji_control/config/joint_limits.json').read_text())
        for side in SIDES:
            S_B = fixed_scene_base(urdf, side)
            C_B = rigid(source_cal['transforms']['camera_T_'+side+'_base'], 'source camera base')
            source = rigid(scene[side+'_scene_T_wuji_base'], 'source roots')
            if source.shape != (len(t),4,4):
                raise ValueError('Unexpected source shape')
            local = np.tile(np.eye(4), (len(t),1,1))
            local[:,:3,3] = hand[side+'_root_xyz_in_sharpa_m']
            local[:,:3,:3] = Rotation.from_rotvec(hand[side+'_root_rotvec_in_sharpa_rad']).as_matrix()
            if not np.allclose(scene[side+'_scene_T_sharpa_root'] @ local, source, atol=1e-9, rtol=0):
                raise ValueError('Source hand-root composition mismatch')
            target, chain = transfer(source, S_B, C_B, M, G)
            alternative, _ = transfer(source, S_B, C_B, M, alt)
            cal_path = root/align['sides'][side]['calibration']
            paths.append(cal_path)
            e_c, c_h = tool_frames(json.loads(cal_path.read_text()), align['sides'][side]['R_board_center_hand'], align['board_center_in_board_m'])
            center, eef = convert_poses(target, e_c, c_h)
            # Verify that the three already-converted B-point trajectories are
            # preserved; only their reference frame changes here.
            old_error = float(np.max(np.abs(center - chain @ old[side+'_scene_T_board_center'])))
            if old_error > 1e-9:
                raise ValueError('Existing B-point trajectory was changed')
            q = hand[side+'_wuji_joints_rad'].copy()
            names, path = hand_order(root, side, hand[side+'_wuji_joint_names'].tolist())
            paths.append(path)
            bounds = np.asarray(limits[side])
            if q.shape != (len(t),20) or not np.isfinite(q).all() or np.any(q<bounds[:,0]-1e-6) or np.any(q>bounds[:,1]+1e-6):
                raise ValueError('Invalid or out-of-limit hand targets')
            payload[side+'_hand_joint_names'] = names
            payload[side+'_T_sdk_eef_board_center'] = e_c
            payload[side+'_T_board_center_hand'] = c_h
            roots.append(target); centers.append(center); eefs.append(eef); values.append(q);alt_roots.append(alternative)
            details[side] = dict(T_source_scene_source_base=S_B.tolist(), T_source_camera_source_base=C_B.tolist(),
                T_tron_base_source_scene=chain.tolist(), T_sdk_eef_board_center=e_c.tolist(), T_board_center_hand=c_h.tolist(),
                prior_B_trajectory_max_abs=old_error, tool_roundtrip_max_abs=float(np.max(np.abs(eef@e_c@c_h-target))),
                hand_values_unchanged=bool(np.array_equal(q,old[side+'_wuji_joints_rad'])),
                eef_xyz_min_m=eef[:,:3,3].min(axis=0).tolist(), eef_xyz_max_m=eef[:,:3,3].max(axis=0).tolist(),
                alternate_target_calibration_B_difference_max_m=float(np.linalg.norm(alternative[:,:3,3]-target[:,:3,3],axis=1).max()))
        roots, centers, matrices = [np.stack(x,axis=1) for x in (roots,centers,eefs)]
        values = np.stack(values,axis=1)
        eef = np.concatenate([matrices[:,:,:3,3], Rotation.from_matrix(matrices[:,:,:3,:3].reshape(-1,3,3)).as_quat().reshape(len(t),2,4)],axis=-1)
        # Continuous quaternion signs make interchange easier; Slerp itself is sign-invariant.
        for side_index in range(2):
            for k in range(1,len(t)):
                if np.dot(eef[k-1,side_index,3:],eef[k,side_index,3:])<0: eef[k,side_index,3:] *= -1
        speed = np.max(abs(np.diff(values,axis=0))/np.diff(t)[:,None,None],axis=(1,2))
        meta = dict(schema='replay_hand_cross_v1', episode=episode_name, frame='tron2_base_link',
            transform_convention='T_A_B maps B into A; metres; xyzw quaternions', placement_policy='preserve_common_frame_v1',
            per_arm_reanchoring_allowed=False, execution_ready=False, motion_commands_sent=0,
            alignment_version=str(hand['alignment_version']), package_alignment_reapplied=False,
            controlled_hand_origin='B: calibration board center = Wuji l_base_link/r_base_link origin',
            camera_alignment='Measured two-workcell task mapping, source per-arm measured bases, common target 41.5-degree camera extrinsic',
            target_camera_extrinsic_policy=config['target_extrinsic_policy'], target_head_pitch_yaw_rad=target_cals[0]['new_head_pitch_yaw_rad'],
            mapping_observation_closure_max_abs=closure, T_tron_base_target_camera=G.tolist(),
            T_target_camera_source_camera=M.tolist(), frames=len(t), duration_s=float(t[-1]),
            hand_original_peak_speed_rad_s=float(speed.max()),
            hand_only_uniform_speed_ceiling=float(min(1,1/speed.max())),
            hand_only_retimed_lower_bound_s=float(np.maximum(np.diff(t),np.max(abs(np.diff(values,axis=0)),axis=(1,2))).sum()),
            source_workcell_match=config['source_workcell_match'], target_extrinsic_independently_validated=False,
            environment_collision_validated=False, source_files={str(p.resolve()):sha(p) for p in paths}, sides=details)
        output.mkdir(parents=True)
        np.savez_compressed(output/'replay_input.npz', timestamp_ns=stamps,t=t,eef=eef,hands=values,
            tron_base_T_hand=roots,tron_base_T_board_center=centers,tron_base_T_sdk_eef=matrices,
            # Compatibility alias used by the existing offline hand-sphere checker.
            hand_root_T_tron_base=roots, metadata_json=np.asarray(json.dumps(meta)), **payload)
        meta['output_sha256'] = sha(output/'replay_input.npz')
        save(output/'conversion_report.json',meta)
        print(json.dumps(dict(episode=episode_name,frames=len(t),duration_s=float(t[-1]),sides=details),ensure_ascii=False),flush=True)
    return meta


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--work',type=Path)
    p.add_argument('--episode',choices=['episode594','episode615','episode616'])
    a=p.parse_args();work=a.work or a.root/'replay_robot/task_transfer_20261110'
    for name in [a.episode] if a.episode else ['episode594','episode615','episode616']:
        build(a.root,work,name)
