"""Offline board-correspondence transfer of a Wuji camera-frame trajectory.

No task repositioning, no human-to-hand retargeting and no hardware interfaces.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
sys.path.insert(0, str(ROOT/'replay_hand'))
from build_board_aligned395 import tool_frames
from build_exported_trajectory import model_names
from episode_cross import load_cross_episode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def rigid(value):
    value = np.asarray(value, dtype=float)
    assert value.shape[-2:] == (4, 4) and np.isfinite(value).all()
    np.testing.assert_allclose(value[..., 3, :], np.broadcast_to([0, 0, 0, 1], value[..., 3, :].shape), atol=1e-10, rtol=0)
    r = value[..., :3, :3]
    np.testing.assert_allclose(np.swapaxes(r, -1, -2)@r, np.broadcast_to(np.eye(3), r.shape), atol=1e-8, rtol=0)
    np.testing.assert_allclose(np.linalg.det(r), 1, atol=1e-8, rtol=0)
    return value


def resample(t, poses, hands, u):
    result = np.tile(np.eye(4), (len(u), 2, 1, 1))
    q = np.empty((len(u), 2, 20))
    for side in range(2):
        result[:, side, :3, :3] = Slerp(t, Rotation.from_matrix(poses[:, side, :3, :3]))(u).as_matrix()
        for axis in range(3):
            result[:, side, axis, 3] = np.interp(u, t, poses[:, side, axis, 3])
        for joint in range(20):
            q[:, side, joint] = np.interp(u, t, hands[:, side, joint])
    return result, q


def main():
    output = WORK/'replay_input_50hz.npz'
    assert not output.exists(), 'Refusing to overwrite prior conversion'
    paths = {name: WORK/'inputs'/name for name in ['wuji2_camera_replay_trajectory.npz', 'source_observation.json', 'zhuji_observation.json', 'zhuji_camera_41p5.json']}
    source = json.loads(paths['source_observation.json'].read_text())
    target = json.loads(paths['zhuji_observation.json'].read_text())
    camera = json.loads(paths['zhuji_camera_41p5.json'].read_text())
    for key in ['board_fingerprint', 'board_config', 'board_axes', 'convention']:
        assert source[key] == target[key], 'Board/convention mismatch: '+key
    assert source['camera_id'] == 'orbbec_gemini_335' and target['camera_id'] == 'zhuji'
    S, D, G = rigid(source['camera_T_board']), rigid(target['camera_T_board']), rigid(camera['T_base_camera'])
    for obs, pose in [(source, S), (target, D)]:
        np.testing.assert_allclose(rigid(obs['board_T_camera'])@pose, np.eye(4), atol=1e-9, rtol=0)
    M = rigid(D@np.linalg.inv(S)); chain = rigid(G@M)
    closure = float(np.max(abs(M@S-D)))
    assert closure < 1e-9
    model = ROOT/'replay_robot/models/tron2_wuji_board_v1_20261110'
    config_path = model/'alignment_config.json'
    config = json.loads(config_path.read_text())
    limits_path = ROOT/'teleop/wuji_control/config/joint_limits.json'
    limits = json.loads(limits_path.read_text())
    paths.update(config=config_path, limits=limits_path, converter=Path(__file__), tool_helper=ROOT/'replay_hand/build_board_aligned395.py', model_name_helper=ROOT/'replay_hand/build_exported_trajectory.py')
    original = np.load(paths['wuji2_camera_replay_trajectory.npz'], allow_pickle=False)
    expected = {'frame': 'undistorted_color_optical', 'pose_target': 'Wuji l_base_link / r_base_link; not FR3 flange',
                'quaternion_order': 'wxyz', 'position_unit': 'metre', 'joint_unit': 'radian',
                'alignment_version': 'human_camera_direct_fit_v1', 'model': 'Wuji hand2_beta early v2026.6.27'}
    for key, value in expected.items():
        assert str(original[key]) == value, 'Unsupported '+key
    assert original['side_order'].tolist() == ['left', 'right']
    t, stamps = original['time_s'].copy(), original['timestamp_ns'].copy()
    n = len(t)
    assert n >= 2 and t.shape == stamps.shape == (n,) and t[0] == 0
    assert np.isfinite(t).all() and np.all(np.diff(t)>0) and np.all(np.diff(stamps)>0)
    np.testing.assert_allclose(t, (stamps-stamps[0])/1e9, atol=1e-9, rtol=0)
    # Preserve the source's ~10 Hz clock; no fabricated dropped-frame repair.
    assert np.diff(t).max() < .11
    roots, boards, sdk, fingers, details = [], [], [], [], {}
    for side, name in enumerate(['left', 'right']):
        h = rigid(original[name+'_root_pose']); q = original[name+'_joint_positions_rad'].copy()
        assert h.shape == (n, 4, 4) and q.shape == (n, 20) and np.isfinite(q).all()
        np.testing.assert_allclose(original[name+'_position_m'], h[:, :3, 3], atol=1e-9, rtol=0)
        quat = original[name+'_quaternion_wxyz']
        assert quat.shape == (n,4) and np.isfinite(quat).all()
        np.testing.assert_allclose(np.linalg.norm(quat, axis=1), 1, atol=1e-8, rtol=0)
        np.testing.assert_allclose(Rotation.from_quat(quat[:, [1,2,3,0]]).as_matrix(), h[:, :3, :3], atol=1e-8, rtol=0)
        assert original[name+'_joint_names'].tolist() == model_names(name)
        bounds = np.asarray(limits[name])
        assert np.all(q >= bounds[:, 0]-1e-8) and np.all(q <= bounds[:, 1]+1e-8), 'Hand targets outside installed limits'
        calpath = model/'calibration'/(name+'.json'); paths[name+'_tool_calibration'] = calpath
        eb, bh = tool_frames(json.loads(calpath.read_text()), config['sides'][name]['R_board_center_hand'], config['board_center_in_board_m'])
        root = rigid(chain@h); board = rigid(root@np.linalg.inv(bh)); e = rigid(board@np.linalg.inv(eb))
        np.testing.assert_allclose(e@eb@bh, root, atol=1e-9, rtol=0)
        roots.append(root); boards.append(board); sdk.append(e); fingers.append(q)
        details[name] = dict(T_sdk_eef_board_center=eb.tolist(), T_board_center_hand=bh.tolist(),
                             target_xyz_min_m=e[:, :3, 3].min(0).tolist(), target_xyz_max_m=e[:, :3, 3].max(0).tolist(),
                             hand_speed_peak_rad_s=float(np.max(abs(np.diff(q,axis=0))/np.diff(t)[:,None])),
                             hand_limit_min_margin_rad=float(np.minimum(q-bounds[:,0],bounds[:,1]-q).min()))
    roots, boards, sdk, fingers = [np.stack(a, axis=1) for a in (roots, boards, sdk, fingers)]
    old_rel = np.linalg.inv(original['left_root_pose'])@original['right_root_pose']
    relative_error = float(abs(np.linalg.inv(roots[:,0])@roots[:,1]-old_rel).max())
    assert relative_error < 1e-9
    meta = dict(schema='replay_hand_cross_v1', frame='tron2_base_link', placement_policy='preserve_common_frame_v1',
                execution_ready=False, motion_commands_sent=0, source_stream=str(original['source_stream']),
                source_camera_id=source['camera_id'], source_frame=str(original['frame']),
                alignment_version=str(original['alignment_version']), source_files={str(p):sha(p) for p in paths.values()},
                target_head_pitch_yaw_rad=camera['new_head_pitch_yaw_rad'], controlled_hand_origin='B = Wuji base_link origin = confirmed board center',
                task_placement=dict(method='Corresponding observed task-board frames; no additional task repositioning',
                    target_camera_T_source_camera=M.tolist(), T_tron_base_source_camera=chain.tolist(),
                    board_fingerprint=source['board_fingerprint'], physical_validation_complete=False),
                per_arm_reanchoring_allowed=False, additional_rigid_task_placement_applied=False,
                hand_retargeting_applied=False, local_z_correction_applied=False, wrist_offset_added=False, adapter_length_added=False,
                bimanual_relative_matrix_error_max=relative_error,
                assumptions=['User-supplied Orbbec observation corresponds to this trajectory camera and unchanged mounting.',
                    'Undistorted color optical coordinates use the original optical axes; no extra rectification rotation.',
                    'Corresponding board placements define task origins and axes.',
                    'Existing 41.5-degree target camera-to-base estimate remains applicable; not independently physically validated.'],
                timestamp_semantics='Source device clock preserved; not asserted to be Unix epoch',
                time_preprocessing='Position linear / orientation Slerp / finger linear interpolation at original times; no duration scaling.',
                original_sample_max_gap_s=float(np.diff(t).max()), original_frames=n,
                sides=details)
    # Keep the 150 original samples as an audit artifact. The loader's 100 ms
    # rule rejects 100.24 ms source spacing, so provide an explicit 50 Hz adapter.
    def package(times, poses, hands, timestamps, root_poses=None, board_poses=None):
        eef = np.empty((len(times), 2, 7)); eef[:,:,:3] = poses[:,:,:3,3]
        for s in range(2):
            quat = Rotation.from_matrix(poses[:,s,:3,:3]).as_quat()
            for k in range(1,len(quat)):
                if quat[k]@quat[k-1]<0:quat[k]*=-1
            eef[:,s,3:] = quat
        out = dict(t=times,timestamp_ns=timestamps,eef=eef,hands=hands,metadata_json=np.asarray(json.dumps(meta)))
        if root_poses is not None:out['hand_root_T_tron_base']=root_poses
        if board_poses is not None:out['board_center_T_tron_base']=board_poses
        for name in ['left','right']:
            out[name+'_hand_joint_names']=np.asarray([name+'_finger%d_joint%d'%(f,j) for f in range(1,6) for j in range(1,5)])
        return out
    original_path = WORK/'converted_source_samples.npz'
    np.savez_compressed(original_path, **package(t,sdk,fingers,stamps,roots,boards))
    u = np.unique(np.r_[np.arange(0.,t[-1],.02),t[-1]])
    es, qs = resample(t,sdk,fingers,u)
    # Derive tool frames from interpolated SDK poses so fixed tool offsets are exact.
    hs, bs = [], []
    for name,s in [('left',0),('right',1)]:
        eb=np.asarray(details[name]['T_sdk_eef_board_center']);bh=np.asarray(details[name]['T_board_center_hand'])
        bs.append(es[:,s]@eb);hs.append(es[:,s]@eb@bh)
    values = package(u,es,qs,stamps[0]+np.rint(u*1e9).astype(np.int64),np.stack(hs,1),np.stack(bs,1))
    values.update(original_time_s=t,original_timestamp_ns=stamps,original_hands=fingers)
    np.savez_compressed(output, **values)
    loaded_t,loaded_eef,loaded_hands,_ = load_cross_episode(output,50.)
    np.testing.assert_allclose(loaded_t,u,atol=1e-12,rtol=0)
    np.testing.assert_allclose(loaded_eef[:,:,:3],values['eef'][:,:,:3],atol=1e-12,rtol=0)
    np.testing.assert_allclose(loaded_hands,qs,atol=1e-12,rtol=0)
    from movej_prepare import EEF_BOX
    outside=(sdk[:,:,:3,3]<EEF_BOX[None,:,:,0])|(sdk[:,:,:3,3]>EEF_BOX[None,:,:,1])
    result=dict(metadata=meta,source_frames=n,adapted_frames=len(u),duration_s=float(t[-1]),
                task_mapping_translation_m=M[:3,3].tolist(),task_mapping_rotation_deg=float(np.rad2deg(Rotation.from_matrix(M[:3,:3]).magnitude())),
                observation_closure_max_abs=closure,source_observation_quality=source['quality'],target_observation_quality=target['quality'],
                first_sdk_eef_xyz_m=sdk[0,:,:3,3].tolist(),servo_box_outside_source_frames_per_arm=outside.any(2).sum(0).tolist(),
                original_gap_exceeds_loader_limit_count=int(np.sum(np.diff(t)>.1)),
                source_sample_artifact=str(original_path),replay_input=str(output),replay_input_sha256=sha(output),
                loader_passed=True,extra_task_repositioning_applied=False,hardware_commands_sent=0)
    save(WORK/'conversion_report.json',result)
    save(WORK/'task_mapping_orbbec_to_zhuji.json',dict(target_camera_T_source_camera=M.tolist(),source_camera_T_target_camera=np.linalg.inv(M).tolist(),source_camera_id=source['camera_id'],target_camera_id=target['camera_id'],board_fingerprint=source['board_fingerprint'],formula='M = zhuji_camera_T_board @ inverse(orbbec_camera_T_board)',observation_closure_max_abs=closure))
    print(json.dumps({k:v for k,v in result.items() if k!='metadata'},indent=2),flush=True)


if __name__=='__main__':
    main()
