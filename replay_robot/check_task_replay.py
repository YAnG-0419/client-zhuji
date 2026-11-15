"""Offline-only diagnostics. Invalid IK samples are never execution plans.

Reuses deployed Pinocchio FK/bounded IK and replay limits. Target poses are
never moved, clipped or re-anchored to make a failed trajectory appear valid.
"""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def save(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def diagnose(root, work, episode):
    sys.path[:0]=[str(root/'Pinocchio'),str(root/'replay'),str(root/'tools'),str(root/'replay_hand')]
    import pinocchio as pin
    from pin_model import PinArms
    from episode_cross import load_cross_episode
    from planner_cross import saved_home
    from movej_prepare import EEF_BOX
    folder=work/'episodes'/episode
    loaded=load_cross_episode(folder/'replay_input.npz',50.)
    with np.load(folder/'replay_input.npz',allow_pickle=False) as z:
        t,targets,hands=z['t'].copy(),z['eef'].copy(),z['hands'].copy()
        meta=json.loads(str(z['metadata_json']))
    report=dict(episode=episode,execution_ready=False,motion_commands_sent=0,
                current_loader_passed=True,resampled_frames_50hz=len(loaded[0]),
                original_frames=len(t),source_duration_s=float(t[-1]),
                ik_tolerances=dict(position_m=.010,rotation_rad=.015),
                interpolation_and_swept_collision_checked=False,
                diagnostic_q_not_an_execution_plan=True)
    xyz=targets[:,:,:3]
    outside=np.any((xyz<EEF_BOX[None,:,:,0])|(xyz>EEF_BOX[None,:,:,1]),axis=(1,2))
    report['servop_workspace']=dict(all_source_frames_inside=bool(not outside.any()),outside_frames=int(outside.sum()))
    hand_speed=np.abs(np.diff(hands,axis=0))/np.diff(t)[:,None,None]
    report['hand_timing']=dict(peak_original_rad_s=float(hand_speed.max()),
        speed_limit_rad_s=1., uniform_speed_ceiling=float(min(1.,1/hand_speed.max())),
        minimum_duration_without_arm_constraints_s=float(np.maximum(np.diff(t),np.max(np.abs(np.diff(hands,axis=0)),axis=(1,2))).sum()),
        synchronized_execution_clock_required=True)
    # Independent source FK checks the recorded joint-name remapping, removing
    # the nominal base, and all Sharpa installation transforms used by the package.
    source=root/'replay_robot/traj/sharpa_to_wuji2_beta'/episode
    src=pin.buildModelFromUrdf(str(source.parent/'assets/motion/source_assembly.urdf'))
    sd=src.createData();sq=pin.neutral(src);worst=0.
    with np.load(source/'source_sharpa.npz',allow_pickle=False) as raw, np.load(source/'wuji_motion_scene.npz',allow_pickle=False) as scene:
        maps=[]
        for side in ['left','right']:
            ids=[]
            for name in raw[side+'_arm_names'].tolist():
                name=name.replace('_fr3_joint','_fr3v2_joint')
                if not src.existJointName(name):raise ValueError('Missing source joint '+name)
                ids.append(src.joints[src.getJointId(name)].idx_q)
            maps.append(ids)
        for k in range(len(t)):
            for side,ids in zip(['left','right'],maps):sq[ids]=raw[side+'_arm_q'][k]
            pin.framesForwardKinematics(src,sd,sq)
            for side in ['left','right']:
                fk=sd.oMf[src.getFrameId(side+'_hand_C_MC')].homogeneous
                worst=max(worst,float(np.abs(fk-scene[side+'_scene_T_sharpa_root'][k]).max()))
    if worst>1e-8:raise ValueError('Source FK mismatch %.6g'%worst)
    report['source_independent_fk_max_abs']=worst
    solver=PinArms();head=np.asarray(meta['target_head_pitch_yaw_rad'])
    home=saved_home(root);home[14:]=head
    seeds=[('saved_home',home),('ready',np.r_[[0,0,0,-1.6,0,0,0]*2,head]),
           ('midpoint',np.r_[(solver.lower[:14]+solver.upper[:14])/2,head])]
    existing=json.loads((folder/'existing_cross_seed_failure.json').read_text())
    for row in existing['attempts']:
        if 'solved_q16' in row:
            q=np.array(row['solved_q16']);q[14:]=head;seeds.append(('existing_'+row['name'],q))
    rng=np.random.default_rng(594615616)
    for k in range(6):seeds.append(('random_%d'%k,np.r_[rng.uniform(solver.lower[:14]+.01,solver.upper[:14]-.01),head]))

    def residual(q,target):
        p,r=solver.fk(q)
        ep=np.linalg.norm(p-target[:,:3],axis=1)
        er=(Rotation.from_matrix(r).inv()*Rotation.from_quat(target[:,3:])).magnitude()
        return ep,er

    attempts=[];solutions=[]
    for name,q in seeds:
        q=np.clip(q,solver.lower+1e-5,solver.upper-1e-5)
        for arm in range(2):q[7*arm:7*arm+7]=solver._bounded(q,arm,targets[0])
        ep,er=residual(q,targets[0]);cost=ep**2+(.3*er)**2
        attempts.append(dict(seed=name,position_m=ep.tolist(),rotation_rad=er.tolist(),q16=q.tolist()))
        solutions.append((q,cost))
    # The arms have independent kinematic chains; combine each side's best seed.
    q=home.copy()
    for arm in range(2):
        best=min(solutions,key=lambda a:a[1][arm])[0];q[7*arm:7*arm+7]=best[7*arm:7*arm+7]
    report['independent_first_frame_attempts']=attempts
    ep,er=residual(q,targets[0]);report['independent_first_frame_best']=dict(position_m=ep.tolist(),rotation_rad=er.tolist(),q16=q.tolist())
    qs=[];positions=[];rotations=[]
    for k,target in enumerate(targets):
        if k:
            for arm in range(2):q[7*arm:7*arm+7]=solver._bounded(q,arm,target)
        ep,er=residual(q,target);qs.append(q.copy());positions.append(ep);rotations.append(er)
        if k%150==0:print(episode,'independent source-frame IK',k,'/',len(t),flush=True)
    qs,positions,rotations=map(np.asarray,(qs,positions,rotations))
    valid=(positions<=.010)&(rotations<=.015)
    report['all_original_frame_ik']=dict(per_arm_passing_frames=valid.sum(axis=0).tolist(),
        both_arms_passing_frames=int(valid.all(axis=1).sum()), all_frames_pass=bool(valid.all()),
        per_arm_max_position_m=positions.max(axis=0).tolist(),per_arm_max_rotation_rad=rotations.max(axis=0).tolist(),
        first_invalid_source_s=float(t[np.flatnonzero(~valid.all(axis=1))[0]]) if not valid.all() else None,
        max_adjacent_joint_change_rad=float(np.abs(np.diff(qs[:,:14],axis=0)).max()),
        method='Pinocchio bounded least squares; previous-solution initialization; 12 first-frame seeds; current SDK/URDF limit intersection')
    dt=np.maximum.reduce([np.diff(t),np.max(abs(np.diff(qs[:,:14],axis=0)),axis=1)/.3,np.max(abs(np.diff(hands,axis=0)),axis=(1,2))])
    report['diagnostic_timing']=dict(speed_limited_duration_s=float(dt.sum()),
        arm_speed_limit_rad_s=.3,hand_speed_limit_rad_s=1.,execution_usable=bool(valid.all()),
        note='Invalid IK samples make this duration diagnostic only; acceleration and valid-path timing cannot be certified.')
    report['pending']=[] if valid.all() else ['Valid full arm path unavailable; arm/hand/body swept collision and measured-start approach depend on solving reachability.']
    report['pending']+=['Environment and measured adapter/cable geometry are not available.',
        'Source workcell/Sharpa mount applicability and physical target accuracy remain unverified.',
        'Current-state approach and device/hand tracking require later live preflight.']
    np.savez_compressed(folder/'ik_diagnostics_not_for_execution.npz',t=t,diagnostic_q16=qs,
                        position_error_m=positions,rotation_error_rad=rotations,pose_pass=valid)
    save(folder/'offline_check.json',report)
    print(episode,json.dumps(report['all_original_frame_ik']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--episode',required=True,choices=['episode594','episode615','episode616'])
    a=p.parse_args();diagnose(a.root,a.root/'replay_robot/task_transfer_20261110',a.episode)
