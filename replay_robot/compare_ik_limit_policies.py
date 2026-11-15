"""Offline comparison only; URDF limits are NOT a verified ServoP envelope.

No control modules, transports, ROS nodes or execution entry points are used.
Original replay inputs and live limit definitions are never modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def intervals(t,mask):
    ids=np.flatnonzero(mask)
    if not len(ids):return []
    chunks=np.split(ids,np.flatnonzero(np.diff(ids)>1)+1)
    return [dict(start_s=float(t[c[0]]),end_s=float(t[c[-1]]),frames=len(c)) for c in chunks]


def compare(root,episode):
    sys.path.insert(0,str(root/'Pinocchio'))
    from pin_model import PinArms,Q16
    source=root/'replay_robot/task_transfer_20261110/episodes'/episode
    out=root/'replay_robot/limit_policy_comparison_20261110'/episode
    if out.exists():raise FileExistsError('Refusing to overwrite comparison '+str(out))
    out.mkdir(parents=True)
    with np.load(source/'replay_input.npz',allow_pickle=False) as z:t=z['t'].copy();target=z['eef'].copy()
    with np.load(source/'ik_diagnostics_not_for_execution.npz',allow_pickle=False) as z:
        baseline_q=z['diagnostic_q16'].copy();baseline_valid=z['pose_pass'].copy()
    solver=PinArms();movej_lower=solver.lower.copy();movej_upper=solver.upper.copy()
    solver.lower=solver.model.lowerPositionLimit[solver.iq].copy()
    solver.upper=solver.model.upperPositionLimit[solver.iq].copy()
    assert np.all(solver.lower<=movej_lower) and np.all(solver.upper>=movej_upper)
    head=baseline_q[0,14:];rng=np.random.default_rng(61110)
    first_seeds=[baseline_q[0],np.r_[[0,0,0,-1.6,0,0,0]*2,head],np.r_[(solver.lower[:14]+solver.upper[:14])/2,head]]
    for k in range(9):first_seeds.append(np.r_[rng.uniform(solver.lower[:14]+.001,solver.upper[:14]-.001),head])

    def errors(q,goal):
        p,r=solver.fk(q)
        pe=np.linalg.norm(p-goal[:,:3],axis=1)
        re=(Rotation.from_matrix(r).inv()*Rotation.from_quat(goal[:,3:])).magnitude()
        return pe,re

    def quality(pe,re):
        # Prefer a passing result over any nonpassing weighted optimum.
        return (bool(pe>.010 or re>.015),float(pe*pe+(.3*re)**2))

    q=baseline_q[0].copy();first=[]
    for seed in first_seeds:
        work=np.clip(seed.copy(),solver.lower+1e-5,solver.upper-1e-5)
        for arm in range(2):work[7*arm:7*arm+7]=solver._bounded(work,arm,target[0])
        pe,re=errors(work,target[0]);first.append((work,pe,re))
    for arm in range(2):
        best=min(first,key=lambda row:quality(row[1][arm],row[2][arm]))
        q[7*arm:7*arm+7]=best[0][7*arm:7*arm+7]
    qs=[];pos=[];rot=[]
    for k,goal in enumerate(target):
        if k:
            for arm in range(2):q[7*arm:7*arm+7]=solver._bounded(q,arm,goal)
        pe,re=errors(q,goal)
        for arm in range(2):
            if pe[arm]>.010 or re[arm]>.015:
                alternate=baseline_q[k].copy()
                alternate[7*arm:7*arm+7]=solver._bounded(alternate,arm,goal)
                ap,ar=errors(alternate,goal)
                if quality(ap[arm],ar[arm])<quality(pe[arm],re[arm]):
                    q[7*arm:7*arm+7]=alternate[7*arm:7*arm+7];pe[arm]=ap[arm];re[arm]=ar[arm]
                if baseline_valid[k,arm] and (pe[arm]>.010 or re[arm]>.015):
                    q[7*arm:7*arm+7]=baseline_q[k,7*arm:7*arm+7]
                    pe,re=errors(q,goal)
        qs.append(q.copy());pos.append(pe.copy());rot.append(re.copy())
        if k%200==0:print(episode,'URDF-only diagnostic',k,'/',len(t),flush=True)
    qs,pos,rot=map(np.asarray,(qs,pos,rot));valid=(pos<=.010)&(rot<=.015)
    # A numerical solver cannot turn a known admissible baseline solution into
    # evidence of infeasibility under looser limits.
    assert np.all(valid[baseline_valid])
    outside=np.any((qs[:,:14]<movej_lower[:14])|(qs[:,:14]>movej_upper[:14]),axis=1)
    steps=np.max(abs(np.diff(qs[:,:14],axis=0)),axis=1)
    report=dict(episode=episode,frames=len(t),duration_s=float(t[-1]),motion_commands_sent=0,execution_ready=False,
        servop_joint_limits_verified=False,live_limits_changed=False,
        hypothesis='URDF-only mechanical model bounds, not certified ServoP bounds',
        original_movej_policy=dict(per_arm_passing_frames=baseline_valid.sum(axis=0).tolist(),both_passing_frames=int(baseline_valid.all(axis=1).sum())),
        urdf_only_policy=dict(per_arm_passing_frames=valid.sum(axis=0).tolist(),both_passing_frames=int(valid.all(axis=1).sum()),
            left_intervals=intervals(t,valid[:,0]),right_intervals=intervals(t,valid[:,1]),both_intervals=intervals(t,valid.all(axis=1)),
            first_position_mm=(pos[0]*1000).tolist(),first_rotation_deg=np.rad2deg(rot[0]).tolist(),
            final_position_mm=(pos[-1]*1000).tolist(),final_rotation_deg=np.rad2deg(rot[-1]).tolist(),
            all_frames_pass=bool(valid.all()),both_passing_outside_movej_frames=int((valid.all(axis=1)&outside).sum()),
            max_joint_step_rad=float(steps.max()),large_step_count_over_0p2rad=int((steps>.2).sum())),
        joint_limits={Q16[i]:dict(movej_lower=float(movej_lower[i]),movej_upper=float(movej_upper[i]),urdf_lower=float(solver.lower[i]),urdf_upper=float(solver.upper[i])) for i in range(14)},
        source_sha256={str(p):sha(p) for p in [source/'replay_input.npz',source/'ik_diagnostics_not_for_execution.npz',root/'Pinocchio/pin_model.py',Path(__file__)]},
        not_checked=['ServoP controller range','MoveJ-to-ServoP transition','continuous trajectory/collision','environment','hardware'])
    np.savez_compressed(out/'diagnostics_not_for_execution.npz',t=t,diagnostic_q16=qs,position_error_m=pos,rotation_error_rad=rot,pose_pass=valid,outside_movej=outside)
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(episode,json.dumps(report['urdf_only_policy']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--episode',required=True,choices=['episode594','episode615','episode616'])
    a=p.parse_args();compare(a.root,a.episode)
