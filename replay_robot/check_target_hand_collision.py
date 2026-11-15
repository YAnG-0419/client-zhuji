"""Sampled inter-hand mesh distance at target B poses; no arm IK required.

Uses the frozen assembly's exact URDF collision geometry and its 40 finger
joints. This does not certify swept motion, self-collision, arms or environment.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pinocchio as pin
import hppfcl


def check(root, episode):
    model_dir=root/'replay_robot/models/tron2_wuji_board_v1_20261110'
    urdf=model_dir/'robot.urdf'
    model=pin.buildModelFromUrdf(str(urdf))
    geometry=pin.buildGeomFromUrdf(model,str(urdf),pin.GeometryType.COLLISION,package_dirs=[str(model_dir)])
    data=model.createData();gd=pin.GeometryData(geometry)
    folder=root/'replay_robot/task_transfer_20261110/episodes'/episode
    with np.load(folder/'replay_input.npz',allow_pickle=False) as z:
        hands=z['hands'].copy();roots=z['tron_base_T_hand'].copy();times=z['t'].copy()
    with np.load(root/'replay_robot/traj/sharpa_to_wuji2_beta'/episode/'wuji_trajectory.npz',allow_pickle=False) as z:
        indices=[]
        for side in ['left','right']:
            names=z[side+'_wuji_joint_names'].tolist()
            if not all(model.existJointName(n) for n in names):raise ValueError('Missing hand joint')
            indices.append([model.joints[model.getJointId(n)].idx_q for n in names])
    ids=[[i for i,o in enumerate(geometry.geometryObjects) if o.name.startswith(prefix)] for prefix in ['l_','r_']]
    if min(map(len,ids))<20:raise ValueError('Incomplete hand collision model')
    centers=[];radii=[]
    for side_ids in ids:
        cs=[];rs=[]
        for i in side_ids:
            shape=geometry.geometryObjects[i].geometry;shape.computeLocalAABB()
            cs.append(np.array(shape.aabb_center));rs.append(float(shape.aabb_radius))
        centers.append(np.array(cs));radii.append(np.array(rs))
    q=pin.neutral(model)
    minimum=float('inf');minimum_frame=0;minimum_pair=None;exact_queries=0
    failed=[];collisions=[];sample_lower=[]
    for k in range(len(times)):
        for side in range(2):q[indices[side]]=hands[k,side]
        pin.framesForwardKinematics(model,data,q);pin.updateGeometryPlacements(model,data,geometry,gd)
        placements=[];world_centers=[]
        for side,prefix in enumerate(['l','r']):
            target=pin.SE3(roots[k,side,:3,:3],roots[k,side,:3,3])
            change=target*data.oMf[model.getFrameId(prefix+'_base_link')].inverse()
            ts=[change*gd.oMg[i] for i in ids[side]]
            placements.append(ts)
            world_centers.append(np.array([m.rotation@c+m.translation for m,c in zip(ts,centers[side])]))
        gap=np.linalg.norm(world_centers[0][:,None]-world_centers[1][None],axis=-1)-radii[0][:,None]-radii[1][None]
        for i,j in np.argwhere(gap<.01):
            a,b=placements[0][i],placements[1][j]
            result=hppfcl.DistanceResult()
            gap[i,j]=hppfcl.distance(geometry.geometryObjects[ids[0][i]].geometry,hppfcl.Transform3f(a.rotation,a.translation),
                geometry.geometryObjects[ids[1][j]].geometry,hppfcl.Transform3f(b.rotation,b.translation),hppfcl.DistanceRequest(),result)
            exact_queries+=1
        i,j=np.unravel_index(np.argmin(gap),gap.shape);value=float(gap[i,j]);sample_lower.append(value)
        if value<minimum:
            minimum,minimum_frame,minimum_pair=value,k,[geometry.geometryObjects[ids[0][i]].name,geometry.geometryObjects[ids[1][j]].name]
        if value<.01:failed.append(k)
        if value<=0:collisions.append(k)
        if k%400==0:print(episode,'hand mesh check',k,'/',len(times),flush=True)
    report=dict(episode=episode,frames=len(times),collision_shapes=[len(x) for x in ids],
        scope='All original sampled target hand-to-hand URDF collision meshes; bounding-sphere broadphase and FCL mesh narrowphase',
        sampled_intersections_detected=bool(collisions),intersection_frames=len(collisions),
        all_sampled_hand_pairs_at_least_10mm=not failed,frames_below_10mm=len(failed),
        minimum_separation_lower_bound_m=minimum,minimum_source_s=float(times[minimum_frame]),minimum_pair=minimum_pair,
        exact_mesh_pair_queries=exact_queries,execution_ready=False,motion_commands_sent=0,
        not_checked=['swept interpolation','within-hand self-collision','arm-hand','approach','environment','adapter/cables'])
    (folder/'target_hand_collision.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(folder/'target_hand_collision_samples.npz',t=times,separation_lower_bound_m=sample_lower)
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--episode',required=True,choices=['episode594','episode615','episode616'])
    a=p.parse_args();check(a.root,a.episode)
