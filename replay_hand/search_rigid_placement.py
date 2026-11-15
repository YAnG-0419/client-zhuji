"""Offline common SE(3) placement search for episode395, no device transport."""
import itertools
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'replay'), str(ROOT/'tools')]
import replay as base
from transport import replay_backend
from episode import pose_error
from planner_cross import saved_home
from movej_prepare import check_servop_box


def transform(eef, rigid):
    result=eef.copy()
    result[:,:,:3]=eef[:,:,:3] @ rigid[:3,:3].T+rigid[:3,3]
    for s in range(2):
        result[:,s,3:]=(Rotation.from_matrix(rigid[:3,:3])*Rotation.from_quat(eef[:,s,3:])).as_quat()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'replay_hand/data/episode395_board_aligned_v1.npz')
    parser.add_argument('--output',type=Path,default=ROOT/'replay_hand/alignment_20261110/rigid_search')
    parser.add_argument('--max-candidates',type=int,default=3)
    args=parser.parse_args();source=args.source
    with np.load(source) as f: eef=f['eef'].copy();t=f['t'].copy()
    _,Model,_=replay_backend();model=Model();home=saved_home(ROOT)
    pivot=eef[0,:,:3].mean(axis=0)
    out=args.output
    out.mkdir(exist_ok=True,parents=True)
    reports=[];accepted=[]
    samples=np.unique(np.linspace(0,len(t)-1,32).astype(int))
    rotations=sorted(itertools.product([10,15,20,5,25,0,-10,-20],[0,-5,5,-15,15],[0,-5,5,-15,15]),key=lambda x:abs(x[0]-10)+abs(x[1])+abs(x[2]))
    for angles in rotations:
      r=Rotation.from_euler('xyz',angles,degrees=True).as_matrix()
      for center in [[x,y,z] for x in [.45,.40,.50] for y in [0.,-.05,.05] for z in [-.15,-.10,-.20]]:
        rigid=np.eye(4);rigid[:3,:3]=r;rigid[:3,3]=center-r@pivot
        targets=transform(eef,rigid)
        report=dict(rotation_xyz_deg=list(angles),first_eef_midpoint_m=center,
                    T_target_base_original_base=rigid.tolist(),coarse_passed=False)
        try:
            for index in samples:check_servop_box(targets[index])
            q=home.copy();wp=wr=0.;minimum=1.
            for index in samples:
                q=base.solve_eef(model,q.copy(),targets[index]);model.set_q(q)
                p,er=pose_error(model.poses(),targets[index]);margin,pair=model.clearance()
                wp=max(wp,float(max(p)));wr=max(wr,float(max(er)));minimum=min(minimum,float(margin))
                report.update(last_source_s=float(t[index]),max_position_m=wp,max_rotation_rad=wr,min_clearance_m=minimum)
                if max(p)>(.001 if index==0 else base.IK_POSITION_TOL) or max(er)>(.0035 if index==0 else base.IK_ROTATION_TOL):
                    raise ValueError('IK residual')
                if margin<base.MODEL_CLEARANCE_M:raise ValueError('Arm collision clearance')
                if np.min(np.minimum(q[:14]-model.lower[:14],model.upper[:14]-q[:14]))<1e-5:
                    raise ValueError('Joint boundary')
            report['coarse_passed']=True
            accepted.append(report)
            print('COARSE CANDIDATE '+json.dumps(report),flush=True)
        except ValueError as e:report['failure']=str(e)
        reports.append(report)
        (out/'search.json').write_text(json.dumps(dict(attempts=reports,accepted=accepted),indent=2)+'\n')
        if len(reports)%15==0:print('RIGID SEARCH %d; accepted=%d'%(len(reports),len(accepted)),flush=True)
        if len(accepted)>=args.max_candidates:return


if __name__=='__main__':main()
