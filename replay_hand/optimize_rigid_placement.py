"""Offline joint/scene placement optimization; diagnostics, never execution."""
import json
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'replay'),str(ROOT/'tools')]
import replay as base
from transport import replay_backend
from planner_cross import saved_home, candidates
from search_rigid_placement import transform


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'replay_hand/data/episode395_board_aligned_v1.npz')
    parser.add_argument('--output',type=Path,default=ROOT/'replay_hand/alignment_20261110/rigid_search')
    parser.add_argument('--seed',choices=['saved_home','test_ik_ready','joint_midpoint'],default='saved_home')
    parser.add_argument('--initial-report',type=Path)
    parser.add_argument('--angles',type=float,nargs=3,default=[15,0,0])
    parser.add_argument('--center',type=float,nargs=3,default=[.45,-.05,-.10])
    args=parser.parse_args()
    with np.load(args.source,allow_pickle=False) as f:
        eef=f['eef'];times=f['t']
    indices=np.unique(np.linspace(0,len(times)-1,7).astype(int))
    target=eef[indices];n=len(indices);pivot=eef[0,:,:3].mean(axis=0)
    _,Model,_=replay_backend();model=Model();home=saved_home(ROOT)
    def placement(x):
        g=np.eye(4);g[:3,:3]=Rotation.from_euler('xyz',x[:3]).as_matrix();g[:3,3]=x[3:6]-g[:3,:3]@pivot
        return g
    out=args.output;out.mkdir(exist_ok=True,parents=True)
    xg=np.r_[np.deg2rad(args.angles),args.center]
    posed=transform(target,placement(xg));qs=[]
    seed=dict(candidates(ROOT,model))[args.seed].copy()
    for desired in posed:
        seed=base.solve_eef(model,seed.copy(),desired)
        qs.append(seed[:14].copy())
    initial=np.r_[xg,np.array(qs).ravel()]
    if args.initial_report:
        previous=json.loads(args.initial_report.read_text())
        initial=np.r_[np.deg2rad(previous['rotation_xyz_deg']),previous['first_eef_midpoint_m'],np.asarray(previous['sample_q16'])[:,:14].ravel()]
    lower=np.r_[np.deg2rad([-45,-45,-45]),[.30,-.2,-.4],np.tile(model.lower[:14]+.005,n)]
    upper=np.r_[np.deg2rad([45,45,45]),[.65,.2,.15],np.tile(model.upper[:14]-.005,n)]
    initial=np.clip(initial,lower+1e-7,upper-1e-7)
    def residual(x):
        desired=transform(target,placement(x));parts=[]
        for i,qarm in enumerate(x[6:].reshape(n,14)):
            q=home.copy();q[:14]=qarm;model.set_q(q);actual=model.poses()
            for s in range(2):
                parts.extend(actual[s,:3]-desired[i,s,:3])
                parts.extend(.3*(Rotation.from_quat(desired[i,s,3:])*Rotation.from_quat(actual[s,3:]).inv()).as_rotvec())
        # Small regularization selects among equivalent solutions, not an acceptance relaxation.
        return np.r_[parts,.00001*(x-initial)]
    sparsity=lil_matrix((12*n+len(initial),len(initial)),dtype=int)
    for i in range(n):
        sparsity[i*12:(i+1)*12,:6]=1
        sparsity[i*12:(i+1)*12,6+i*14:6+(i+1)*14]=1
    for i in range(len(initial)):sparsity[12*n+i,i]=1
    result=least_squares(residual,initial,bounds=(lower,upper),jac_sparsity=sparsity.tocsr(),
                         max_nfev=350,ftol=1e-9,xtol=1e-9,gtol=1e-9,x_scale='jac',verbose=1)
    g=placement(result.x);res=residual(result.x)[:12*n].reshape(n,2,6)
    report=dict(optimization_success=bool(result.success),message=result.message,nfev=result.nfev,
                source=str(args.source),
                rotation_xyz_deg=np.rad2deg(result.x[:3]).tolist(),first_eef_midpoint_m=result.x[3:6].tolist(),
                T_target_base_original_base=g.tolist(),sample_times_s=times[indices].tolist(),
                position_error_m=np.linalg.norm(res[:,:,:3],axis=-1).tolist(),
                rotation_error_rad=(np.linalg.norm(res[:,:,3:],axis=-1)/.3).tolist(),
                sample_q16=np.c_[result.x[6:].reshape(n,14),np.tile(home[14:],(n,1))].tolist(),
                execution_ready=False,motion_commands_sent=0)
    (out/'optimized.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
