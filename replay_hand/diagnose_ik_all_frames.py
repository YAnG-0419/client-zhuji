"""Offline per-frame IK diagnosis; never creates a device transport."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'replay'),str(ROOT/'tools')]
import replay as base
from transport import replay_backend
from episode import pose_error,ROBOT_NAMES
from planner_cross import candidates

def intervals(t,mask):
    ix=np.flatnonzero(mask);out=[]
    for run in np.split(ix,np.flatnonzero(np.diff(ix)>1)+1):
        if len(run):out.append([float(t[run[0]]),float(t[run[-1]]),len(run)])
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--pin-bounded',action='store_true');p.add_argument('--pin-analytic',action='store_true');p.add_argument('--start-index',type=int,default=0);p.add_argument('--end-index',type=int);a=p.parse_args()
    with np.load(a.episode,allow_pickle=False) as f:t=f['t'].copy();eef=f['eef'].copy()
    full_count=len(t);t=t[a.start_index:a.end_index];eef=eef[a.start_index:a.end_index]
    if not len(t):raise ValueError('Empty slice')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    progress=a.output.with_suffix('.jsonl').open('w',buffering=1)
    _,Model,_=replay_backend();m=Model();seeds=candidates(ROOT,m);previous=seeds[0][1].copy();rows=[];qs=[]
    fk_check={}
    if a.pin_bounded or a.pin_analytic:
        sys.path.insert(0,str(ROOT/'Pinocchio'))
        from pin_model import PinArms
        from scipy.spatial.transform import Rotation
        pin=PinArms()
        np.testing.assert_allclose(pin.lower,m.lower,atol=1e-5,rtol=0)
        np.testing.assert_allclose(pin.upper,m.upper,atol=1e-5,rtol=0)
        pin.lower=m.lower.copy();pin.upper=m.upper.copy()
        maxp=maxr=0.
        for _,qcheck in seeds:
            m.set_q(qcheck);old=m.poses();pcheck,rcheck=pin.fk(qcheck)
            maxp=max(maxp,float(np.max(abs(old[:,:3]-pcheck))))
            maxr=max(maxr,float(np.max((Rotation.from_quat(old[:,3:]).inv()*Rotation.from_matrix(rcheck)).magnitude())))
        if maxp>1e-6 or maxr>1e-6:raise ValueError('FK model mismatch')
        fk_check=dict(position_max_m=maxp,rotation_max_rad=maxr,limits_identical=True)
    def solve(seed,target):
        if not (a.pin_bounded or a.pin_analytic):return base.solve_eef(m,seed.copy(),target)
        q=np.clip(seed,pin.lower+2e-5,pin.upper-2e-5)
        for side in range(2):
            if not a.pin_analytic:
                q[side*7:side*7+7]=pin._bounded(q,side,target);continue
            import pinocchio as pinlib
            from scipy.optimize import least_squares
            sl=slice(side*7,side*7+7);rt=Rotation.from_quat(target[side,3:]).as_matrix()
            def fun(x):
                q[sl]=x;pos,rot=pin.fk(q)
                return np.r_[pos[side]-target[side,:3],.3*pinlib.log3(rot[side]@rt.T)]
            def jac(x):
                q[sl]=x;pos,rot=pin.fk(q);err=rot[side]@rt.T;j=pin.jacobian(q,side)
                return np.vstack([j[:3],.3*pinlib.Jlog3(err)@err.T@j[3:]])
            if 'analytic_jacobian_max_error' not in fk_check:
                x=q[sl].copy();eps=1e-6
                numeric=np.column_stack([(fun(x+np.eye(7)[i]*eps)-fun(x-np.eye(7)[i]*eps))/(2*eps) for i in range(7)])
                error=float(np.max(abs(numeric-jac(x))))
                if error>1e-6:raise ValueError('Jacobian check failed '+str(error))
                fk_check['analytic_jacobian_max_error']=error
            q[sl]=least_squares(fun,q[sl].copy(),jac=jac,bounds=(pin.lower[sl]+2e-5,pin.upper[sl]-2e-5),max_nfev=500,ftol=1e-11,xtol=1e-11,gtol=1e-11).x
        return q
    for k,target in enumerate(eef):
        pool=[]
        for name,seed in [('previous',previous)]+seeds:
            q=solve(seed,target);m.set_q(q);pe,re=pose_error(m.poses(),target)
            margins=np.minimum(q[:14]-m.lower[:14],m.upper[:14]-q[:14]).reshape(2,7).min(axis=1)
            ok=(pe<=base.IK_POSITION_TOL)&(re<=base.IK_ROTATION_TOL)&(margins>=1e-5-1e-10)
            score=np.maximum(pe/base.IK_POSITION_TOL,re/base.IK_ROTATION_TOL)+(margins<1e-5-1e-10)*100
            pool.append((q,score,name))
            if ok.all():break
        best=previous.copy();names=[]
        for s in range(2):
            item=min(pool,key=lambda x:x[1][s]);best[s*7:s*7+7]=item[0][s*7:s*7+7];names.append(item[2])
        m.set_q(best);pe,re=pose_error(m.poses(),target)
        delta=np.minimum(best[:14]-m.lower[:14],m.upper[:14]-best[:14]);margins=delta.reshape(2,7).min(axis=1)
        clearance,pair=m.clearance();step=np.max(abs(best[:14]-previous[:14])) if k else 0.
        rows.append(dict(t=float(t[k]),position_m=pe.tolist(),rotation_rad=re.tolist(),joint_margin_rad=margins.tolist(),near_limit_joints=[ROBOT_NAMES[i] for i in np.flatnonzero(delta<.0001)],pose_pass=((pe<=base.IK_POSITION_TOL)&(re<=base.IK_ROTATION_TOL)).tolist(),joint_pass=(margins>=1e-5-1e-10).tolist(),clearance_m=float(clearance),step_rad=float(step),seeds=names))
        progress.write(json.dumps(dict(frame_index=a.start_index+k,q16=best.tolist(),**rows[-1]))+'\n')
        previous=best;qs.append(best)
        if k%30==0:print('DIAGNOSTIC IK %d/%d'%(k+1,len(t)),flush=True)
    report=dict(solver='Pinocchio analytic bounded' if a.pin_analytic else ('Pinocchio bounded' if a.pin_bounded else 'replay original'),fk_check=fk_check,start_index=a.start_index,full_frame_count=full_count,episode=str(a.episode),frames=len(t),duration_s=float(t[-1]),motion_commands_sent=0,note='Independent per-frame multi-start IK; not continuous-path or collision certification',per_side={})
    for s,name in enumerate(['left','right']):
        pose=np.array([r['pose_pass'][s] for r in rows]);joint=np.array([r['joint_pass'][s] for r in rows])
        report['per_side'][name]=dict(pose_failure_intervals_s=intervals(t,~pose),joint_failure_intervals_s=intervals(t,~joint),combined_failure_intervals_s=intervals(t,~(pose&joint)),failed_frames=int(sum(~(pose&joint))))
    progress.close();report['rows']=rows;a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');np.savez_compressed(a.output.with_suffix('.npz'),t=t,q16=np.asarray(qs))
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
if __name__=='__main__':main()
