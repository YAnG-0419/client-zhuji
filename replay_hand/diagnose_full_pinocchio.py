"""OFFLINE ONLY: run existing full replay certification with bounded Pinocchio IK.
Does not expose --check/--execute and does not modify the deployed entry point.
"""
import argparse,sys,json,time,os,hashlib,shutil
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'replay'),str(ROOT/'tools'),str(ROOT/'Pinocchio')]
import replay as base
from pin_model import PinArms
import pinocchio as pinlib
from full_sync import run
from episode import pose_error
from transport import replay_backend

def main():
 p=argparse.ArgumentParser();p.add_argument('--episode',type=Path,required=True);p.add_argument('--seed-cache',type=Path);a=p.parse_args()
 a.speed=1.;a.rate=300.;a.execute=False;a.check=False
 out=ROOT/'replay_hand/runs'/('offline_pin_full_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()));out.mkdir()
 pin=PinArms();_,Model,_=replay_backend();m=Model();pin.lower=m.lower.copy();pin.upper=m.upper.copy()
 report=dict(completed=False,execute_requested=False,arm_motion_commands_sent=0,hand_target_updates=0,solver='Offline-only bounded analytical Pinocchio; existing full_sync checks unchanged')
 known={}
 if a.seed_cache:
  import replay_hand as adapter
  from episode_cross import load_cross_episode
  old=json.loads((a.seed_cache/'result.json').read_text())['source']
  def cached_loader(path,hz,dest):
   t,eef,hands,meta=load_cross_episode(path,hz)
   if meta['episode_sha256']!=old['episode_sha256'] or meta['fk']!=old['fk'] or meta['placement_policy']!='preserve_common_frame_v1':raise ValueError('Seed cache input/model mismatch')
   with np.load(a.seed_cache/'cross_seed.npz',allow_pickle=False) as z:
    np.testing.assert_array_equal(t,z['t']);qs=z['q16'].copy()
   if qs.shape!=(len(t),16) or not np.isfinite(qs).all() or np.any(qs<m.lower+1e-5) or np.any(qs>m.upper-1e-5):raise ValueError('Invalid cached seed bounds')
   if np.max(abs(np.diff(qs[:,:14],axis=0)))>.2:raise ValueError('Cached seed discontinuity')
   for q,target in zip(qs,eef):
    m.set_q(q);ep,er=pose_error(m.poses(),target)
    if max(ep)>base.IK_POSITION_TOL or max(er)>base.IK_ROTATION_TOL:raise ValueError('Cached seed residual')
    known[target.tobytes()]=q.copy()
   m.set_q(qs[0])
   from replay_thresholds import MODEL_CLEARANCE_M
   if m.clearance()[0]<MODEL_CLEARANCE_M:raise ValueError('Cached first-frame clearance')
   meta['cross_anchor']=old['cross_anchor'];meta['cross_seed']=old['cross_seed']
   for name in ['cross_seed.npz','cross_seed.json']:shutil.copy2(a.seed_cache/name,dest/name)
   report['verified_seed_cache']=str(a.seed_cache)
   print('VERIFIED CACHED CONTINUOUS IK',len(t),'frames',flush=True)
   return t,eef,hands,qs,meta
  adapter.cross_loader=cached_loader
 def solve(model,seed,target):
  cached=known.get(target.tobytes())
  if cached is not None and np.array_equal(seed,cached):return cached.copy()
  q=np.clip(seed,pin.lower+2e-5,pin.upper-2e-5).copy()
  for side in range(2):
   sl=slice(side*7,side*7+7);rt=Rotation.from_quat(target[side,3:]).as_matrix()
   def fun(x):
    q[sl]=x;pos,rot=pin.fk(q)
    return np.r_[pos[side]-target[side,:3],.3*pinlib.log3(rot[side]@rt.T)]
   def jac(x):
    q[sl]=x;pos,rot=pin.fk(q);err=rot[side]@rt.T;j=pin.jacobian(q,side)
    return np.vstack([j[:3],.3*pinlib.Jlog3(err)@err.T@j[3:]])
   q[sl]=least_squares(fun,q[sl].copy(),jac=jac,bounds=(pin.lower[sl]+2e-5,pin.upper[sl]-2e-5),max_nfev=500,ftol=1e-11,xtol=1e-11,gtol=1e-11).x
  model.set_q(q);actual=model.poses();pos,rot=pin.fk(q)
  if np.max(abs(pos-actual[:,:3]))>1e-6 or np.max((Rotation.from_quat(actual[:,3:]).inv()*Rotation.from_matrix(rot)).magnitude())>1e-6:raise ValueError('FK mismatch')
  return q
 base.solve_eef=solve
 try:run(a,out,report);report['completed']=True
 except Exception as e:report['error']=type(e).__name__+': '+str(e);print(report['error'],flush=True)
 finally:(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print('REPORT',out/'result.json',flush=True)
 return 0 if report['completed'] else 1
if __name__=='__main__':raise SystemExit(main())
