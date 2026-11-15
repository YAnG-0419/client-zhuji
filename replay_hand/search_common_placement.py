"""Offline translation search; never changes relative bimanual transforms."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'replay'), str(ROOT/'tools')]
import replay as base
from transport import replay_backend
from episode import pose_error
from planner_cross import saved_home


def main():
    source = ROOT/'replay_hand/data/episode395_board_aligned_v1.npz'
    with np.load(source) as f:
        eef=f['eef'].copy();t=f['t'].copy()
    _, Model, _ = replay_backend();model=Model();home=saved_home(ROOT)
    reports=[]
    out=ROOT/'replay_hand/alignment_20261110'
    out.mkdir(exist_ok=True)
    for dx in [-.10,-.20,-.30,-.35,-.40]:
        for dz in [0.,-.10,-.20]:
            shift=np.array([dx,.05,dz])
            targets=eef.copy();targets[:,:,:3]+=shift
            q=home.copy();worst_p=worst_r=0.;minimum=1.
            entry=dict(translation_m=shift.tolist(), coarse_valid=False)
            for index in np.unique(np.linspace(0,len(t)-1,20).astype(int)):
                q=base.solve_eef(model,q.copy(),targets[index]);model.set_q(q)
                p,r=pose_error(model.poses(),targets[index]);clearance,_=model.clearance()
                worst_p=max(worst_p,float(max(p)));worst_r=max(worst_r,float(max(r)));minimum=min(minimum,float(clearance))
                if max(p)>base.IK_POSITION_TOL or max(r)>base.IK_ROTATION_TOL or clearance<base.MODEL_CLEARANCE_M:
                    entry['failed_source_s']=float(t[index]);break
            else:
                entry['coarse_valid']=True
            entry.update(position_residual_m=worst_p,rotation_residual_rad=worst_r,clearance_m=minimum)
            reports.append(entry);print(json.dumps(entry),flush=True)
            (out/'common_placement_search.json').write_text(json.dumps(reports,indent=2)+'\n')


if __name__=='__main__':main()
