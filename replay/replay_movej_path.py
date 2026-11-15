"""Replay-specific adaptive swept checks, independent of ROS and hardware.

Adapted from wanchi/replay_human_20261102/movej_path.py. Existing model
licenses are retained; SOURCES.json records the original file hash.
"""
import numpy as np
from replay_thresholds import MODEL_CLEARANCE_M


def check_segment(model,start,goal,pump=None):
    # Certify continuous intervals with the same bound; subdivide only when
    # the bound is too conservative. Endpoints are checked before recursion.
    if pump:pump()
    checks=[model.check_path(np.array([start]),clearance=MODEL_CLEARANCE_M),
            model.check_path(np.array([goal]),clearance=MODEL_CLEARANCE_M)]
    count=max(1,int(np.ceil(np.max(abs(goal[:14]-start[:14]))/.002)))
    points=np.linspace(start,goal,count+1)
    pending=list(zip(points[:-1],points[1:]))[::-1]
    while pending:
        if pump:pump()
        a,b=pending.pop()
        try:
            if hasattr(model,'check_interval'):
                checks.append(model.check_interval(a,b,clearance=MODEL_CLEARANCE_M))
            else:
                checks.append(model.check_path(np.array([a,b]),clearance=MODEL_CLEARANCE_M))
        except ValueError:
            if np.max(abs(b[:14]-a[:14]))<=.00001:raise
            middle=(a+b)/2
            pending.extend([(middle,b),(a,middle)])
    return min(c['conservative_min_clearance_m'] for c in checks)



def check_current_segment(model,read_state,goal,pump=None):
    """Certify the latest start without a fixed position-change or speed gate.

    A start displacement changes each interpolated joint pose by at most that
    displacement. The existing geometric radius bounds bound clearance loss.
    Recompute the path when that bound cannot retain the 5 mm requirement.
    """
    start,_=read_state();start=start.copy()
    geoms={g for pair in model.pairs for g in pair}
    radius={g:model.joint_radius_bounds(g,count=16) for g in geoms}
    radii=np.array([radius[g]+radius[h] for g,h in model.pairs])
    for attempt in range(3):
        target=goal.copy();target[14:]=start[14:]
        bound=check_segment(model,start,target,pump)
        if pump:pump()
        current,dq=read_state();current=current.copy()
        if np.any(current<model.lower+1e-5) or np.any(current>model.upper-1e-5):
            raise ValueError('Current start violates model joint limits')
        loss=float(np.max(radii @ np.abs(current-start)))
        if bound-loss>=MODEL_CLEARANCE_M:
            return current,dq,bound-loss,attempt+1
        start=current
    raise RuntimeError('Could not certify 5 mm clearance from changing start; no MoveJ sent')
