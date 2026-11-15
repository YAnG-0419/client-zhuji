"""Find a small, explicitly certified preparation route, without robot I/O."""
import numpy as np
from replay_thresholds import MODEL_CLEARANCE_M
import replay as base
from episode import pose_error


def candidates(start,goal,model=None):
    yield 'direct',np.stack([start,goal])
    if model is not None:
        model.set_q(start);pose=model.poses().copy()
        for forward,outward in ((.10,0.),(.15,.05),(.10,.10),(.20,.10)):
            target=pose.copy();target[:,0]+=forward;target[0,1]+=outward;target[1,1]-=outward
            mid=base.solve_eef(model,start.copy(),target);model.set_q(mid)
            p,r=pose_error(model.poses(),target)
            if max(p)<.001 and max(r)<.0035:
                yield 'hands_forward_%.2f_outward_%.2f'%(forward,outward),np.stack([start,mid,goal])
    for spread in (.4,.6):
        mid=start.copy();mid[1]=spread;mid[8]=-spread
        yield 'spread_shoulders_'+str(spread),np.stack([start,mid,goal])
    for angle in (-1.2,-1.6,-2.0):
        mid=start.copy();mid[[3,10]]=angle
        yield 'bend_both_elbows_'+str(angle),np.stack([start,mid,goal])
    ready=start.copy();ready[:14]=np.tile([0.,0.,0.,-1.6,0.,0.,0.],2)
    yield 'test_ik_ready',np.stack([start,ready,goal])


def plan_approach(model,checker,start,goal,hands):
    goal=np.asarray(goal).copy();goal[14:]=start[14:]
    attempts=[]
    for name,path in candidates(np.asarray(start),goal,model):
        detail=dict(name=name)
        try:
            detail['arm_path']=model.check_path(path,clearance=MODEL_CLEARANCE_M)
            detail['hand_body_path']=checker.check_path(path,np.repeat(np.asarray(hands)[None],len(path),axis=0))
            detail['accepted']=True;attempts.append(detail)
            return path,dict(selected=name,q16_waypoints=path.tolist(),attempts=attempts)
        except ValueError as e:
            detail.update(accepted=False,error=str(e));attempts.append(detail)
    error=ValueError('No certified hand-inclusive preparation route: '+str(attempts))
    error.attempts=attempts
    raise error
