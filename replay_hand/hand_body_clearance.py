"""Confirmed assembly mesh clearance: hands against each other and robot body.

Complements the existing arm/body/head checker. Does not model environment,
cables, or the illustrative adapter, and does not certify within-hand contacts.
Swept radius bound adapted from wanchi arm_model.py (Apache-2.0 model sources).
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import time
import numpy as np
from replay_thresholds import MODEL_CLEARANCE_M
import mujoco
from scipy.spatial.transform import Rotation

ARM_NAMES=[p+'_'+s+'_Joint' for s in ('L','R') for p in
           ('proximal_pitch','proximal_roll','proximal_yaw','elbow','wrist_yaw','wrist_pitch','wrist_roll')]
NAMES=ARM_NAMES+['head_pitch_Joint','head_yaw_Joint']
for side in ('l','r'):
    for finger in ('thumb','index_finger','middle_finger','ring_finger','pinky'):
        for suffix in (('cmc_flex','cmc_abd','mcp','ip') if finger=='thumb' else ('mcp_flex','mcp_abd','pip','dip')):
            NAMES.append(side+'_'+finger+'_'+suffix)


class HandBodyClearance:
    def __init__(self,path):
        path=Path(path).resolve();root=ET.parse(path).getroot()
        compiler=root.find('./mujoco/compiler')
        compiler.set('meshdir',str(path.parent));compiler.set('strippath','false')
        compiler.set('fusestatic','false');compiler.set('discardvisual','true')
        for i,c in enumerate(root.findall('./link/collision')):c.set('name','collision_'+str(i))
        self.model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
        if not hasattr(mujoco.mjtEnableBit,'mjENBL_NATIVECCD'):raise RuntimeError('Native CCD required')
        self.model.opt.enableflags|=mujoco.mjtEnableBit.mjENBL_NATIVECCD
        self.data=mujoco.MjData(self.model)
        self.jids=np.array([mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in NAMES])
        if np.any(self.jids<0) or len(set(self.jids))!=56:raise ValueError('Incomplete 56-joint map')
        self.qadr=self.model.jnt_qposadr[self.jids]
        self.bodies=[mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_BODY,int(b)) or 'world' for b in self.model.geom_bodyid]
        self.pairs=[]
        for g in range(self.model.ngeom):
            for h in range(g+1,self.model.ngeom):
                a,b=self.bodies[g],self.bodies[h]
                sa=a[:2] if a.startswith(('l_','r_')) else None
                sb=b[:2] if b.startswith(('l_','r_')) else None
                if (not sa and not sb) or (sa and sa==sb):continue
                # The palm/wrist and its own attachment carrier form a fixed assembly.
                if any((a in (s+'_base_link',s+'_wrist') and b=='wrist_roll_'+s.upper()+'_Link') or
                       (b in (s+'_base_link',s+'_wrist') and a=='wrist_roll_'+s.upper()+'_Link') for s in ('l','r')):continue
                self.pairs.append((g,h))
        if not self.pairs:raise ValueError('No hand/body collision pairs')
        self.radius=np.array([self.radius_bound(g) for g in range(self.model.ngeom)])
        self.gs=np.array([g for g,h in self.pairs]);self.hs=np.array([h for g,h in self.pairs])
        self.pair_radius=self.radius[self.gs]+self.radius[self.hs]

    def radius_bound(self,g):
        bound=np.zeros(56)
        distance=np.linalg.norm(self.model.geom_pos[g])+self.model.geom_rbound[g]
        b=int(self.model.geom_bodyid[g])
        while b:
            for i,j in enumerate(self.jids):
                if self.model.jnt_bodyid[j]==b:bound[i]=distance+np.linalg.norm(self.model.jnt_pos[j])
            distance+=np.linalg.norm(self.model.body_pos[b]);b=int(self.model.body_parentid[b])
        return bound

    def set_q(self,q16,hands):
        q=np.r_[q16,np.asarray(hands).ravel()]
        if q.shape!=(56,) or not np.isfinite(q).all():raise ValueError('Expected finite q16 and hands[2,20]')
        self.data.qpos[self.qadr]=q;mujoco.mj_forward(self.model,self.data)

    def poses(self):
        poses=[]
        for side in ('L','R'):
            b=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'sdk_eef_'+side)
            poses.append(np.r_[self.data.xpos[b],Rotation.from_matrix(self.data.xmat[b].reshape(3,3)).as_quat()])
        return np.array(poses)

    def clearance(self,delta=None):
        delta=np.zeros(56) if delta is None else abs(np.asarray(delta))
        allowances=np.sum(self.pair_radius*delta,axis=1)/2
        lower=(np.linalg.norm(self.data.geom_xpos[self.gs]-self.data.geom_xpos[self.hs],axis=1)
               -self.model.geom_rbound[self.gs]-self.model.geom_rbound[self.hs]-allowances)
        best=float('inf');pair=None
        for index in np.argsort(lower):
            if lower[index]>=best:break
            g,h=self.pairs[index];allowance=float(allowances[index])
            distance=float(mujoco.mj_geomDistance(self.model,self.data,g,h,.5+allowance,None))-allowance
            if distance<best:best=distance;pair=[self.bodies[g],self.bodies[h]]
        return best,pair

    def check_path(self,qs,hands,required=MODEL_CLEARANCE_M,heartbeat=None):
        required=max(required,MODEL_CLEARANCE_M)
        values=np.c_[qs,np.asarray(hands).reshape(len(qs),40)]
        minimum=float('inf');pair=None;count=0;last_heartbeat=time.monotonic()
        for i in range(len(values)):
            a,b=values[max(0,i-1)],values[i]
            n=max(1,int(np.ceil(np.max(abs(b-a))/.01)))
            for k in range(n):
                mid=a+(b-a)*((k+.5)/n);self.set_q(mid[:16],mid[16:].reshape(2,20))
                bound,p=self.clearance((b-a)/n);count+=1
                if bound<minimum:minimum=bound;pair=p
                if bound<required:
                    raise ValueError('Hand/body swept collision at frame %d: %.6fm %s'%(i,bound,p))
                if heartbeat is not None and time.monotonic()-last_heartbeat>=.02:
                    heartbeat();last_heartbeat=time.monotonic()
            if i and i%200==0:print('HAND/BODY CLEARANCE %d/%d'%(i,len(values)),flush=True)
        return dict(samples=count,min_swept_clearance_m=minimum,closest_bodies=pair,required_m=required,
                    scope='Inter-hand and hand-versus-arms/body/head meshes; own fixed palm carrier excluded',
                    excluded=['within-hand contacts','environment','cables','adapter geometry'])
