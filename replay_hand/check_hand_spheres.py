"""Offline conservative bounding-sphere check of inter-hand collision meshes.

Checks sampled targets only; does not certify arm-hand, environment or approach.
Uses original collision mesh extents, not fingertip-only point distances.
"""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


def stl_radius(path):
    raw=path.read_bytes()
    if len(raw)<84:raise ValueError('Invalid STL')
    count=int.from_bytes(raw[80:84],'little')
    if len(raw)!=84+50*count:raise ValueError('Expected binary STL')
    dtype=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')])
    vertices=np.frombuffer(raw,dtype=dtype,count=count,offset=84)['v'].reshape(-1,3)
    return float(np.linalg.norm(vertices,axis=1).max())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--episode',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'keypoints21-action-converter-20261105'))
    from sharpa21_to_wuji2 import Wuji2Skeleton
    with np.load(a.episode) as f:
        hands=f['hands'];poses=f['hand_root_T_tron_base'];times=f['t']
    radii=np.empty((len(times),2));counts={}
    for side_index,side in enumerate(['left','right']):
        model=Wuji2Skeleton(side);mesh_dir=a.model/'meshes'/('wuji_'+side)
        urdf=a.model/'sources'/('wuji_'+side+'_original.urdf')
        colliders=[]
        for link in ET.parse(urdf).getroot().findall('link'):
            body=mujoco.mj_name2id(model.model,mujoco.mjtObj.mjOBJ_BODY,link.get('name'))
            for c in link.findall('collision'):
                origin=c.find('origin');offset=np.zeros(3) if origin is None else np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
                g=c.find('geometry');mesh=g.find('mesh')
                if mesh is not None:
                    scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                    radius=stl_radius(mesh_dir/Path(mesh.get('filename')).name)*max(abs(scale))
                elif g.find('sphere') is not None:radius=float(g.find('sphere').get('radius'))
                elif g.find('box') is not None:radius=np.linalg.norm(np.fromstring(g.find('box').get('size'),sep=' '))/2
                elif g.find('cylinder') is not None:
                    cyl=g.find('cylinder');radius=np.hypot(float(cyl.get('radius')),float(cyl.get('length'))/2)
                else:raise ValueError('Unsupported collision shape')
                if body<0:raise ValueError('Missing collision body')
                colliders.append((body,float(radius+np.linalg.norm(offset))))
        counts[side]=len(colliders)
        if not colliders:raise ValueError('No hand collision geometry')
        for k,q in enumerate(hands[:,side_index]):
            model.fk(q)
            radii[k,side_index]=max(np.linalg.norm(model.data.xpos[b])+r for b,r in colliders)
    distance=np.linalg.norm(poses[:,0,:3,3]-poses[:,1,:3,3],axis=1)
    clearance=distance-radii.sum(axis=1)
    idx=int(clearance.argmin())
    report=dict(scope='Sampled hand-to-hand collision-mesh enclosing spheres only',
                all_sampled_hand_meshes_disjoint=bool(clearance.min()>0),
                minimum_conservative_separation_m=float(clearance.min()),source_s=float(times[idx]),
                max_enclosing_radius_m=radii.max(axis=0).tolist(),collision_shapes=counts,
                execution_ready=False,not_checked=['swept interpolation','arm-hand','approach','environment','adapters'])
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
