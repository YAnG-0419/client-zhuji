"""Apply one rigid task placement to both arms and all tool reference poses."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def apply(source, output, rigid, provenance):
    g=np.asarray(rigid,dtype=float)
    if (g.shape!=(4,4) or not np.isfinite(g).all() or
        not np.allclose(g[3],[0,0,0,1],atol=1e-12) or
        not np.allclose(g[:3,:3].T@g[:3,:3],np.eye(3),atol=1e-10) or
        abs(np.linalg.det(g[:3,:3])-1)>1e-10):
        raise ValueError('Task placement must be a proper rigid transform')
    with np.load(source,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
    meta=json.loads(str(data['metadata_json']))
    if meta.get('placement_policy') not in ('preserve_common_frame_v1','source_scene_unplaced') or 'task_placement' in meta:
        raise ValueError('Expected original common-frame conversion; no repeated placement')
    source_frame=meta.get('frame')
    source_suffix='source_scene' if meta['placement_policy']=='source_scene_unplaced' else 'tron_base'
    if (source_suffix=='source_scene' and source_frame!='lychee_root') or (source_suffix=='tron_base' and source_frame!='tron2_base_link'):
        raise ValueError('Unknown source frame')
    old=data['hand_root_T_'+source_suffix].copy()
    eef=data['eef'];eef[:,:,:3]=eef[:,:,:3]@g[:3,:3].T+g[:3,3]
    for side in range(2):
        eef[:,side,3:]=(Rotation.from_matrix(g[:3,:3])*Rotation.from_quat(eef[:,side,3:])).as_quat()
    for prefix in ['hand_root_T_','board_center_T_']:
        data[prefix+'tron_base']=g@data[prefix+source_suffix]
    new=data['hand_root_T_tron_base']
    relative_error=float(abs(np.linalg.inv(old[:,0])@old[:,1]-np.linalg.inv(new[:,0])@new[:,1]).max())
    if relative_error>1e-9:raise AssertionError('Bimanual relative pose changed')
    meta.update(frame='tron2_base_link',placement_policy='preserve_common_frame_v1',execution_ready=False,
        task_placement=dict(T_target_base_original_base=g.tolist(),source_frame=source_frame,
        provenance=provenance,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        bimanual_relative_matrix_error_max=relative_error,
        semantics='Offline task placement, not a measured camera calibration'))
    data['metadata_json']=np.asarray(json.dumps(meta))
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**data)
    output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--search',type=Path,required=True);args=p.parse_args()
    search=json.loads(args.search.read_text());chosen=search['accepted'][0] if 'accepted' in search else search
    m=apply(args.source,args.output,chosen['T_target_base_original_base'],str(args.search))
    print(json.dumps(m['task_placement'],indent=2))
