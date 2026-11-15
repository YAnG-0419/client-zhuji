"""Convert a declared Wuji-base export to B-point/SDK poses in SOURCE scene.

No camera mapping, root retargeting, wrist shift or package Z correction is
applied. A separate common rigid task placement is required before playback.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from build_board_aligned395 import tool_frames

SIDES=('left','right')


def model_names(side):
    return [side[0]+'_'+finger+'_'+joint
            for finger in ('thumb','index_finger','middle_finger','ring_finger','pinky')
            for joint in (('cmc_flex','cmc_abd','mcp','ip') if finger=='thumb' else ('mcp_flex','mcp_abd','pip','dip'))]


def convert(source,output,config_dir,limits_path):
    config=json.loads((config_dir/'alignment_config.json').read_text())
    limits=json.loads(limits_path.read_text())
    provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (source,config_dir/'alignment_config.json',limits_path)}
    with np.load(source,allow_pickle=False) as a:
        expected={'frame':'lychee_root','quaternion_order':'wxyz','position_unit':'metre','joint_unit':'radian',
                  'alignment_version':'wuji_beta_local_z_left_p90_right_m90_v1',
                  'model':'Wuji hand2_beta early v2026.6.27',
                  'pose_target':'Wuji l_base_link / r_base_link; not FR3 flange'}
        for key,value in expected.items():
            if str(a[key])!=value:raise ValueError('Unsupported '+key+': '+str(a[key]))
        if a['side_order'].tolist()!=list(SIDES):raise ValueError('Unexpected side order')
        t=a['time_s'].copy();stamps=a['timestamp_ns'].copy();n=len(t)
        if (t.ndim!=1 or n<2 or not np.isfinite(t).all() or abs(t[0])>1e-9 or
            np.any(np.diff(t)<=0) or np.max(np.diff(t))>.1 or stamps.shape!=(n,) or
            np.any(np.diff(stamps)<=0) or not np.allclose(t,(stamps-stamps[0])/1e9,atol=1e-9,rtol=0)):
            raise ValueError('Invalid time axis')
        eef=np.empty((n,2,7));hands=np.empty((n,2,20));roots=[];boards=[];details={}
        for i,s in enumerate(SIDES):
            h=a[s+'_root_pose'].copy();q=a[s+'_joint_positions_rad'].copy()
            if h.shape!=(n,4,4) or q.shape!=(n,20) or not np.isfinite(h).all() or not np.isfinite(q).all():
                raise ValueError('Invalid pose/joint arrays')
            r=h[:,:3,:3]
            if (not np.allclose(h[:,3],[0,0,0,1],atol=1e-10,rtol=0) or
                not np.allclose(r.transpose(0,2,1)@r,np.eye(3),atol=1e-8,rtol=0) or
                not np.allclose(np.linalg.det(r),1.,atol=1e-8,rtol=0)):
                raise ValueError('Invalid rigid root pose')
            wxyz=a[s+'_quaternion_wxyz']
            if wxyz.shape!=(n,4) or not np.isfinite(wxyz).all() or not np.allclose(np.linalg.norm(wxyz,axis=1),1.,atol=1e-8,rtol=0):
                raise ValueError('Invalid quaternion')
            if not np.allclose(Rotation.from_quat(wxyz[:,[1,2,3,0]]).as_matrix(),r,atol=1e-8,rtol=0):
                raise ValueError('Quaternion/matrix disagreement')
            if not np.allclose(a[s+'_position_m'],h[:,:3,3],atol=1e-9,rtol=0):raise ValueError('Position/matrix disagreement')
            if a[s+'_joint_names'].tolist()!=model_names(s):raise ValueError('Unexpected model joint order')
            bounds=np.asarray(limits[s])
            if np.any(q<bounds[:,0]-1e-8) or np.any(q>bounds[:,1]+1e-8):raise ValueError('Hand command outside control limits')
            cp=config_dir/'calibration'/(s+'.json');provenance[str(cp)]=hashlib.sha256(cp.read_bytes()).hexdigest()
            e_b,b_h=tool_frames(json.loads(cp.read_text()),config['sides'][s]['R_board_center_hand'],config['board_center_in_board_m'])
            board=h@np.linalg.inv(b_h);target=board@np.linalg.inv(e_b)
            error=float(abs(target@e_b@b_h-h).max())
            if error>1e-9:raise ValueError('Tool roundtrip failed')
            eef[:,i,:3]=target[:,:3,3];eef[:,i,3:]=Rotation.from_matrix(target[:,:3,:3]).as_quat()
            hands[:,i]=q;roots.append(h);boards.append(board)
            details[s]=dict(T_eef_B=e_b.tolist(),T_B_hand=b_h.tolist(),roundtrip_max=error,
                            root_position_preserved=True,joint_values_unchanged=True)
        meta=dict(schema='replay_hand_cross_v1',frame='lychee_root',placement_policy='source_scene_unplaced',
                  execution_ready=False,motion_commands_sent=0,source_files=provenance,sides=details,
                  alignment_version=str(a['alignment_version']),package_alignment_reapplied=False,
                  wrist_offset_added=False,adapter_length_added=False,
                  hand_source='Original exported Wuji q20, unchanged; mapped by verified model/control names')
        data=dict(t=t,timestamp_ns=stamps,eef=eef,hands=hands,
                  hand_root_T_source_scene=np.stack(roots,axis=1),board_center_T_source_scene=np.stack(boards,axis=1),
                  metadata_json=np.asarray(json.dumps(meta)))
        for s in SIDES:data[s+'_hand_joint_names']=np.asarray([s+'_finger%d_joint%d'%(f,j) for f in range(1,6) for j in range(1,5)])
        output.parent.mkdir(exist_ok=True,parents=True);np.savez_compressed(output,**data)
        output.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(json.dumps(dict(output=str(output),frames=n,duration_s=t[-1],frame='lychee_root',sides=details),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    root=Path(__file__).resolve().parents[1]
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parent/'alignment_20261110')
    p.add_argument('--limits',type=Path,default=root/'keypoints21-action-converter-20261105/assets/wuji2/joint_limits.json')
    args=p.parse_args();convert(args.input,args.output,args.config,args.limits)
