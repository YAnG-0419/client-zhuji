"""Archive the confirmed assembly. File operations and offline kinematics only."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path('/home/guest/pengyang')
OUT=ROOT/'replay_robot/models/tron2_wuji_board_v1_20261110'
assert not OUT.exists(), 'Version already exists; do not overwrite'
OUT.mkdir(parents=True)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def origin(T):
    return dict(xyz=' '.join('%.16g'%v for v in T[:3,3]),rpy=' '.join('%.16g'%v for v in Rotation.from_matrix(T[:3,:3]).as_euler('xyz')))
def add_frame(robot,parent,child,T):
    ET.SubElement(robot,'link',name=child)
    j=ET.SubElement(robot,'joint',name=child+'_fixed',type='fixed')
    ET.SubElement(j,'parent',link=parent);ET.SubElement(j,'child',link=child)
    ET.SubElement(j,'origin',**origin(T))

source_files={}
def snapshot(source,dest):
    p=OUT/dest;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,p)
    source_files[str(source)]=dict(sha256=sha(source),snapshot=dest)

body=ROOT/'record/config/model/robot.urdf'
robot=ET.parse(body).getroot();robot.set('name','tron2_wuji_board_confirmed_v1')
snapshot(body,'sources/tron2_original.urdf')
snapshot(ROOT/'record/config/model/robot_description_LICENSE','licenses/TRON2_LICENSE')
snapshot(ROOT/'replay_robot/traj/sharpa_to_wuji2_beta/WUJI_MODEL_LICENSE','licenses/WUJI_MODEL_LICENSE')
snapshot(ROOT/'replay_robot/traj/sharpa_to_wuji2_beta/WUJI_VENDORED_LICENSE','licenses/WUJI_VENDORED_LICENSE')
snapshot(ROOT/'replay_robot/board_frame_alignment.json','sources/board_frame_alignment_original.json')
snapshot(ROOT/'replay_robot/adapter_observation_20261110.json','adapter_observation.json')
config=json.loads((ROOT/'replay_robot/board_frame_alignment.json').read_text())
notes={}
mesh_count=0

def relocate_meshes(doc,source_urdf,category):
    global mesh_count
    for mesh in doc.findall('.//mesh'):
        name=mesh.get('filename')
        if name.startswith('package://'):
            src=Path('/home/guest/wanchi/code/replay_eef/vendor/tron2-robot-description/tron2a/DACH_TRON2A/meshes')/Path(name).name
        else:src=(source_urdf.parent/name).resolve()
        rel='meshes/'+category+'/'+src.name
        dst=OUT/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            shutil.copyfile(src,dst);mesh_count+=1
            source_files[str(src)]=dict(sha256=sha(src),snapshot=rel)
        else:assert sha(src)==sha(dst)
        mesh.set('filename',rel)

# Remove only the original grasper leaves. Wuji hands replace those visuals.
for letter in 'LR':
    for child in list(robot):
        if child.get('name') in ('grasper_'+letter+'_Link','grasper_'+letter+'_Joint'):robot.remove(child)
relocate_meshes(robot,body,'tron2')
for side,letter in [('left','L'),('right','R')]:
    handpath=ROOT/'teleop/wuji_teleop/models/hand2_beta/urdf'/(side+'.urdf')
    snapshot(handpath,'sources/wuji_'+side+'_original.urdf')
    hand=ET.parse(handpath).getroot();relocate_meshes(hand,handpath,'wuji_'+side)
    for child in hand:robot.append(copy.deepcopy(child))
    calpath=ROOT/config['sides'][side]['calibration']
    snapshot(calpath,'calibration/'+side+'.json')
    config['sides'][side]['calibration']=str((OUT/'calibration'/ (side+'.json')).relative_to(ROOT))
    cal=json.loads(calpath.read_text())
    board=np.array(cal['T_eef_board']);center=np.eye(4);center[:3,3]=config['board_center_in_board_m']
    e_c=board@center;c_h=np.eye(4);c_h[:3,:3]=config['sides'][side]['R_board_center_hand']
    w_e=np.eye(4);w_e[:3,3]=[-.0338,0,-.2125]
    add_frame(robot,'wrist_roll_'+letter+'_Link','sdk_eef_'+letter,w_e)
    add_frame(robot,'sdk_eef_'+letter,'board_center_'+letter,e_c)
    # The hand root already exists; append just its mounting joint.
    mount=ET.SubElement(robot,'joint',name='wuji_'+side+'_mount_fixed',type='fixed')
    ET.SubElement(mount,'parent',link='board_center_'+letter)
    ET.SubElement(mount,'child',link=side[0]+'_base_link')
    ET.SubElement(mount,'origin',**origin(c_h))
    # Display-only adapter bridges the unchanged reference endpoints.
    a=np.array([-.0317,0,-.0812]);b=(w_e@e_c)[:3,3];direction=b-a;length=float(np.linalg.norm(direction))
    z=direction/length;x=np.cross([0,1,0],z);x/=np.linalg.norm(x);y=np.cross(z,x)
    T=np.eye(4);T[:3,:3]=np.column_stack((x,y,z));T[:3,3]=(a+b)/2
    adapter='illustrative_adapter_'+letter
    add_frame(robot,'wrist_roll_'+letter+'_Link',adapter,T)
    link=next(v for v in robot.findall('link') if v.get('name')==adapter)
    visual=ET.SubElement(link,'visual');geo=ET.SubElement(visual,'geometry');ET.SubElement(geo,'cylinder',radius='0.018',length='%.16g'%length)
    material=ET.SubElement(visual,'material',name='illustrative_adapter_material_'+letter);ET.SubElement(material,'color',rgba='0.4 0.5 0.6 1')
    notes[side]=dict(T_wrist_roll_sdk_eef=w_e.tolist(),T_sdk_eef_board_center=e_c.tolist(),T_board_center_hand=c_h.tolist(),adapter_visual_length_m=length)

ET.ElementTree(robot).write(OUT/'robot.urdf',encoding='utf-8',xml_declaration=True)
write_json(OUT/'alignment_config.json',config)
snapshot(ROOT/'replay_robot/align_board_frames.py','tools/align_board_frames.py')
snapshot(ROOT/'replay_robot/test_board_alignment.py','tools/test_board_alignment.py')

# Offline FK verification at neutral and two nonzero arm/head configurations.
import pinocchio as pin
original=pin.buildModelFromUrdf(str(body));assembled=pin.buildModelFromUrdf(str(OUT/'robot.urdf'))
max_error=0.0
for amount in [0.0,0.05,-0.08]:
    qo=pin.neutral(original);qa=pin.neutral(assembled)
    for j in range(1,original.njoints):
        name=original.names[j];oj=original.joints[j];aj=assembled.joints[assembled.getJointId(name)]
        if oj.nq==1:qo[oj.idx_q]=amount;qa[aj.idx_q]=amount
    od=original.createData();ad=assembled.createData()
    pin.framesForwardKinematics(original,od,qo);pin.framesForwardKinematics(assembled,ad,qa)
    for side,letter in [('left','L'),('right','R')]:
        w=od.oMf[original.getFrameId('wrist_roll_'+letter+'_Link')].homogeneous
        e=w@np.array(notes[side]['T_wrist_roll_sdk_eef']);c=e@np.array(notes[side]['T_sdk_eef_board_center']);h=c@np.array(notes[side]['T_board_center_hand'])
        for name,expected in [('sdk_eef_'+letter,e),('board_center_'+letter,c),(side[0]+'_base_link',h)]:
            actual=ad.oMf[assembled.getFrameId(name)].homogeneous
            max_error=max(max_error,float(np.max(np.abs(actual-expected))))
assert max_error<1e-10
for mesh in robot.findall('.//mesh'):assert (OUT/mesh.get('filename')).is_file()
for side,letter in [('left','L'),('right','R')]:
    link=next(v for v in robot.findall('link') if v.get('name')=='illustrative_adapter_'+letter)
    assert link.find('collision') is None

record=dict(version='tron2_wuji_board_v1_20261110',authority='operator-confirmed in chat',
    model_usage='Offline assembly model and frozen conversion reference; not installed into existing motion planners',
    configuration_usage='Existing align_board_frames.py can use this alignment_config.json with --config',
    joints=int(assembled.nq),mesh_files=mesh_count,assembly=notes,
    validation=dict(fk_reference_cases=3,fk_matrix_max_abs_error=max_error,all_mesh_references_resolved=True),
    adapter=dict(operator_approximate_length_m=.05,visual_radius_m=.018,radius_measured=False,collision_included=False),
    source_scene_to_tron_base=None,execution_ready=False,motion_commands_sent=0,source_files=source_files)
write_json(OUT/'model_record.json',record)
write_json(OUT/'SHA256SUMS.json',{str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()})
print(json.dumps({k:record[k] for k in ['version','joints','mesh_files','validation','execution_ready']},ensure_ascii=False))
