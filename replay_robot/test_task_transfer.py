"""Coordinate-direction regression checks for the task-transfer chain."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from prepare_task_replay import transfer, fixed_scene_base


def pose(xyz,angles):
    t=np.eye(4);t[:3,3]=xyz;t[:3,:3]=Rotation.from_euler('xyz',angles).as_matrix();return t


def test_nominal_scene_mount_cancels():
    actual=pose([.4,.2,.1],[.2,.5,-.1]);camera=pose([.1,.2,.3],[-.2,.1,.7])
    mapping=pose([-.1,.04,.3],[.1,-.3,.2]);target=pose([.2,-.2,.1],[.7,.3,-.2])
    for nominal in [np.eye(4),pose([3,-2,1],[1,.2,-2])]:
        result,_=transfer(nominal@actual,nominal,camera,mapping,target)
        np.testing.assert_allclose(result,target@mapping@camera@actual,atol=1e-14)


def test_shared_camera_transform_preserves_relative_hand_pose():
    left=pose([.4,.1,.2],[.3,.1,.2]);right=pose([.4,-.1,.2],[-.3,-.1,.2])
    M=pose([.03,-.01,.2],[.1,.2,.3]);G=pose([.2,.1,.3],[-.2,.4,.1])
    a,_=transfer(left,np.eye(4),np.eye(4),M,G)
    b,_=transfer(right,np.eye(4),np.eye(4),M,G)
    np.testing.assert_allclose(np.linalg.inv(a)@b,np.linalg.inv(left)@right,atol=1e-14)


def test_camera_mapping_is_left_multiplied():
    hand=pose([1,0,0],[.2,.3,0]);M=pose([0,2,0],[0,0,np.pi/2])
    out,_=transfer(hand,np.eye(4),np.eye(4),M,np.eye(4))
    np.testing.assert_allclose(out[:3,3],[0,3,0],atol=1e-14)


def test_moving_scene_base_rejected(tmp_path):
    p=tmp_path/'source.urdf'
    p.write_text('<robot><joint type="revolute"><parent link="lychee_root"/><child link="left_fr3v2_link0"/><origin xyz="0 0 0"/></joint></robot>')
    with pytest.raises(ValueError,match='Moving joint'):fixed_scene_base(p,'left')
