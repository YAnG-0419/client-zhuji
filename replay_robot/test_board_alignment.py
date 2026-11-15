"""Offline tests for transform direction, axis signs and shifted board origin."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from align_board_frames import tool_frames, convert_poses


def test_operator_right_axes_and_inverse_pose_direction():
    r = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    assert np.array_equal(r @ [1,0,0], [0,1,0])
    assert np.array_equal(r @ [0,1,0], [-1,0,0])
    assert np.array_equal(r @ [0,0,1], [0,0,1])
    ec,ch = tool_frames({'T_eef_board':np.eye(4)},r,[0,0,0])
    sc,_ = convert_poses(np.eye(4)[None],ec,ch)
    # Identity hand pose -> centered board X is -world Y, not +world Y.
    assert np.allclose(sc[0,:3,0], [0,-1,0])
    assert np.allclose(sc[0,:3,1], [1,0,0])


def test_center_is_rotated_board_point_not_raw_board_translation():
    eb=np.eye(4);eb[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix();eb[:3,3]=[.1,.2,.3]
    ec,_=tool_frames({'T_eef_board':eb},np.eye(3),[.14,.09,0])
    assert np.allclose(ec[:3,3],[.01,.34,.3])


@pytest.mark.parametrize('angle',[0,90])
def test_arbitrary_world_pose_and_tool_translation_roundtrip(angle):
    eb=np.eye(4);eb[:3,:3]=Rotation.from_euler('xyz',[.2,-.5,1.1]).as_matrix();eb[:3,3]=[.08,.13,.07]
    r=Rotation.from_euler('z',angle,degrees=True).as_matrix()
    ec,ch=tool_frames({'T_eef_board':eb},r,[.14,.09,0])
    sh=np.tile(np.eye(4),(3,1,1));sh[:,:3,:3]=Rotation.from_euler('xyz',[[.4,.2,.1],[.5,-.3,.7],[1.2,.4,-.8]]).as_matrix();sh[:,:3,3]=[[.4,.2,.1],[.5,.3,.2],[.6,.4,.3]]
    sc,se=convert_poses(sh,ec,ch)
    assert np.allclose(sc[:,:3,3],sh[:,:3,3])
    assert np.allclose(se@ec,sc)
    assert np.allclose(se@ec@ch,sh)
    if angle==0:assert np.allclose(sc,sh)


def test_reflection_rejected():
    with pytest.raises(ValueError,match='reflected'):
        tool_frames({'T_eef_board':np.eye(4)},np.diag([1,1,-1]),[0,0,0])
