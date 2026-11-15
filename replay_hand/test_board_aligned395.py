import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from build_board_aligned395 import wrist_to_base, resample_pose, tool_frames


class BoardAlignmentTests(unittest.TestCase):
    def test_old_wrist_root_recovers_absolute_model_points(self):
        rng = np.random.default_rng(14)
        action = rng.normal(size=(8, 26))
        wrist = np.array([.0003, -.003, .0285])
        points = rng.normal(size=(21, 3))
        base = wrist_to_base(action, wrist)
        for k in range(8):
            r = Rotation.from_rotvec(action[k, 3:6]).as_matrix()
            expected = (points-wrist) @ r.T+action[k, :3]
            actual = points @ base[k, :3, :3].T+base[k, :3, 3]
            np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_right_axis_and_tool_inverse(self):
        axes = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        np.testing.assert_array_equal(axes @ [1,0,0], [0,1,0])
        np.testing.assert_array_equal(axes @ [0,1,0], [-1,0,0])
        cal = np.eye(4); cal[:3,:3] = Rotation.from_euler('xyz',[.3,.2,-.5]).as_matrix()
        cal[:3,3] = [.01,.02,.03]
        e_c, c_h = tool_frames({'T_eef_board':cal.tolist()}, axes, [.14,.09,0])
        source = np.eye(4); source[:3,3] = [.6,-.4,.2]
        eef = source @ np.linalg.inv(c_h) @ np.linalg.inv(e_c)
        np.testing.assert_allclose(eef @ e_c @ c_h, source, atol=1e-14)

    def test_common_transform_preserves_bimanual_relative_pose(self):
        rng = np.random.default_rng(9)
        ts = np.tile(np.eye(4),(3,1,1))
        ts[:,:3,:3] = Rotation.random(3,random_state=rng).as_matrix()
        ts[:,:3,3] = rng.normal(size=(3,3))
        left,right,placement = ts
        np.testing.assert_allclose(np.linalg.inv(placement @ left) @ (placement @ right),
                                   np.linalg.inv(left) @ right, atol=1e-14)

    def test_interpolation_uses_relative_integer_timestamps(self):
        start = 1794000000000000000
        stamps = np.array([start,start+20000000],dtype=np.int64)
        poses = np.tile(np.eye(4),(2,1,1));poses[1,0,3]=.02
        result = resample_pose(stamps,poses,np.array([start,start+10000000,start+20000000]))
        np.testing.assert_allclose(result[:,0,3],[0,.01,.02],atol=1e-14)
        with self.assertRaises(ValueError):
            resample_pose(stamps,poses,np.array([start-1,start+20000000]))


if __name__ == '__main__':
    unittest.main()
