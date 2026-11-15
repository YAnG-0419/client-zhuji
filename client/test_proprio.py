import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from hand_retarget_wuji2 import Wuji2StreamingRetargeter, pose_error
from online_core import DeploymentMapping
from proprio import (camera_hand_to_sdk_eef, firmware_joint_names,
                     measured_keypoints, ordered_q20)


def pose(xyz, euler):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_euler("xyz", euler).as_matrix()
    value[:3, 3] = xyz
    return value


class ProprioTests(unittest.TestCase):
    def mapping(self):
        camera = pose([0.1, -0.2, 0.7], [0.15, -0.2, 0.3])
        tools = {
            "left": pose([0.01, 0.02, 0.07], [0.1, 0.0, -0.2]),
            "right": pose([-0.01, -0.02, 0.069], [-0.1, 0.0, 0.2]),
        }
        hands = {side: Wuji2StreamingRetargeter(side).hand for side in ("left", "right")}
        return DeploymentMapping(
            camera, tools, {"left": 0.073, "right": 0.069}, 0.05,
            {side: np.c_[hands[side].lo, hands[side].hi] for side in hands},
            {}, {"camera_mapping_mode": "live_zhuji_camera"})

    def test_named_joint_state_reorders_to_finger_major(self):
        names = list(reversed(firmware_joint_names("right")))
        q = np.arange(20, dtype=float) / 10
        values = list(reversed(q))
        np.testing.assert_allclose(ordered_q20("right", names, values), q)
        with self.assertRaisesRegex(ValueError, "missing Wuji"):
            ordered_q20("right", names[:-1], values[:-1])

    def test_measured_fk_and_inverse_tool_chain_are_exact(self):
        mapping = self.mapping()
        base_T_eef = {
            "left": pose([0.45, 0.25, 0.5], [0.2, -0.1, 0.4]),
            "right": pose([0.45, -0.25, 0.5], [-0.2, 0.1, -0.4]),
        }
        q = {}
        expected_roots = {}
        for side in ("left", "right"):
            hand = Wuji2StreamingRetargeter(side).hand
            q[side] = np.clip(np.linspace(-0.15, 0.3, 20),
                              hand.lo + 0.05, hand.hi - 0.05)
            expected_roots[side] = (np.linalg.inv(mapping.base_from_source_camera)
                                    @ base_T_eef[side]
                                    @ mapping.sdk_eef_from_hand[side])
        points = measured_keypoints(mapping, base_T_eef, q)
        for index, side in enumerate(("left", "right")):
            solver = Wuji2StreamingRetargeter(side, "free_root", thresholds={
                "max_joint_step_rad": 1.0,
                "max_root_translation_step_m": 1.0,
                "max_root_rotation_step_rad": 2.0,
            })
            solver.begin_chunk(q[side], expected_roots[side])
            answer = solver.retarget_frame(points[index], 1, normalize=False)
            self.assertTrue(answer["accepted"], answer["diagnostic"])
            translation, angle = pose_error(answer["cam_T_root"], expected_roots[side])
            self.assertLess(translation, 1e-9)
            self.assertLess(angle, 1e-9)
            recovered = camera_hand_to_sdk_eef(mapping, side, answer["cam_T_root"])
            np.testing.assert_allclose(recovered, base_T_eef[side], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
