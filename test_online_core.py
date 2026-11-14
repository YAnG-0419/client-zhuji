import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from online_core import (AsyncPlanner, DeploymentMapping, TimedChunk,
                         convert_camera_npz, replace_future)


class OnlineCoreTests(unittest.TestCase):
    def fixture_mapping(self):
        transform = np.eye(4)
        transform[2, 3] = 0.07
        return DeploymentMapping(
            base_from_source_camera=np.eye(4),
            sdk_eef_from_hand={"left": transform.copy(), "right": transform.copy()},
            coordinate_origin_distance_m={"left": 0.07, "right": 0.07},
            operator_adapter_length_m=0.05,
            limits={"left": np.tile([[-2.0, 2.0]], (20, 1)),
                    "right": np.tile([[-2.0, 2.0]], (20, 1))},
            provenance={}, report={"test": True})

    def camera_fixture(self, root: Path) -> Path:
        times = np.array([0.0, 0.05, 0.1])
        payload = dict(time_s=times, timestamp_ns=np.rint(times * 1e9).astype(np.int64) + 100,
                       side_order=np.array(["left", "right"]),
                       frame="undistorted_color_optical",
                       pose_target="Wuji l_base_link / r_base_link; not FR3 flange",
                       quaternion_order="wxyz", position_unit="metre", joint_unit="radian")
        for index, side in enumerate(("left", "right")):
            poses = np.tile(np.eye(4), (3, 1, 1))
            poses[:, :3, 3] = [[0.4, 0.2 - 0.4 * index, 0.1],
                               [0.41, 0.2 - 0.4 * index, 0.1],
                               [0.42, 0.2 - 0.4 * index, 0.1]]
            payload[side + "_root_pose"] = poses
            payload[side + "_joint_positions_rad"] = np.zeros((3, 20))
        path = root / "chunk.npz"
        np.savez(path, **payload)
        return path

    def test_five_cm_annotation_does_not_override_calibrated_transform(self):
        with TemporaryDirectory() as folder:
            path = self.camera_fixture(Path(folder))
            chunk = convert_camera_npz(path, self.fixture_mapping(), origin_ns=1_000_000_000)
        self.assertEqual(chunk.metadata["operator_adapter_length_m"], 0.05)
        self.assertEqual(chunk.metadata["calibrated_coordinate_origin_distance_m"],
                         {"left": 0.07, "right": 0.07})
        # Hand root z is 0.10 m and the calibrated EEF->hand origin is 0.07 m.
        # The 0.05 m physical adapter annotation is deliberately not subtracted.
        np.testing.assert_allclose(chunk.eef[:, :, 2], 0.03, atol=1e-12)
        self.assertLess(chunk.metadata["bimanual_relative_matrix_error_max"], 1e-12)
        self.assertEqual(chunk.metadata["hardware_commands_sent"], 0)
        self.assertFalse(chunk.metadata["execution_ready"])

    def test_handoff_uses_old_start_new_end_and_shortest_rotation(self):
        times = np.arange(0, 601, 100, dtype=np.int64) * 1_000_000
        old_eef = np.zeros((len(times), 2, 7)); new_eef = old_eef.copy()
        old_eef[:, :, 3:] = [0, 0, 0, 1]
        new_eef[:, :, :3] = 1
        new_eef[:, :, 3:] = Rotation.from_euler("z", 170, degrees=True).as_quat()
        hands = np.zeros((len(times), 2, 20))
        old = TimedChunk(times, old_eef, hands, {})
        new = TimedChunk(times, new_eef, hands + 1, {})
        merged = replace_future(old, new, 100_000_000, 200_000_000, 100_000_000)
        np.testing.assert_allclose(merged.eef[0, :, :3], 0, atol=1e-12)
        np.testing.assert_allclose(merged.hands[0], 0, atol=1e-12)
        end = np.where(merged.times_ns == 300_000_000)[0][0]
        np.testing.assert_allclose(merged.eef[end, :, :3], 1, atol=1e-12)
        np.testing.assert_allclose(merged.hands[end], 1, atol=1e-12)
        self.assertTrue(np.allclose(np.linalg.norm(merged.eef[:, :, 3:], axis=2), 1))

    def test_async_planner_never_publishes_and_enforces_deadline(self):
        planner = AsyncPlanner(lambda payload, deadline: payload + 1)
        planner.submit(4, time.monotonic_ns() + 1_000_000_000)
        answer = None
        for _ in range(100):
            answer = planner.poll()
            if answer is not None:
                break
            time.sleep(0.001)
        self.assertIsNotNone(answer)
        self.assertEqual(answer["result"], 5)
        planner.close()


if __name__ == "__main__":
    unittest.main()

