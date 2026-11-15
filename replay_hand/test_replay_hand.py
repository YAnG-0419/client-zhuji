import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from anchor_cross import MOUNT_CENTER_EEF_M, reanchor_eef
from episode_cross import SCHEMA, hand_names, load_cross_episode


class CrossEpisodeTests(unittest.TestCase):
    def make_episode(self, folder):
        root = Path(folder)
        (root.parent / "record/config/model").mkdir(parents=True, exist_ok=True)
        (root.parent / "record/config/model/robot.urdf").write_text("test")
        t = np.array([0.0, 0.02, 0.04])
        eef = np.zeros((3, 2, 7)); eef[:, :, 6] = 1.0
        hands = np.zeros((3, 2, 20))
        path = root / "episode.npz"
        np.savez_compressed(
            path, timestamp_ns=np.array([1, 2, 3]), t=t, eef=eef, hands=hands,
            left_hand_joint_names=np.asarray(hand_names("left")),
            right_hand_joint_names=np.asarray(hand_names("right")),
            metadata_json=np.asarray(json.dumps({"schema": SCHEMA, "frame": "tron2_base_link"})),
        )
        return path

    def test_joint_names(self):
        self.assertEqual(len(hand_names("left")), 20)
        self.assertEqual(hand_names("right")[0], "right_finger1_joint1")

    def test_load_shapes_and_last_frame(self):
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "replay_hand"
            package.mkdir()
            path = self.make_episode(package)
            # Mimic the deployed layout expected by episode_cross.__file__.
            import episode_cross
            original = episode_cross.__file__
            episode_cross.__file__ = str(package / "episode_cross.py")
            try:
                t, eef, hands, meta = load_cross_episode(path, 50.0)
            finally:
                episode_cross.__file__ = original
            self.assertEqual(eef.shape, (3, 2, 7))
            self.assertEqual(hands.shape, (3, 2, 20))
            self.assertAlmostEqual(t[-1], 0.04)
            self.assertEqual(meta["fk"]["quaternion_order"], "xyzw")

    def test_reject_nonunit_quaternion(self):
        with TemporaryDirectory() as temporary:
            package = Path(temporary) / "replay_hand"
            package.mkdir()
            path = self.make_episode(package)
            with np.load(path, allow_pickle=False) as data:
                values = {name: data[name].copy() for name in data.files}
            values["eef"][:, :, 3:] = 0.0
            np.savez_compressed(path, **values)
            import episode_cross
            original = episode_cross.__file__
            episode_cross.__file__ = str(package / "episode_cross.py")
            try:
                with self.assertRaisesRegex(ValueError, "quaternion"):
                    load_cross_episode(path, 50.0)
            finally:
                episode_cross.__file__ = original

    def test_reanchor_preserves_relative_mount_motion(self):
        eef = np.zeros((3, 2, 7))
        eef[:, :, 6] = 1.0
        eef[1:, 0, :3] = [[0.02, -0.01, 0.03], [0.04, -0.02, 0.01]]
        eef[1:, 1, :3] = [[-0.01, 0.02, 0.01], [-0.03, 0.04, 0.02]]
        eef[:, 0, 3:] = Rotation.from_euler("z", [0.0, 0.1, 0.2]).as_quat()
        eef[:, 1, 3:] = Rotation.from_euler("x", [0.0, -0.1, -0.2]).as_quat()
        anchor = np.array([
            [0.6, 0.3, 0.0, *Rotation.from_euler("y", 0.5).as_quat()],
            [0.6, -0.3, 0.0, *Rotation.from_euler("y", -0.5).as_quat()],
        ])
        target, report = reanchor_eef(eef, anchor)
        self.assertTrue(np.allclose(target[0, :, :3], anchor[:, :3], atol=1e-10, rtol=0))
        self.assertLess(np.max((
            Rotation.from_quat(target[0, :, 3:]).inv() * Rotation.from_quat(anchor[:, 3:])
        ).magnitude()), 1e-10)
        for side in range(2):
            source_r = Rotation.from_quat(eef[:, side, 3:])
            target_r = Rotation.from_quat(target[:, side, 3:])
            source_mount = eef[:, side, :3] + source_r.apply(MOUNT_CENTER_EEF_M[side])
            target_mount = target[:, side, :3] + target_r.apply(MOUNT_CENTER_EEF_M[side])
            self.assertTrue(np.allclose(
                target_mount - target_mount[0], source_mount - source_mount[0], atol=1e-10, rtol=0
            ))
            source_delta = source_r * source_r[0].inv()
            target_delta = target_r * target_r[0].inv()
            self.assertLess(np.max((source_delta.inv() * target_delta).magnitude()), 1e-10)
        self.assertTrue(report["relative_motion_preserved"])


if __name__ == "__main__":
    unittest.main()
