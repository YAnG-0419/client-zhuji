import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from online_dryrun import run
from timing_policy import FRAME_PER_STEP_30HZ


class OnlineDryrunIntegrationTests(unittest.TestCase):
    def test_recorded_state_fake_gt_full_32_step_chain(self):
        client = Path(__file__).resolve().parent
        root = client.parent
        args = SimpleNamespace(
            root=root,
            config=client / "deployment_config.json",
            source=client / "testdata/episode023_i800_state.npz",
            server="fake-gt", root_mode="both", image_mode="virtual_cam0",
            time_scale=FRAME_PER_STEP_30HZ, steps=32, task="flower")
        result = run(args)
        self.assertEqual(set(result["outputs"]), {"free_root", "kabsch_fixed_root"})
        self.assertAlmostEqual(result["schedule"].duration_s, 32 / 30.0, places=9)
        for mode, report in result["reports"].items():
            with self.subTest(mode=mode):
                self.assertTrue(report["completed"])
                self.assertEqual(report["rejected_frames"], [])
                self.assertEqual(report["eef_box_violations"], 0)
                self.assertLess(report["fake_gt"]["root_translation_m"]["max"], 0.001)
                self.assertLess(report["fake_gt"]["root_rotation_rad"]["max"],
                                0.5 * 3.141592653589793 / 180)
                self.assertLess(report["fake_gt"]["sdk_eef_translation_m"]["max"], 0.001)
                self.assertLess(report["fake_gt"]["sdk_eef_rotation_rad"]["max"],
                                0.5 * 3.141592653589793 / 180)
                self.assertTrue(report["offline_pinocchio_ik"]["all_reachable"])
                self.assertTrue(report["offline_pinocchio_ik"]["arm_joint_speed_within_1rad_s"])


if __name__ == "__main__":
    unittest.main()

