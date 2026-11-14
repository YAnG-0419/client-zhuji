import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from live_planner import budget_decay_steps, synchronized_step_times
from live_client import ChunkSampler


def pose(x, angle):
    return np.r_[x, Rotation.from_euler("z", angle).as_quat()]


class LivePlannerTests(unittest.TestCase):
    def test_anchor_decay_respects_budget(self):
        steps, saturated = budget_decay_steps(np.full(20, 0.29), 0.0, 0.0)
        self.assertFalse(saturated)
        self.assertLessEqual(0.29 * 1.5 / steps * 30.0, 0.5 + 1e-9)
        self.assertEqual(budget_decay_steps(np.zeros(20), 0.0, 0.0)[0], 12)
        self.assertTrue(budget_decay_steps(np.full(20, 2.0), 0.0, 0.0)[1])

    def test_time_stretch_only_where_caps_bind(self):
        start_eef = np.stack([pose([0.4, 0.2, 0.0], 0.0), pose([0.4, -0.2, 0.0], 0.0)])
        eef = np.stack([start_eef] * 3)
        eef[2, 0, 0] += 0.05                   # 5 cm in one step -> linear cap binds
        arm = np.zeros((3, 14)); hand = np.zeros((3, 2, 20))
        times, reasons = synchronized_step_times(arm, hand, eef, np.zeros(14), np.zeros((2, 20)),
                                                 start_eef, arm_cap=0.8, hand_cap=0.9,
                                                 linear_cap=0.25, angular_cap=1.0)
        np.testing.assert_allclose(np.diff(np.r_[0, times])[:2], 1 / 30)
        self.assertAlmostEqual(times[2] - times[1], 0.05 / 0.25)
        self.assertEqual(reasons, ["frame", "frame", "eef_linear"])

    def test_sampler_starts_at_command_and_ends_at_plan(self):
        start = np.stack([pose([0.4, 0.2, 0.0], 0.0), pose([0.4, -0.2, 0.0], 0.0)])
        plan = {"times_s": np.array([0.1, 0.2]),
                "eef": np.stack([start, np.stack([pose([0.5, 0.2, 0.0], 0.5), start[1]])]),
                "q20": np.stack([np.zeros((2, 20)), np.ones((2, 20))])}
        sampler = ChunkSampler(start, np.zeros((2, 20)), plan)
        eef, q = sampler(0.0)
        np.testing.assert_allclose(eef, start)
        eef, q = sampler(1.0)
        np.testing.assert_allclose(eef[0, :3], [0.5, 0.2, 0.0])
        np.testing.assert_allclose(q, 1.0)
        eef, _ = sampler(0.15)
        self.assertAlmostEqual(eef[0, 0], 0.45)
        self.assertAlmostEqual(np.linalg.norm(eef[0, 3:]), 1.0)


if __name__ == "__main__":
    unittest.main()
