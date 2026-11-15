import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from execution_postprocess import (
    apply_local_anchor, check_stationary_since_sample, clamp_q_report,
    q_speed_report, select_root_mode, smooth_q_trajectory,
    smooth_root_trajectory, trajectory_jitter)


def poses(length=20):
    output = np.repeat(np.eye(4)[None], length, axis=0)
    output[:, 0, 3] = np.linspace(0.0, 0.1, length)
    output[:, :3, :3] = Rotation.from_euler(
        "z", np.linspace(0.0, 0.4, length)).as_matrix()
    return output


class ExecutionPostprocessTests(unittest.TestCase):
    def test_decay_anchor_is_exact_at_zero_and_gone_at_decay_steps(self):
        fitted = poses()
        measured = fitted[0].copy()
        measured[:3, 3] += [0.01, -0.02, 0.03]
        measured[:3, :3] = measured[:3, :3] @ Rotation.from_rotvec(
            [0.0, 0.0, math.pi - 1e-8]).as_matrix()
        q = np.tile(np.linspace(-0.2, 0.2, 20), (20, 1))
        measured_q = q[0] + 0.1
        limits = np.tile([-1.0, 1.0], (20, 1))
        roots, anchored_q, report = apply_local_anchor(
            fitted, q, measured, measured_q, limits, mode="decay", decay_steps=12)
        np.testing.assert_allclose(roots[0], measured, atol=1e-8, rtol=0)
        np.testing.assert_allclose(anchored_q[0], measured_q, atol=1e-12, rtol=0)
        np.testing.assert_allclose(roots[12], fitted[12], atol=1e-12, rtol=0)
        np.testing.assert_allclose(anchored_q[12], q[12], atol=1e-12, rtol=0)
        self.assertAlmostEqual(report["local_offset_rotation_rad"], math.pi, places=7)

    def test_constant_and_none_modes(self):
        fitted = poses(5); measured = fitted[0].copy(); measured[1, 3] = 0.02
        q = np.zeros((5, 2)); measured_q = np.array([0.1, -0.1])
        limits = np.tile([-1.0, 1.0], (2, 1))
        constant, cq, _ = apply_local_anchor(
            fitted, q, measured, measured_q, limits, mode="constant")
        none, nq, _ = apply_local_anchor(
            fitted, q, measured, measured_q, limits, mode="none")
        np.testing.assert_allclose(constant[0], measured, atol=1e-12)
        np.testing.assert_allclose(cq, np.tile(measured_q, (5, 1)))
        np.testing.assert_allclose(none, fitted)
        np.testing.assert_allclose(nq, q)

    def test_smoothing_reduces_second_difference(self):
        roots = poses(21)
        roots[1::2, 1, 3] += 0.002
        q = np.zeros((21, 3)); q[1::2] = 0.03
        before = trajectory_jitter(roots, q)
        smooth_roots = smooth_root_trajectory(roots, 5)
        smooth_q = smooth_q_trajectory(q, 5)
        after = trajectory_jitter(smooth_roots, smooth_q)
        self.assertLess(after["root_translation_second_difference_m"]["mean"],
                        before["root_translation_second_difference_m"]["mean"])
        self.assertLess(after["q_second_difference_rad"]["mean"],
                        before["q_second_difference_rad"]["mean"])

    def test_speed_gate_includes_measured_to_step_zero_and_clamp_is_diagnostic(self):
        q = np.array([[0.2], [0.21], [0.22]])
        times = np.array([33_333_333, 66_666_667, 100_000_000])
        report = q_speed_report(q, np.array([0.0]), times, 0)
        self.assertGreater(report["peak_rad_s"], 5.9)
        self.assertEqual(report["peak_step"], 0)
        projected, clamp = clamp_q_report(q, np.array([0.0]), times, 0)
        self.assertLessEqual(projected[0, 0], 1.0 / 30.0 + 1e-8)
        self.assertTrue(clamp["diagnostic_only"])
        self.assertFalse(clamp["execution_ready"])

    def test_stationary_thresholds_and_auto_mode(self):
        before = np.repeat(np.eye(4)[None], 2, axis=0)
        after = before.copy(); after[0, 0, 3] = 0.003
        report = check_stationary_since_sample(before, after,
                                               np.zeros((2, 20)), np.zeros((2, 20)))
        self.assertFalse(report["stationary"])
        self.assertEqual(select_root_mode("auto", "humanhand")[0], "free_root")
        self.assertEqual(select_root_mode("auto", "realrobot")[0],
                         "kabsch_fixed_root")


if __name__ == "__main__":
    unittest.main()
