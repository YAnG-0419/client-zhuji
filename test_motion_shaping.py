import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from motion_shaping import (build_splice, interpolate_eef, joint_rates,
                            pose_rates, shape_timeline, speed_profile)


CAPS = {"arm_cap": 0.8, "hand_cap": 0.9, "linear_cap": 0.25, "angular_cap": 1.0,
        "arm_accel_cap": 2.0, "hand_accel_cap": 2.5,
        "linear_accel_cap": 0.5, "angular_accel_cap": 2.0}


def ramp_streams(steps=32, hand_step=0.03, eef_step=0.004):
    """Constant-rate motion: every step moves the same amount."""
    k = np.arange(steps + 1)
    q20 = np.zeros((steps + 1, 2, 20)); q20[:, :, 0] = k[:, None] * hand_step
    q16 = np.zeros((steps + 1, 16)); q16[:, 0] = k * 0.01
    eef = np.zeros((steps + 1, 2, 7)); eef[:, :, 3] = 1.0
    eef[:, :, 0] = 0.4 + k[:, None] * eef_step
    return q16, q20, eef


class ShapeTimelineTests(unittest.TestCase):
    def test_easing_starts_and_ends_slowly_and_keeps_waypoints(self):
        q16, q20, eef = ramp_streams()
        base = np.full(32, 1 / 30)
        shaped = shape_timeline(base, arm_q=q16[:, :14], hand_q=q20, eef=eef, caps=CAPS)
        times = np.r_[0.0, shaped.times]
        dt = np.diff(times)
        # Same number of waypoints; only the spacing changed.
        self.assertEqual(len(shaped.times), 32)
        self.assertGreater(dt[0], dt[len(dt) // 2] * 1.5)
        self.assertGreater(dt[-1], dt[len(dt) // 2] * 1.5)
        self.assertGreater(shaped.duration_s, shaped.nominal_duration_s)

    def test_slowdown_enforces_velocity_and_acceleration_caps(self):
        q16, q20, eef = ramp_streams(hand_step=0.08)   # 2.4 rad/s at 30 Hz
        base = np.full(32, 1 / 30)
        shaped = shape_timeline(base, arm_q=q16[:, :14], hand_q=q20, eef=eef, caps=CAPS)
        self.assertGreater(shaped.slowdown, 2.0)
        self.assertLessEqual(shaped.peaks["hand_rad_s"], CAPS["hand_cap"] * 1.001)
        self.assertLessEqual(shaped.peaks["hand_rad_s2"], CAPS["hand_accel_cap"] * 1.001)
        self.assertLessEqual(shaped.peaks["linear_m_s"], CAPS["linear_cap"] * 1.001)

    def test_disabling_easing_reproduces_the_plain_timeline(self):
        q16, q20, eef = ramp_streams(hand_step=0.005, eef_step=0.0005)
        base = np.full(32, 1 / 30)
        shaped = shape_timeline(base, arm_q=q16[:, :14], hand_q=q20, eef=eef, caps=CAPS,
                                ease_in=False, ease_out=False)
        self.assertEqual(shaped.slowdown, 1.0)
        np.testing.assert_allclose(shaped.times, np.cumsum(base))

    def test_impossible_chunk_is_refused(self):
        q16, q20, eef = ramp_streams(hand_step=1.0)
        with self.assertRaises(ValueError):
            shape_timeline(np.full(32, 1 / 30), arm_q=q16[:, :14], hand_q=q20, eef=eef,
                           caps=CAPS, max_slowdown=4.0)


class SpliceTests(unittest.TestCase):
    def plan(self, steps=32):
        q16, q20, eef = ramp_streams(steps)
        model = np.arange(1, steps + 1) / 30.0
        return model, eef[1:], q20[1:], q16[1:]

    def test_skips_covered_points_and_matches_current_state(self):
        model, eef, q20, q16 = self.plan()
        current_eef = np.zeros((2, 7)); current_eef[:, 3] = 1.0
        current_eef[:, 0] = 0.4 + 0.004 * 16 + 0.002      # 2 mm off the plan
        current_q20 = np.zeros((2, 20)); current_q20[:, 0] = 0.03 * 16 + 0.01
        velocity = np.zeros((2, 3)); velocity[:, 0] = 0.004 * 30     # following the plan
        splice = build_splice(progress_model_s=16 / 30.0, plan_model_t=model, plan_eef=eef,
                              plan_q20=q20, plan_q16=q16, current_eef=current_eef,
                              current_q20=current_q20, current_eef_velocity=velocity,
                              caps=CAPS, moving=True)
        self.assertIsNotNone(splice)
        self.assertEqual(splice.skipped_points, 16)
        # No jump at the splice: the first commanded point is one capped step
        # away from where the robot actually is.
        step = float(splice.times[0])
        self.assertLessEqual(abs(splice.eef[0, 0, 0] - current_eef[0, 0]),
                             CAPS["linear_cap"] * step)
        self.assertLessEqual(abs(splice.q20[0, 0, 0] - current_q20[0, 0]),
                             CAPS["hand_cap"] * step)
        # ... and ends on the plan.
        np.testing.assert_allclose(splice.eef[-1, :, :3], eef[-1, :, :3], atol=1e-9)
        np.testing.assert_allclose(splice.q20[-1], q20[-1], atol=1e-9)

    def test_from_rest_eases_in_instead_of_matching_velocity(self):
        model, eef, q20, q16 = self.plan()
        current_eef = np.zeros((2, 7)); current_eef[:, 3] = 1.0
        current_eef[:, 0] = 0.4 + 0.004 * 16
        current_q20 = np.zeros((2, 20)); current_q20[:, 0] = 0.03 * 16
        splice = build_splice(progress_model_s=16 / 30.0, plan_model_t=model, plan_eef=eef,
                              plan_q20=q20, plan_q16=q16, current_eef=current_eef,
                              current_q20=current_q20,
                              current_eef_velocity=np.zeros((2, 3)), caps=CAPS, moving=False)
        self.assertIsNotNone(splice)
        step = np.diff(np.r_[0.0, splice.times])
        self.assertGreater(step[0], step[len(step) // 2] * 1.5)   # eased in
        self.assertLessEqual(splice.peaks["hand_rad_s2"], CAPS["hand_accel_cap"] * 1.05)

    def test_returns_none_when_nothing_useful_is_left(self):
        model, eef, q20, q16 = self.plan()
        current_eef = np.zeros((2, 7)); current_eef[:, 3] = 1.0
        self.assertIsNone(build_splice(
            progress_model_s=model[-1], plan_model_t=model, plan_eef=eef, plan_q20=q20,
            plan_q16=q16, current_eef=current_eef, current_q20=np.zeros((2, 20)),
            current_eef_velocity=np.zeros((2, 3)), caps=CAPS, moving=True))

    def test_rejects_a_splice_that_would_break_the_caps(self):
        model, eef, q20, q16 = self.plan()
        current_eef = np.zeros((2, 7)); current_eef[:, 3] = 1.0
        current_eef[:, 0] = 0.4 + 0.004 * 16 + 0.5        # 50 cm away: impossible
        self.assertIsNone(build_splice(
            progress_model_s=16 / 30.0, plan_model_t=model, plan_eef=eef, plan_q20=q20,
            plan_q16=q16, current_eef=current_eef, current_q20=np.zeros((2, 20)),
            current_eef_velocity=np.zeros((2, 3)), caps=CAPS, moving=True))


class HelperTests(unittest.TestCase):
    def test_speed_profile_bounds(self):
        u = np.linspace(0, 1.0, 50)
        profile = speed_profile(u, 1.0, 0.25, 0.2)
        self.assertAlmostEqual(profile[0], 0.2, places=6)
        self.assertAlmostEqual(profile[-1], 0.2, places=6)
        self.assertAlmostEqual(profile.max(), 1.0, places=6)

    def test_interpolate_eef_is_unit_and_endpoint_exact(self):
        eef = np.zeros((2, 2, 7))
        eef[0, :, 3:] = Rotation.from_euler("z", 0.0).as_quat()
        eef[1, :, 3:] = Rotation.from_euler("z", 1.0).as_quat()
        eef[1, :, 0] = 1.0
        out = interpolate_eef([0.0, 1.0], eef, [0.0, 0.5, 1.0])
        np.testing.assert_allclose(np.linalg.norm(out[:, :, 3:], axis=2), 1.0)
        np.testing.assert_allclose(out[0], eef[0], atol=1e-12)
        self.assertAlmostEqual(out[1, 0, 0], 0.5)

    def test_rates_see_the_rest_edges(self):
        times = np.array([0.0, 0.1, 0.2])
        values = np.array([[0.0], [0.1], [0.2]])
        _, without = joint_rates(times, values, rest_start=False, rest_end=False)
        _, with_edges = joint_rates(times, values, rest_start=True, rest_end=True)
        self.assertEqual(without, 0.0)          # constant speed in the middle
        self.assertGreater(with_edges, 0.0)     # but it starts and stops

    def test_pose_rates_measure_rotation(self):
        eef = np.zeros((3, 1, 7))
        for i in range(3):
            eef[i, 0, 3:] = Rotation.from_euler("z", 0.1 * i).as_quat()
        (_, _), (angular, _) = pose_rates([0.0, 0.1, 0.2], eef, rest_start=False, rest_end=False)
        self.assertAlmostEqual(angular, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
