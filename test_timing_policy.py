import unittest

from timing_policy import (DEFAULT_MODE, DEFAULT_STEPS, FRAME_PER_STEP_30HZ,
                           NATIVE_TIME, ModelTiming, build_schedule)


def humanhand_model():
    return ModelTiming.from_info({
        "contract": "humanhand",
        "fps": 30.0,
        "history_offset_frames": 4,
        "keypoint_time_s": [index / 120.0 for index in range(1, 33)],
    })


class TimingPolicyTests(unittest.TestCase):
    def test_default_executes_32_steps_as_32_frames_at_30hz(self):
        model = humanhand_model()
        schedule = build_schedule(model, sample_ns=1_000_000_000,
                                  exec_start_ns=2_000_000_000)
        self.assertEqual(DEFAULT_MODE, FRAME_PER_STEP_30HZ)
        self.assertEqual(DEFAULT_STEPS, 32)
        self.assertEqual(schedule.mode, FRAME_PER_STEP_30HZ)
        self.assertEqual(len(schedule.deadlines_ns), 32)
        self.assertEqual(schedule.deadlines_ns[0], 2_033_333_333)
        self.assertEqual(schedule.deadlines_ns[-1], 3_066_666_667)
        self.assertAlmostEqual(schedule.duration_s, 32 / 30.0, places=9)
        periods = {right - left for left, right in
                   zip(schedule.deadlines_ns, schedule.deadlines_ns[1:])}
        self.assertTrue(periods.issubset({33_333_333, 33_333_334}))
        self.assertFalse(schedule.report["execution_ready"])
        self.assertEqual(schedule.report["hardware_commands_sent"], 0)

    def test_history_capture_contract_stays_at_four_camera_frames(self):
        model = humanhand_model()
        report = model.history_report()
        self.assertEqual(report["capture_offset_frames"], 4)
        self.assertAlmostEqual(report["capture_offset_s"], 4 / 30.0)
        self.assertAlmostEqual(report["frame_per_step_equivalent_execution_periods"], 16.0)
        self.assertAlmostEqual(report["frame_per_step_equivalent_span_s"], 16 / 30.0)
        self.assertTrue(report["equivalent_value_is_diagnostic_only"])

    def test_native_time_is_retained_only_as_explicit_comparison(self):
        model = humanhand_model()
        schedule = build_schedule(model, sample_ns=5_000_000_000, mode=NATIVE_TIME)
        self.assertEqual(schedule.deadlines_ns[0], 5_008_333_333)
        self.assertEqual(schedule.deadlines_ns[-1], 5_266_666_667)
        self.assertAlmostEqual(schedule.duration_s, 8 / 30.0, places=9)
        self.assertEqual(schedule.report["mode"], NATIVE_TIME)
        self.assertTrue(schedule.report["native_time_is_diagnostic_only"])
        self.assertFalse(schedule.report["selected_mode_is_execution_default"])

    def test_future_selection_discards_only_expired_prefix(self):
        schedule = build_schedule(humanhand_model(), sample_ns=1_000_000_000,
                                  exec_start_ns=2_000_000_000)
        selected = schedule.future(2_100_000_000, reserve_s=0.05)
        self.assertEqual(selected.source_indices[0], 4)
        self.assertEqual(selected.report["future_selection"]["discarded_steps"], 4)
        with self.assertRaises(TimeoutError):
            schedule.future(4_000_000_000)

    def test_rejects_ambiguous_or_invalid_timing(self):
        model = humanhand_model()
        with self.assertRaisesRegex(ValueError, "exec_start"):
            build_schedule(model, sample_ns=100)
        with self.assertRaisesRegex(ValueError, "unknown timing"):
            build_schedule(model, sample_ns=100, exec_start_ns=100, mode="slow025")
        with self.assertRaisesRegex(ValueError, "steps"):
            build_schedule(model, sample_ns=100, exec_start_ns=100, steps=0)
        with self.assertRaisesRegex(ValueError, "anchored"):
            build_schedule(model, sample_ns=100, exec_start_ns=101, mode=NATIVE_TIME)


if __name__ == "__main__":
    unittest.main()
