import unittest

import numpy as np

from observation_buffer import ObservationBuffer, STATE_STREAMS


class ObservationBufferTests(unittest.TestCase):
    def ready(self):
        buffer = ObservationBuffer()
        base_header = 1_800_000_000_000_000_000  # Epoch, deliberately not monotonic.
        base_receive = 50_000_000_000
        raw = bytes(640 * 480 * 3)
        buffer.set_execution_active(True)
        for index in range(20):
            buffer.add_image(base_header + index * 33_333_333,
                             base_receive + index * 33_333_333, raw)
        current = base_header + 19 * 33_333_333
        for stream in STATE_STREAMS:
            buffer.add_state(stream, current - 3_000_000, [1.0, 2.0], ["a", "b"])
        buffer.set_camera_info({"test": True})
        return buffer, base_receive + 19 * 33_333_333

    def test_snapshot_uses_epoch_header_for_alignment_and_monotonic_for_age(self):
        buffer, receive = self.ready()
        result = buffer.snapshot(now_monotonic_ns=receive + 20_000_000)
        self.assertEqual(result["head_rgb"].shape, (480, 640, 3))
        self.assertEqual(result["sample_ns"] - result["history_sample_ns"],
                         16 * 33_333_333)
        self.assertEqual(result["history_clock"], "execution")
        self.assertAlmostEqual(result["measured_camera_fps"], 30.0, places=4)
        self.assertEqual(set(result["states"]), set(STATE_STREAMS))
        self.assertFalse(result["head_rgb"].flags.writeable)
        self.assertFalse(result["states"]["left_eef"].flags.writeable)

    def test_missing_wuji_and_missing_history_fail_closed(self):
        buffer, receive = self.ready()
        buffer.states["right_wuji"].clear()
        with self.assertRaisesRegex(RuntimeError, "no samples"):
            buffer.snapshot(now_monotonic_ns=receive)
        fresh = ObservationBuffer(); raw = bytes(640 * 480 * 3)
        fresh.add_image(100_000_000_000, 10_000_000_000, raw)
        fresh.set_camera_info({})
        with self.assertRaisesRegex(RuntimeError, "history"):
            fresh.snapshot(now_monotonic_ns=10_000_000_001)

    def test_execution_history_skips_hold_intervals(self):
        buffer = ObservationBuffer(); raw = bytes(640 * 480 * 3)
        base_header = 1_000_000_000_000
        base_receive = 5_000_000_000
        buffer.set_execution_active(True)
        for index in range(18):
            buffer.add_image(base_header + index * 33_333_333,
                             base_receive + index * 33_333_333, raw)
        buffer.set_execution_active(False)
        for index in range(18, 48):
            buffer.add_image(base_header + index * 33_333_333,
                             base_receive + index * 33_333_333, raw)
        buffer.set_execution_active(True)
        for index in range(48, 51):
            buffer.add_image(base_header + index * 33_333_333,
                             base_receive + index * 33_333_333, raw)
        current = base_header + 50 * 33_333_333
        for stream in STATE_STREAMS:
            buffer.add_state(stream, current, [0.0], ["a"])
        buffer.set_camera_info({})
        result = buffer.snapshot(now_monotonic_ns=base_receive + 50 * 33_333_333)
        self.assertLess(result["history_sample_ns"], base_header + 18 * 33_333_333)
        self.assertEqual(result["history_clock"], "execution")

    def test_shadow_can_omit_execution_history_with_reason(self):
        buffer = ObservationBuffer(); raw = bytes(640 * 480 * 3)
        for index in range(3):
            buffer.add_image(1_000_000_000 + index * 33_333_333,
                             2_000_000_000 + index * 33_333_333, raw)
        current = 1_000_000_000 + 2 * 33_333_333
        for stream in STATE_STREAMS:
            buffer.add_state(stream, current, [0.0], ["a"])
        buffer.set_camera_info({})
        result = buffer.snapshot(now_monotonic_ns=2_000_000_000 + 2 * 33_333_333,
                                 allow_missing_history=True)
        self.assertIsNone(result["history_rgb"])
        self.assertEqual(result["history_missing_reason"],
                         "no completed execution history")

    def test_reader_error_and_stale_image_fail_closed(self):
        buffer, receive = self.ready()
        buffer.reader_error = ValueError("callback")
        with self.assertRaisesRegex(RuntimeError, "reader failed"):
            buffer.snapshot(now_monotonic_ns=receive)
        buffer.reader_error = None
        with self.assertRaisesRegex(RuntimeError, "stale"):
            buffer.snapshot(now_monotonic_ns=receive + 1_000_000_000)


if __name__ == "__main__":
    unittest.main()

