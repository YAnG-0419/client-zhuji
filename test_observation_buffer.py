import unittest

import numpy as np

from observation_buffer import ObservationBuffer, STATE_STREAMS


class ObservationBufferTests(unittest.TestCase):
    def ready(self):
        buffer = ObservationBuffer()
        base_header = 1_800_000_000_000_000_000  # Epoch, deliberately not monotonic.
        base_receive = 50_000_000_000
        raw = bytes(640 * 480 * 3)
        for index in range(6):
            buffer.add_image(base_header + index * 33_333_333,
                             base_receive + index * 33_333_333, raw)
        current = base_header + 5 * 33_333_333
        for stream in STATE_STREAMS:
            buffer.add_state(stream, current - 3_000_000, [1.0, 2.0], ["a", "b"])
        buffer.set_camera_info({"test": True})
        return buffer, base_receive + 5 * 33_333_333

    def test_snapshot_uses_epoch_header_for_alignment_and_monotonic_for_age(self):
        buffer, receive = self.ready()
        result = buffer.snapshot(history_offset_frames=4,
                                 now_monotonic_ns=receive + 20_000_000)
        self.assertEqual(result["head_rgb"].shape, (480, 640, 3))
        self.assertEqual(result["sample_ns"] - result["history_sample_ns"],
                         4 * 33_333_333)
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

