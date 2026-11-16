import unittest

from live_planner import resolve_timing


HUMANHAND = {"contract": "humanhand", "speed_ratio": 4.0,
             "training_history_offset_frames": 4}
REALROBOT = {"contract": "realrobot", "speed_ratio": 1.0,
             "training_history_offset_frames": 16}


class ResolveTimingTests(unittest.TestCase):
    def test_server_defaults(self):
        self.assertEqual(resolve_timing(HUMANHAND), (4.0, 16))
        self.assertEqual(resolve_timing(REALROBOT), (1.0, 16))

    def test_requested_ratio_scales_the_history_offset(self):
        # The 2026-11-17 humanhand run is served at ratio 2: 8 history frames
        # and 32 steps over 0.53 s instead of 16 frames over 1.07 s.
        self.assertEqual(resolve_timing(HUMANHAND, 2.0), (2.0, 8))
        self.assertEqual(resolve_timing(HUMANHAND, 1.0), (1.0, 4))
        self.assertEqual(resolve_timing(HUMANHAND, 8.0), (8.0, 32))
        self.assertEqual(resolve_timing(REALROBOT, 2.0), (2.0, 32))

    def test_follows_a_changed_server_default(self):
        self.assertEqual(resolve_timing(dict(HUMANHAND, speed_ratio=2.0)), (2.0, 8))

    def test_falls_back_for_a_server_without_speed_ratio(self):
        self.assertEqual(resolve_timing({"contract": "humanhand"}), (4.0, 16))
        self.assertEqual(resolve_timing({"contract": "realrobot"}), (1.0, 16))

    def test_rejects_out_of_range_or_degenerate_ratios(self):
        for ratio in (0.1, 20.0):
            with self.assertRaises(ValueError):
                resolve_timing(HUMANHAND, ratio)
        with self.assertRaises(ValueError):
            resolve_timing({"contract": "humanhand",
                            "training_history_offset_frames": 1}, 0.25)


if __name__ == "__main__":
    unittest.main()
