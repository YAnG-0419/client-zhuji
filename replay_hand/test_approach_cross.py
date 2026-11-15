"""Hardware-free first-frame regression tests with a simulated clock and robot."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from approach_cross import execute_approach, startup_grace


class ApproachTests(unittest.TestCase):
    def simulate(self, trajectory, expected_error=None, certificate_error=None, tick_error=None):
        clock = SimpleNamespace(now=0.)
        start = np.zeros(16)
        goal = start.copy()
        goal[0] = 1.
        robot = SimpleNamespace(armed=False, commands_sent=0)
        robot.healthy = lambda: (start.copy(), start.copy())
        robot.arm = lambda **kw: setattr(robot, 'armed', True)
        robot.query = lambda *a, **kw: None
        session = SimpleNamespace(robot=robot, model=SimpleNamespace(
            lower=np.full(16, -2.), upper=np.full(16, 2.)), hand_data=[np.zeros((2, 20))])

        def tick(*args):
            clock.now += .05
            if tick_error:
                raise RuntimeError(tick_error)
            q = start.copy()
            q[0] = trajectory(clock.now)
            return q, start.copy(), None, session.hand_data[0]

        session.tick = tick
        base = SimpleNamespace(
            findings=SimpleNamespace(permit=lambda *a, **kw: False,
                                     clip_to_envelope=lambda lower, upper, q, *a: q),
            ROBOT_NAMES=['joint%d' % i for i in range(14)],
            NO_MOTION_PROGRESS_RAD=.002, NO_MOTION_MIN_TRAVEL_RAD=.05,
            SETTLE_ALLOWANCE_S=0.)

        def certify(*args):
            if certificate_error:
                raise ValueError(certificate_error)
            return start.copy(), start.copy(), .02, 1

        def sleep(seconds):
            clock.now += seconds

        result = {}
        with TemporaryDirectory() as folder, \
                patch.dict('sys.modules', {'replay_movej_path': SimpleNamespace(check_current_segment=certify)}), \
                patch('approach_cross.time.monotonic', lambda: clock.now), \
                patch('approach_cross.time.time', lambda: 1000. + clock.now), \
                patch('approach_cross.time.sleep', sleep):
            if expected_error:
                with self.assertRaisesRegex((RuntimeError, TimeoutError, ValueError), expected_error):
                    execute_approach(base, session, start, goal, 20., Path(folder), result)
            else:
                result['success'] = execute_approach(base, session, start, goal, 20., Path(folder), result)
            log = Path(folder) / 'arm_approach.jsonl'
            samples = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, samples, robot, clock.now

    def test_duration_scaled_window_and_invalid_duration(self):
        self.assertEqual(startup_grace(20), 4)
        self.assertEqual(startup_grace(40), 8)
        self.assertEqual(startup_grace(5), 1.5)
        self.assertEqual(startup_grace(.5), .5)
        for duration in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                startup_grace(duration)

    def test_slow_start_below_old_threshold_completes(self):
        # At 1.5 s this has moved only 0.00165 rad, like the reported slow start.
        result, samples, robot, _ = self.simulate(lambda t: min(1., (t / 19.5) ** 2.5))
        early = [s for s in samples if s.get('event') == 'feedback' and 1.5 <= s['elapsed_s'] < 1.7]
        self.assertTrue(early)
        self.assertLess(early[0]['max_movement_rad'], .002)
        self.assertFalse(early[0]['stalled'])
        self.assertEqual(robot.commands_sent, 1)
        self.assertFalse(robot.armed)
        self.assertGreaterEqual(result['success']['duration_s'], 20)
        self.assertEqual(samples[0]['event'], 'movej_ack')

    def test_frozen_robot_stops_after_window(self):
        result, samples, robot, elapsed = self.simulate(lambda t: 0., 'insufficient startup motion')
        self.assertGreater(elapsed, 4.)
        self.assertLess(elapsed, 4.1)
        self.assertTrue(robot.armed)  # Outer servo.run finally must stop the robot.
        self.assertTrue(samples[-1]['stalled'])
        self.assertEqual(result['arm_approach_watchdog']['last_feedback'],
                         {k: v for k, v in samples[-1].items() if k != 'event'})

    def test_small_feedback_jitter_is_not_motion(self):
        self.simulate(lambda t: .0001 * np.sin(t * 100), 'insufficient startup motion')

    def test_motion_does_not_bypass_final_settling(self):
        _, _, robot, elapsed = self.simulate(lambda t: min(.5, t * .03), 'did not settle')
        self.assertGreater(elapsed, 20.)
        self.assertTrue(robot.armed)

    def test_reported_00181_rad_residual_is_accepted(self):
        result, _, robot, _ = self.simulate(lambda t: min(.9819, t * .1))
        self.assertFalse(robot.armed)
        self.assertEqual(result['arm_approach_watchdog']['arm_position_tolerance_rad'], .1)
        self.assertAlmostEqual(result['arm_approach_watchdog']['last_feedback']['max_target_error_rad'], .0181)

    def test_error_above_01_rad_is_rejected(self):
        self.simulate(lambda t: min(.899, t * .1), 'did not settle')

    def test_collision_failure_sends_no_command(self):
        _, _, robot, _ = self.simulate(lambda t: 0., 'collision rejected',
                                      certificate_error='collision rejected')
        self.assertEqual(robot.commands_sent, 0)
        self.assertFalse(robot.armed)

    def test_runtime_guard_propagates_during_startup(self):
        _, _, robot, elapsed = self.simulate(lambda t: 0., 'stale feedback', tick_error='stale feedback')
        self.assertLess(elapsed, 1.5)
        self.assertTrue(robot.armed)


if __name__ == '__main__':
    unittest.main()
