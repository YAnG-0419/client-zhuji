import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from movej_prepare import approach_movej, check_servop_box


class MoveJPrepareTests(unittest.TestCase):
    def simulate(self, residual=.018, final_speed=0., no_motion=False, reject=False, record_accuracy_only=False):
        clock = SimpleNamespace(now=0.)
        start = np.zeros(16); target = start.copy(); target[0] = .1
        robot = SimpleNamespace(commands_sent=0, armed=False, requests=[])
        robot.healthy = lambda: (start.copy(), np.zeros(16))
        def arm(backend):
            self.assertEqual(backend, 'supervised_movej')
            robot.armed = True
        def query(title, timeout, data):
            self.assertEqual(title, 'request_movej')
            robot.requests.append(data)
            return {'result': 'fail' if reject else 'success'}
        robot.arm, robot.query = arm, query
        def tick(*unused):
            q = start.copy()
            q[0] = 0. if no_motion else min(1., clock.now/3.)*(.1-residual)
            dq = np.zeros(16); dq[0] = final_speed
            return q, dq, None, None
        session = SimpleNamespace(robot=robot, tick=tick)
        result = {}
        with patch('movej_prepare.measured_segment', return_value=(start, target, .02, 1)), \
                patch('movej_prepare.time.monotonic', side_effect=lambda: clock.now), \
                patch('movej_prepare.time.sleep', side_effect=lambda dt: setattr(clock, 'now', clock.now+dt)):
            try:
                approach_movej(session, start, target, 3., None, result, record_accuracy_only=record_accuracy_only)
                return robot, result
            except Exception as exc:
                return robot, result, exc

    def test_movej_settles_without_servoj_and_accepts_018_residual(self):
        robot, result = self.simulate()
        self.assertEqual(robot.commands_sent, 1)
        self.assertEqual(len(robot.requests[0]['joint']), 14)
        self.assertFalse(robot.armed)
        self.assertTrue(result['movej_preparation']['completed'])
        self.assertGreater(result['movej_preparation']['wall_duration_s'], 3.)

    def test_large_residual_blocks_servop_handoff(self):
        robot, result, exc = self.simulate(residual=.025)
        self.assertIsInstance(exc, TimeoutError)
        self.assertTrue(robot.armed)  # Outer executor must stop on failure.
        self.assertNotIn('completed', result['movej_preparation'])

    def test_nonstationary_feedback_blocks_handoff(self):
        robot, result, exc = self.simulate(final_speed=.02)
        self.assertIsInstance(exc, TimeoutError)

    def test_record_only_accepts_stationary_residual_without_changing_target(self):
        robot, result = self.simulate(residual=.05, record_accuracy_only=True)
        self.assertEqual(robot.requests[0]['joint'][0], .1)
        stats = result['movej_preparation']
        self.assertIsNone(stats['position_tolerance_rad'])
        self.assertAlmostEqual(stats['last_feedback']['max_error_rad'], .05)
        self.assertGreaterEqual(stats['wall_duration_s'], 3.3)

    def test_record_only_still_requires_stationary_handoff(self):
        robot, result, exc = self.simulate(residual=.05, final_speed=.02, record_accuracy_only=True)
        self.assertIsInstance(exc, TimeoutError)

    def test_ack_does_not_count_as_motion(self):
        robot, result, exc = self.simulate(no_motion=True)
        self.assertIn('no measured motion', str(exc))

    def test_rejected_movej_aborts(self):
        robot, result, exc = self.simulate(reject=True)
        self.assertIn('rejected', str(exc))

    def test_workspace_refuses_vertical_rest_pose(self):
        poses = np.zeros((2, 7)); poses[:, 6] = 1.; poses[:, 0] = .5
        check_servop_box(poses)
        poses[:, 0] = -.015; poses[:, 2] = -.754
        with self.assertRaisesRegex(ValueError, 'workspace'):
            check_servop_box(poses)


if __name__ == '__main__':
    unittest.main()
