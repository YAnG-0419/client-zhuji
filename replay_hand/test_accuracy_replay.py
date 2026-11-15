from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from accuracy_replay import record_replay, validate_window


class AccuracyReplayTests(unittest.TestCase):
    def test_window_and_duration_limits(self):
        validate_window(np.array([0., 10.]), np.array([0., .5]))
        for T, t in [([0., 16.], [0., .5]), ([0., 1.], [0., .6]),
                     ([0., 0.], [0., .1]), ([0., np.nan], [0., .1])]:
            with self.assertRaises(ValueError):
                validate_window(np.array(T), np.array(t))

    def simulate(self, prepare_only=False, fault=False, emergency_stop_on_complete=True):
        clock = SimpleNamespace(now=0.)
        q = np.zeros(16)
        model = SimpleNamespace(q=q.copy())
        model.set_q = lambda v: setattr(model, 'q', np.array(v))
        def poses():
            a = np.zeros((2, 7)); a[:, 6] = 1.; a[:, 0] = .5+model.q[[0, 7]]
            return a
        model.poses = poses
        robot = SimpleNamespace(armed=False, commands_sent=0, sent=[], stops=0, qstate={}, q_at=0., pose_at=0.)
        def healthy():
            return q.copy(), np.zeros(16)
        robot.healthy = healthy
        robot.arm_servop = lambda: setattr(robot, 'armed', True)
        def send(title, payload):
            self.assertEqual(title, 'request_servop')
            robot.sent.append((clock.now, payload))
            q[0] = payload['left_pos'][0]-.5
            q[7] = payload['right_pos'][0]-.5
        robot.send = send
        def stop():
            robot.armed = False; robot.stops += 1
        robot.emergency_stop = stop
        session = SimpleNamespace(robot=robot, model=model, prepare_only=prepare_only, last_tick=0.)
        def tick(*args):
            if fault and clock.now > .1:
                raise RuntimeError('device protection test fault')
            session.last_tick = robot.q_at = robot.pose_at = clock.now
            # Deliberate 6cm tracking offset must be logged rather than gated.
            robot.pose = dict(left_position=[.5+q[0]+.06, 0., 0.], right_position=[.5+q[7], 0., 0.],
                              left_quat=[1., 0., 0., 0.], right_quat=[1., 0., 0., 0.])
            return q.copy(), np.zeros(16), 0., None
        session.tick = tick
        qs = np.zeros((2, 16)); qs[:, 0] = [.01, .03]
        result = {}
        with TemporaryDirectory() as folder, \
                patch('accuracy_replay.time.monotonic', side_effect=lambda: clock.now), \
                patch('accuracy_replay.time.sleep', side_effect=lambda dt: setattr(clock, 'now', clock.now+dt)), \
                patch('servo_stages.measured_segment', return_value=(np.zeros(16), qs[0].copy(), .02, 1)):
            try:
                record_replay(session, np.array([0., 1.]), np.array([0., .1]), qs, 100., Path(folder), result,
                              emergency_stop_on_complete=emergency_stop_on_complete)
                return robot, result, None
            except Exception as exc:
                return robot, result, exc

    def test_replays_original_path_to_final_frame_and_stops(self):
        robot, result, exc = self.simulate()
        self.assertIsNone(exc)
        self.assertTrue(result['source_replayed'])
        self.assertEqual(robot.stops, 1)
        self.assertEqual(robot.sent[0][1]['left_pos'][0], .5)
        self.assertAlmostEqual(result['accuracy_source_replay']['first_command']['left_pos'][0], .51)
        self.assertAlmostEqual(robot.sent[-1][1]['left_pos'][0], .53)
        self.assertEqual(result['accuracy_source_replay']['last_command_source_time_s'], .1)
        self.assertGreater(result['accuracy_source_replay']['max_eef_position_error_m'][0], .05)

    def test_prepare_only_never_advances_source(self):
        robot, result, exc = self.simulate(prepare_only=True)
        self.assertIsNone(exc)
        self.assertNotIn('accuracy_source_replay', result)
        self.assertFalse(result['source_replayed'])
        self.assertAlmostEqual(robot.sent[-1][1]['left_pos'][0], .51)

    def test_caller_can_defer_success_stop_for_verified_controller_hold(self):
        robot, result, exc = self.simulate(emergency_stop_on_complete=False)
        self.assertIsNone(exc)
        self.assertTrue(result['source_replayed'])
        self.assertEqual(robot.stops, 0)
        self.assertTrue(robot.armed)
        self.assertNotIn('arm_stop', result)

    def test_device_fault_prevents_source_and_leaves_outer_cleanup_armed(self):
        robot, result, exc = self.simulate(fault=True)
        self.assertIsInstance(exc, RuntimeError)
        self.assertTrue(robot.armed)
        self.assertNotIn('accuracy_source_replay', result)


if __name__ == '__main__':
    unittest.main()
