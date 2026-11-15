import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from accuracy_hold import accuracy_guard, record_hold
from arm_only import ArmSession


class AccuracyHoldTests(unittest.TestCase):
    def test_large_accuracy_error_is_recorded_but_speed_still_blocks(self):
        q = np.zeros(16); q[0] = .1
        zero = np.zeros(16)
        report = accuracy_guard(q, zero, zero, zero)
        self.assertAlmostEqual(report['path_deviation_rad'], .1)
        dq = zero.copy()
        for speed in (1.02, 1.2):
            dq[0] = speed
            accuracy_guard(q, dq, zero, zero)
        dq[0] = 1.21
        with self.assertRaisesRegex(RuntimeError, 'speed protection'):
            accuracy_guard(q, dq, zero, zero)
        q[0] = np.nan
        with self.assertRaises(ValueError):
            accuracy_guard(q, zero, zero, zero)

    def simulate(self, fault=None):
        clock = SimpleNamespace(now=0.)
        q = np.zeros(16); dq = np.zeros(16)
        model = SimpleNamespace(q=q.copy(), lower=np.full(16, -2.), upper=np.full(16, 2.))
        model.set_q = lambda value: setattr(model, 'q', np.array(value))
        def poses():
            a = np.zeros((2, 7)); a[:, 6] = 1.; a[:, 0] = .5+model.q[[0, 7]]
            return a
        model.poses = poses
        model.clearance = lambda: ((.001 if fault == 'collision' and q[0] else .02), ['a', 'b'])
        robot = SimpleNamespace(armed=False, commands_sent=0, q_at=0., pose_at=0., info={}, qstate={}, sent=[], stops=0)
        def healthy():
            if fault == 'stale' and q[0]:
                raise RuntimeError('feedback stale')
            robot.q_at = robot.pose_at = clock.now
            robot.pose = dict(left_position=[.5+q[0], 0., 0.], right_position=[.5, 0., 0.],
                              left_quat=[1., 0., 0., 0.], right_quat=[1., 0., 0., 0.])
            return q.copy(), dq.copy()
        def send(title, payload):
            self.assertEqual(title, 'request_servop')
            robot.sent.append((clock.now, payload))
            q[0] = .06
            if fault == 'speed':dq[0] = 1.21
            if fault == 'head':q[14] = .021
            if fault == 'limit':q[0] = 2.1
        def stop():
            robot.stops += 1
            robot.armed = False
        robot.healthy, robot.send, robot.emergency_stop = healthy, send, stop
        robot.arm_servop = lambda: setattr(robot, 'armed', True)
        with TemporaryDirectory() as folder, patch('home_policy.require_status'), \
                patch('accuracy_hold.time.monotonic', side_effect=lambda: clock.now), \
                patch('accuracy_hold.time.sleep', side_effect=lambda dt: setattr(clock, 'now', clock.now+dt)):
            session = ArmSession(robot, model, np.array([0., 1.]), poses()[None].repeat(2, axis=0), Path(folder))
            session.last_tick = -.001
            session.record_accuracy_only = True
            result = {}
            try:
                record_hold(session, q.copy(), .3, 100., Path(folder), result)
                return robot, result, None
            except Exception as exc:
                return robot, result, exc
            finally:
                session.feedback_log.close()

    def test_fixed_target_hold_completes_despite_6cm_error_and_stops(self):
        robot, result, exc = self.simulate()
        self.assertIsNone(exc)
        self.assertTrue(result['accuracy_hold']['completed'])
        self.assertGreater(result['accuracy_hold']['max_eef_position_error_m'][0], .059)
        self.assertEqual(result['arm_stop'], 'sent')
        self.assertEqual(robot.stops, 1)
        self.assertTrue(all(t < .3 for t, _ in robot.sent))
        self.assertTrue(all(p == robot.sent[0][1] for _, p in robot.sent))

    def test_device_protections_still_abort(self):
        for fault in ['speed', 'head', 'limit', 'collision', 'stale']:
            with self.subTest(fault=fault):
                robot, result, exc = self.simulate(fault)
                self.assertIsInstance(exc, RuntimeError)
                self.assertFalse(result['accuracy_hold']['completed'])
                self.assertTrue(robot.armed)  # Outer executor owns emergency cleanup.


if __name__ == '__main__':
    unittest.main()
