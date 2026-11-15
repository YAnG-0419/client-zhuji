import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from servo_stages import min_jerk, duration_for, stage, measured_segment, servo_transport


class Model:
    def set_q(self, q):
        self.q = np.array(q)

    def poses(self):
        eef = np.zeros((2, 7)); eef[:, 6] = 1.
        eef[:, 0] = self.q[[0, 7]]
        return eef


class ServoStagesTests(unittest.TestCase):
    def simulate(self, mode, *, freeze=False, eef_offset=0., duration=1.):
        clock = SimpleNamespace(now=0.)
        model = Model()
        q = np.zeros(16); q[0] = .2; q[7] = -.2; q[14:] = [.1, .05]
        start = q.copy(); target = q.copy(); target[0] += .03; target[7] -= .03
        robot = SimpleNamespace(armed=False, commands_sent=0, q_at=0., pose_at=0., sent=[])
        session = SimpleNamespace(robot=robot, model=model, hand_data=[np.zeros((2, 20))], phases=[])

        def healthy():
            robot.q_at = robot.pose_at = clock.now
            return q.copy(), np.zeros(16)

        def update_pose():
            robot.pose = {side+suffix: value for side, j in [('left', 0), ('right', 7)]
                          for suffix, value in [('_position', [float(q[j]+eef_offset), 0., 0.]),
                                                ('_quat', [1., 0., 0., 0.])]}

        def tick(hands, a, b, phase=None):
            session.phases.append(phase)
            update_pose()
            measured, dq = healthy()
            # Match the runtime path deviation guard on the held reference.
            if np.max(abs(measured[:14]-a[:14])) > np.deg2rad(3):
                raise RuntimeError('tracking guard')
            return measured, dq, None, hands

        def send(title, payload):
            robot.sent.append((clock.now, title, payload))
            if not freeze:
                if title == 'request_servoj':
                    q[:] = payload['q']
                else:
                    q[0], q[7] = payload['left_pos'][0], payload['right_pos'][0]

        robot.healthy = healthy
        robot.arm_servoj = robot.arm_servop = lambda: setattr(robot, 'armed', True)
        robot.send = send
        session.tick = tick
        update_pose()
        result = {}
        with TemporaryDirectory() as folder, \
                patch('servo_stages.time.monotonic', lambda: clock.now), \
                patch('servo_stages.time.sleep', lambda seconds: setattr(clock, 'now', clock.now+seconds)):
            error = None
            try:
                stats = stage(session, start, target, duration, 300, mode,
                              Path(folder), result, .02, 1)
            except Exception as exc:
                error = exc
                stats = next(iter(result.values()))
            logs = [json.loads(line) for line in next(Path(folder).glob('*.jsonl')).read_text().splitlines()]
        return robot, session, start, target, stats, logs, error

    def test_quintic_monotonic_and_speed_acceleration_bounds(self):
        start, goal = np.zeros(16), np.ones(16)
        T = duration_for(start, goal, 0.)
        t = np.linspace(0, T, 10001)
        q = np.array([min_jerk(x/T) for x in t])
        self.assertEqual(q[0], 0.)
        self.assertEqual(q[-1], 1.)
        self.assertTrue(np.all(np.diff(q) >= -1e-12))
        self.assertLessEqual(np.max(np.diff(q)/(t[1]-t[0])), .300001)
        self.assertLessEqual(np.max(abs(np.diff(q, n=2)/(t[1]-t[0])**2)), .50001)

    def test_servoj_measured_first_command_and_head_passthrough(self):
        robot, session, start, _, stats, logs, error = self.simulate('servoj')
        self.assertIsNone(error)
        self.assertEqual(robot.sent[0][1], 'request_servoj')
        self.assertEqual(robot.sent[0][2]['q'], start.tolist())
        self.assertTrue(all(len(x[2]['q']) == 16 and x[2]['filter_ratio'] == 1. for x in robot.sent))
        self.assertTrue(all(x[2]['q'][14:] == start[14:].tolist() for x in robot.sent))
        self.assertTrue(stats['settled'])
        self.assertFalse(robot.armed)
        self.assertTrue(all(x == 0. for x in session.phases))

    def test_servop_hold_and_continuous_join_to_original_first_frame(self):
        robot, session, start, target, stats, logs, error = self.simulate('servop')
        self.assertIsNone(error)
        self.assertEqual(robot.sent[0][2]['left_pos'][0], start[0])
        self.assertTrue(all(x[2]['left_pos'][0] == start[0] for x in robot.sent if x[0] < 1.))
        self.assertAlmostEqual(robot.sent[-1][2]['left_pos'][0], target[0])
        self.assertTrue(all(x == 0. for x in session.phases))
        self.assertIn('actual_rate_hz', stats)
        self.assertIn('feedback_eef_xyz_xyzw', logs[-1])

    def test_frozen_takeover_times_out_and_keeps_cleanup_armed(self):
        robot, _, _, _, stats, _, error = self.simulate('servop', freeze=True)
        self.assertIsInstance(error, TimeoutError)
        self.assertTrue(robot.armed)
        self.assertNotIn('settled', stats)

    def test_takeover_eef_guard_still_stops(self):
        robot, _, _, _, stats, logs, error = self.simulate('servop', eef_offset=.05)
        self.assertIn('takeover EEF error', str(error))
        self.assertTrue(robot.armed)
        self.assertEqual(stats['commands_sent'], 0)
        self.assertEqual(logs[-1]['event'], 'abort')

    def test_collision_rejection_before_arming(self):
        robot = SimpleNamespace(healthy=lambda: (np.zeros(16), np.zeros(16)))
        session = SimpleNamespace(robot=robot, model=None)
        def reject(*a):
            raise ValueError('collision')
        with patch.dict('sys.modules', {'replay_movej_path': SimpleNamespace(check_current_segment=reject)}):
            with self.assertRaisesRegex(ValueError, 'collision'):
                measured_segment(session, np.zeros(16))

    def test_transport_validates_q16_and_arming(self):
        class Parent:
            def send(self, *args, **kw):
                raise AssertionError('unexpected parent call')
        cls = servo_transport(Parent)
        r = cls()
        r.joint_lower, r.joint_upper = np.full(14, -1.), np.full(14, 1.)
        r.healthy = lambda: (np.zeros(16), np.zeros(16))
        r.write_lock, r.closed = threading.Lock(), threading.Event()
        r.armed, r.fault, r.backend = True, None, 'replay_hand_servoj'
        r.info = {'accid': 'fake'}
        packets = []
        r.ws = SimpleNamespace(send=lambda payload: packets.append(json.loads(payload)))
        r.send('request_servoj', {'q': [0.]*16, 'filter_ratio': 1.})
        self.assertEqual(len(packets[0]['data']['q']), 16)
        for q in [[0.]*14, [float('nan')]*16, [2.]*16, [0.]*14+[.1, 0.]]:
            with self.assertRaises(ValueError):
                r.send('request_servoj', {'q': q, 'filter_ratio': 1.})
        r.armed = False
        with self.assertRaises(RuntimeError):
            r.send('request_servoj', {'q': [0.]*16, 'filter_ratio': 1.})

    def test_prepare_only_never_starts_source_stream(self):
        import execution_cross as execution
        session = SimpleNamespace(replay_out=None, replay_result={}, prepare_only=True)
        with patch.object(execution, 'takeover_servop') as takeover, \
                patch.object(execution.upstream, 'stream') as source:
            result = execution.stream(session, [0., 1.], [0., 1.], np.zeros((2, 16)), 300)
        takeover.assert_called_once()
        source.assert_not_called()
        self.assertTrue(result['prepare_only'])

    def test_source_guard_limits_restored_on_failure(self):
        import execution_cross as execution
        session = SimpleNamespace(replay_out=None, replay_result={}, prepare_only=False)
        original = execution.upstream.EEF_TRACK_POS
        with patch.object(execution, 'takeover_servop'), \
                patch.object(execution.upstream, 'stream', side_effect=RuntimeError('fault')):
            with self.assertRaises(RuntimeError):
                execution.stream(session, np.array([0., 1.]), [0., 1.], np.zeros((2, 16)), 300)
        self.assertEqual(execution.upstream.EEF_TRACK_POS, original)


if __name__ == '__main__':
    unittest.main()
