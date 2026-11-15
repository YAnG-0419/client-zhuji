from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from joint_stream import joint_only_transport, source_stream, stream, source_phase
from runtime_cross import stage_guard


class PinocchioReplayTests(unittest.TestCase):
    def test_source_clock_monotonic_for_short_and_long_trajectories(self):
        for total in [.1, .4, 1., 10.]:
            phase = np.array([source_phase(w, total) for w in np.linspace(0, total+2, 10001)])
            self.assertEqual(phase[0], 0.)
            self.assertEqual(phase[-1], total)
            self.assertTrue(np.all(np.diff(phase) >= 0))

    def test_transport_rejects_all_nonjoint_motion(self):
        class Parent:
            pass
        robot = joint_only_transport(Parent)()
        for command in ['request_servop', 'request_movej', 'request_movep']:
            with self.assertRaisesRegex(RuntimeError, 'Only ServoJ'):
                robot.send(command, {})
        with self.assertRaisesRegex(RuntimeError, 'ServoP is disabled'):
            robot.arm_servop()

    def test_prepare_only_uses_joint_alignment_and_never_streams_source(self):
        session = SimpleNamespace(replay_out=None, replay_result={}, prepare_only=True)
        zero = np.zeros(16)
        with patch('joint_stream.measured_segment', return_value=(zero, zero, .02, 1)), \
                patch('joint_stream.stage') as alignment, patch('joint_stream.source_stream') as source:
            answer = stream(session, [0, 1], [0, 1], np.zeros((2, 16)), 300)
        self.assertTrue(answer['prepare_only'])
        self.assertEqual(alignment.call_args.args[5], 'servoj')
        self.assertEqual(alignment.call_args.kwargs['joint_tolerance'], .02)
        source.assert_not_called()

    def simulate(self, fault=None, eef_offset=0.):
        from test_servo_stages import Model
        clock = SimpleNamespace(now=0.)
        q = np.zeros(16); q[14:] = [.1, .05]
        robot = SimpleNamespace(armed=False, commands_sent=0, commands=[])
        robot.healthy = lambda: (q.copy(), np.zeros(16))
        robot.arm_servoj = lambda: setattr(robot, 'armed', True)
        def send(title, payload):
            self.assertEqual(title, 'request_servoj')
            self.assertEqual(len(payload['q']), 16)
            robot.commands.append(payload)
            q[:] = payload['q']
        robot.send = send
        t = np.array([0., .02, .04])
        qs = np.tile(q, (3, 1)); qs[:, 0] = [0., .01, .02]
        eef = np.zeros((3, 2, 7)); eef[:, :, 6] = 1.; eef[:, 0, 0] = qs[:, 0]
        hands = np.zeros((3, 2, 20)); hands[:, :, :] = t[:, None, None]
        session = SimpleNamespace(robot=robot, model=Model(), t=t, eef=eef,
                                  hand_data=hands, replay_result={}, samples=[])
        def tick(hand, start, goal, source):
            if fault and clock.now >= .2:
                raise RuntimeError(fault)
            stage_guard(q.tolist(), [0.]*16, start.tolist(), goal.tolist(), range(14))
            session.samples.append((source, hand.copy()))
            robot.pose = {s+suffix: val for s, i in [('left', 0), ('right', 7)]
                          for suffix, val in [('_position', [float(q[i]+eef_offset), 0., 0.]),
                                              ('_quat', [1., 0., 0., 0.])]}
            return q.copy(), np.zeros(16), None, hand
        session.tick = tick
        with TemporaryDirectory() as folder, \
                patch('joint_stream.time.monotonic', lambda: clock.now), \
                patch('joint_stream.time.sleep', lambda dt: setattr(clock, 'now', clock.now+dt)):
            session.replay_out = Path(folder)
            error = None
            try:
                source_stream(session, np.array([0., .2, .4]), t, qs, 300)
            except Exception as exc:
                error = exc
        return robot, session, error

    def test_joint_stream_and_hands_share_source_phase(self):
        robot, session, error = self.simulate()
        self.assertIsNone(error)
        self.assertGreater(len(robot.commands), 100)
        self.assertTrue(all(x['q'][14:] == [.1, .05] for x in robot.commands))
        phases = np.array([s for s, _ in session.samples])
        self.assertTrue(np.all(np.diff(phases) >= 0))
        self.assertAlmostEqual(phases[-1], .04)
        for phase, hand in session.samples:
            self.assertTrue(np.allclose(hand, phase))
        self.assertTrue(session.replay_result['servoj_stream']['settled'])
        self.assertFalse(robot.armed)

    def test_feedback_error_preserves_stop_cleanup_state(self):
        robot, session, error = self.simulate(fault='stale feedback')
        self.assertIn('stale feedback', str(error))
        self.assertTrue(robot.armed)
        self.assertIn('error', session.replay_result['servoj_stream'])

    def test_eef_tracking_guard_remains_active(self):
        robot, _, error = self.simulate(eef_offset=.05)
        self.assertIn('EEF tracking error', str(error))
        self.assertTrue(robot.armed)

    def test_entry_restores_shared_solver_on_failure(self):
        import replay_hand_pinocchio as entry
        original = entry.base.solve_eef
        solver = SimpleNamespace(solve=lambda *a: None, report=lambda: {'fake': True})
        with TemporaryDirectory() as folder, patch.object(entry, 'PinocchioSolver', return_value=solver), \
                patch.object(entry.execution, 'run', side_effect=RuntimeError('failed')):
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                entry.run(None, Path(folder), {})
        self.assertIs(entry.base.solve_eef, original)


if __name__ == '__main__':
    unittest.main()
