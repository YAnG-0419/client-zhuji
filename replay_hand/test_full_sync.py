"""Hardware-free tests of shared timing, source synchronization and cleanup."""
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from full_sync import (CombinedSession, latch_servop_hold, synchronized_timeline,
                       refresh_arm_start, run)
from hand_only import HandTrackingError
from accuracy_replay import validate_window
from arm_only import ArmSession


class FullSyncTests(unittest.TestCase):
    def test_servop_command_timeout_hold_keeps_developer_mode(self):
        clock = SimpleNamespace(now=0.)
        q = np.zeros(16)
        robot = SimpleNamespace(backend='supervised_servop', armed=True,
                                info={'data': {'working_mode': 'developer_mode'}},
                                pose={'left_position': [.5, 0., 0.], 'right_position': [.5, 0., 0.]},
                                healthy=lambda: (q.copy(), q.copy()))
        result = {}
        with patch('full_sync.time.monotonic', side_effect=lambda: clock.now), \
                patch('full_sync.time.sleep', side_effect=lambda dt: setattr(clock, 'now', clock.now+dt)):
            latch_servop_hold(robot, result, 'Hand tracking error')
        self.assertFalse(robot.armed)
        self.assertEqual(result['arm_stop'], 'controller_hold')
        self.assertEqual(result['controller_hold']['working_mode'], 'developer_mode')
        self.assertFalse(result['controller_hold']['emergency_stop_sent'])

    def test_refresh_waits_for_stationary_current_pose(self):
        clock = SimpleNamespace(now=0.)
        previous = np.zeros(16)
        def tick(*args):
            q = previous.copy();q[13] = .04+min(clock.now, .2)*.1
            return q, np.zeros(16), None, None
        with patch('full_sync.time.monotonic', side_effect=lambda: clock.now), \
                patch('full_sync.time.sleep', side_effect=lambda dt: setattr(clock, 'now', clock.now+dt)):
            result = {};q = refresh_arm_start(SimpleNamespace(tick=tick), previous, result)
        self.assertGreater(clock.now, .45)
        self.assertAlmostEqual(q[13], .06)
        self.assertTrue(result['post_hand_arm_start']['completed'])
        self.assertEqual(previous[13], 0.)

    def test_refresh_rejects_continuing_motion(self):
        clock = SimpleNamespace(now=0.)
        session = SimpleNamespace(tick=lambda *a: (np.zeros(16), np.full(16, .03), None, None))
        with patch('full_sync.time.monotonic', side_effect=lambda: clock.now), \
                patch('full_sync.time.sleep', side_effect=lambda dt: setattr(clock, 'now', clock.now+dt)):
            with self.assertRaisesRegex(RuntimeError, 'stationary'):
                refresh_arm_start(session, np.zeros(16), {})

    def test_feedback_uses_mechanical_bounds_without_relaxing_target_bounds(self):
        q = np.zeros(16);q[12] = .7515
        model = SimpleNamespace(lower=np.full(16, -.7505), upper=np.full(16, .7505),
                                set_q=lambda v: None, clearance=lambda: (.02, ['base', 'wrist']))
        robot = SimpleNamespace(healthy=lambda: (q.copy(), np.zeros(16)), info={})
        limits = np.tile([-.785398, .785398], (16, 1))
        with TemporaryDirectory() as folder, patch('home_policy.require_status'):
            session = ArmSession(robot, model, np.array([0., 1.]), np.zeros((2, 2, 7)),
                                 Path(folder), feedback_limits=limits)
            session.record_accuracy_only = True
            try:
                session.tick(None, q, q)
                self.assertEqual(model.upper[12], .7505)
                q[12] = .786
                with self.assertRaisesRegex(RuntimeError, 'feedback joint range guard'):
                    session.tick(None, q, q)
            finally:session.feedback_log.close()

    def test_either_component_slows_both_on_one_clock(self):
        t = np.array([0., .02, .04, .06])
        q = np.zeros((4, 16));q[1:, 0] = .03
        h = np.zeros((4, 2, 20));h[2:, 1, 0] = .2
        T = synchronized_timeline(t, q, h, 1.)
        np.testing.assert_allclose(np.diff(T), [.03, .2, .02])
        self.assertLessEqual(np.max(abs(np.diff(q[:, :14], axis=0))/np.diff(T)[:, None]), 1.00000001)
        self.assertLessEqual(np.max(abs(np.diff(h, axis=0))/np.diff(T)[:, None, None]), 1.00000001)

    def test_bad_inputs_and_old_short_window_still_rejected(self):
        q, h = np.zeros((2, 16)), np.zeros((2, 2, 20))
        for t, speed in [([0., 0.], 1.), ([0., 1.], float('nan')), ([0., 1.], 1.1)]:
            with self.assertRaises(ValueError):synchronized_timeline(t, q, h, speed)
        with self.assertRaises(ValueError):validate_window(np.array([0., 40.]), np.array([0., 38.]))
        validate_window(np.array([0., 42.]), np.array([0., 38.]), 60., 120.)
        with self.assertRaises(ValueError):validate_window(np.array([0., 121.]), np.array([0., 38.]), 60., 120.)

    def test_hands_receive_same_source_phase_and_failure_propagates(self):
        s = CombinedSession.__new__(CombinedSession)
        s.t = np.array([0., 1.]);s.hand_data = np.stack([np.zeros((2, 20)), np.ones((2, 20))])
        s.hand_session = Mock();s.hand_stage = 'replay'
        q = np.zeros(16)
        with patch('full_sync.ArmSession.tick', return_value=(q, q, 0., None)):
            s.tick(None, q, q, .25)
            np.testing.assert_array_equal(s.hand_session.tick.call_args.args[0], np.full((2, 20), .25))
            self.assertEqual(s.hand_session.tick.call_args.args[2], .25)
            s.hand_session.tick.side_effect = RuntimeError('hand feedback lost')
            with self.assertRaisesRegex(RuntimeError, 'hand feedback lost'):s.tick(None, q, q, .3)

    def fake_run(self, execute, fault=False):
        q = np.zeros(16);t = np.array([0., 1.]);qs = np.stack([q, q]);h = np.zeros((2, 2, 20))
        poses = np.zeros((2, 7));poses[:, 0] = .5;poses[:, 6] = 1.
        model = Mock(lower=np.full(16, -2.), upper=np.full(16, 2.))
        model.poses.return_value = poses
        robot = Mock(armed=False, commands_sent=0)
        robot.info = {'data': {'working_mode': 'developer_mode'}}
        robot.pose = {'left_position': [.5, 0., 0.], 'right_position': [.5, 0., 0.]}
        robot.snapshot.return_value = dict(robot_info={}, q16=q.tolist(), eef_base_xyz_xyzw=poses.tolist())
        robot.healthy.return_value = (q, q)
        hands = Mock(touched=False)
        hands.node.count_publishers.return_value = 1
        hands.start.side_effect = lambda: setattr(hands, 'touched', True)
        hs = Mock();session = Mock()
        def approach(*args, **kwargs):robot.commands_sent += 1;robot.armed = True
        def replay(*args, **kwargs):
            self.assertFalse(kwargs['emergency_stop_on_complete'])
            if fault:
                robot.backend = 'supervised_servop'
                if fault == 'hand':
                    raise HandTrackingError('Hand tracking error 0.2002 rad')
                raise RuntimeError('device feedback fault')
            args[6]['source_replayed'] = True
        def hold(current, outcome, reason, timeout_s=1.0):
            current.armed = False
            report = dict(reason=reason, completed=True, working_mode='developer_mode',
                          emergency_stop_sent=False)
            outcome['controller_hold'] = report
            outcome['arm_stop'] = 'controller_hold'
            outcome['arm_stop_reason'] = reason
            return report
        ros = SimpleNamespace(init=Mock(), shutdown=Mock())
        args = SimpleNamespace(episode=None, speed=1., rate=300., check=not execute, execute=execute,
                               prepare_seconds=30.)
        result = dict(hand_target_updates=0)
        with TemporaryDirectory() as folder, ExitStack() as ctx:
            patches = {
                'full_sync.load_full_episode': Mock(return_value=(t, np.stack([poses]*2), h, qs, {})),
                'transport.replay_backend': Mock(return_value=(None, lambda: model, None)),
                'full_sync.save_validated_plan': Mock(),
                'full_sync.limits_from_install': Mock(return_value=(np.tile([-1., 1.], (2, 20, 1)), {})),
                'replay.make_plan': Mock(return_value=(qs, None, None, {})),
                'replay.certify_or_record': Mock(return_value={}),
                'robot_home.process_commands': Mock(return_value=['wuji-hand-2-driver']),
                'replay_servop.servo_transport': Mock(return_value=lambda *a: robot),
                'home_policy.require_status': Mock(),
                'full_sync.attach_trace': Mock(), 'full_sync.save_trace': Mock(),
                'full_sync.observed_hands_type': Mock(return_value=lambda target: hands),
                'full_sync.HandSession': Mock(return_value=hs),
                'full_sync.CombinedSession': Mock(return_value=session),
                'full_sync.refresh_arm_start': Mock(return_value=q.copy()),
                'full_sync.approach_movej': Mock(side_effect=approach),
                'full_sync.record_replay': Mock(side_effect=replay),
                'full_sync.latch_servop_hold': Mock(side_effect=hold),
                'full_sync.subprocess.Popen': Mock(side_effect=AssertionError('unexpected driver launch')),
            }
            for name, mock in patches.items():ctx.enter_context(patch(name, mock))
            ctx.enter_context(patch.dict(sys.modules, rclpy=ros))
            if fault:
                message = 'Hand tracking error' if fault == 'hand' else 'device feedback fault'
                with self.assertRaisesRegex(RuntimeError, message):run(args, Path(folder), result)
            else:run(args, Path(folder), result)
            if execute:
                self.assertEqual(patches['full_sync.approach_movej'].call_args.args[3], 30.)
                self.assertEqual(
                    patches['full_sync.HandSession'].call_args.kwargs['tracking_log_only_stages'],
                    {'synchronized_replay'})
        return robot, hands, result

    def test_check_never_enables_or_stops_devices(self):
        robot, hands, result = self.fake_run(False)
        self.assertTrue(result['live_preflight_passed'])
        robot.emergency_stop.assert_not_called();hands.start.assert_not_called();hands.stop.assert_not_called()

    def test_stream_fault_stops_both_components(self):
        robot, hands, result = self.fake_run(True, True)
        robot.emergency_stop.assert_called_once();hands.stop.assert_called_once()
        self.assertEqual(result['hands_stop'], 'confirmed')
        self.assertNotIn('execution_completed', result)

    def test_hand_tracking_fault_latches_hold_without_emergency_stop(self):
        robot, hands, result = self.fake_run(True, 'hand')
        robot.emergency_stop.assert_not_called();hands.stop.assert_called_once()
        self.assertEqual(result['arm_stop'], 'controller_hold')
        self.assertEqual(result['controller_hold']['working_mode'], 'developer_mode')

    def test_success_latches_controller_hold_and_stops_hands(self):
        robot, hands, result = self.fake_run(True)
        robot.emergency_stop.assert_not_called();hands.stop.assert_called_once()
        self.assertTrue(result['execution_completed'])
        self.assertEqual(result['arm_stop'], 'controller_hold')
        self.assertEqual(result['arm_stop_reason'], 'synchronized replay complete')
        self.assertEqual(result['controller_hold']['working_mode'], 'developer_mode')
        self.assertIn('log only during synchronized_replay',
                      result['plan']['hand_tracking_policy'])


if __name__ == '__main__':unittest.main()
