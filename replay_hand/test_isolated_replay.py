"""Hardware-free isolation, failure cleanup and trajectory validation tests."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from isolated_cli import select_window
from hand_only import load_hands, validate_targets, hand_timeline, phase, HandSession
from arm_only import ArmSession, test_ik_ready
from episode_cross import SCHEMA, hand_names


class IsolationTests(unittest.TestCase):
    def test_ready_pose_preserves_other_arm_and_head_and_input(self):
        q = np.arange(16, dtype=float)*.01
        before = q.copy()
        for side, offset in [('left', 0), ('right', 7)]:
            ready = test_ik_ready(q, side)
            np.testing.assert_array_equal(q, before)
            np.testing.assert_array_equal(ready[offset:offset+7], [0., 0., 0., -1.6, 0., 0., 0.])
            held = [j for j in range(16) if j not in range(offset, offset+7)]
            np.testing.assert_array_equal(ready[held], q[held])
        both = test_ik_ready(q, 'both')
        np.testing.assert_array_equal(both[:14], [0., 0., 0., -1.6, 0., 0., 0.]*2)
        np.testing.assert_array_equal(both[14:], q[14:])
        np.testing.assert_array_equal(q, before)

    def test_hand_loader_needs_no_arm_arrays_or_model(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'hands.npz'
            np.savez(path, t=[0., .02, .04], hands=np.zeros((3, 2, 20)),
                     metadata_json=json.dumps(dict(schema=SCHEMA)),
                     left_hand_joint_names=hand_names('left'), right_hand_joint_names=hand_names('right'))
            t, values, meta = load_hands(path, 50.)
            self.assertEqual(values.shape, (3, 2, 20))
            self.assertFalse(meta['arm_arrays_read'])

    def test_hand_limits_fail_instead_of_clipping(self):
        limits = np.tile([-1., 1.], (2, 20, 1))
        values = np.zeros((3, 2, 20)); values[2, 1, 0] = 1.01
        with self.assertRaisesRegex(ValueError, 'limits'):
            validate_targets(values, limits)

    def test_hand_timeline_respects_speed_and_phase_is_monotone(self):
        values = np.zeros((2, 2, 20)); values[1, 1, 0] = .5
        T = hand_timeline(np.array([0., .02]), values, .25)
        self.assertGreaterEqual(T[-1], .5)
        for total in [.08, .5, 10.]:
            clock = np.linspace(0., total+min(1., total)+1., 10000)
            samples = np.array([phase(w, total) for w in clock])
            self.assertTrue(np.all(np.diff(samples) >= -1e-12))
            self.assertEqual(samples[0], 0.)
            self.assertEqual(samples[-1], total)

    def test_select_window_rejects_empty_or_out_of_bounds(self):
        t = np.array([0., .02, .04])
        with self.assertRaises(ValueError):
            select_window(t, [], 0., 1.)
        with self.assertRaises(ValueError):
            select_window(t, [], .001, .021)

    def test_arm_guard_records_fault_frame_without_any_hands(self):
        robot = SimpleNamespace(healthy=lambda: (np.zeros(16), np.r_[9.41, np.zeros(15)]), info={})
        with TemporaryDirectory() as folder, patch('home_policy.require_status'):
            session = ArmSession(robot, None, np.array([0., 1.]), np.zeros((2, 2, 7)), Path(folder))
            with self.assertRaisesRegex(RuntimeError, 'speed=9.4100'):
                session.tick(None, np.zeros(16), np.zeros(16))
            session.feedback_log.close()
            row = json.loads((Path(folder)/'arm_feedback.jsonl').read_text())
            self.assertEqual(row['dq16'][0], 9.41)
            self.assertIn('error', row)
            self.assertFalse(hasattr(session, 'hands'))

    def test_arm_guard_accepts_real_controller_integer_zero_velocities(self):
        robot = SimpleNamespace(healthy=lambda: (np.zeros(16), np.zeros(16, dtype=np.int64)), info={})
        model = SimpleNamespace(set_q=lambda q: None, clearance=lambda: (.02, ['a', 'b']),
                                lower=np.full(16, -2.), upper=np.full(16, 2.))
        with TemporaryDirectory() as folder, patch('home_policy.require_status'):
            session = ArmSession(robot, model, np.array([0., 1.]), np.zeros((2, 2, 7)), Path(folder))
            q, dq, progress, unused = session.tick(None, np.zeros(16), np.zeros(16))
            session.feedback_log.close()
            self.assertEqual(int(dq.max()), 0)
            self.assertIsNone(unused)
            row = json.loads((Path(folder)/'arm_feedback.jsonl').read_text())
            self.assertNotIn('error', row)

    def test_hand_guard_records_error_without_setting_another_target(self):
        q = np.zeros((2, 20))
        hands = SimpleNamespace(feedback=lambda: dict(left=q[0].tolist(), right=q[1].tolist()),
                                diagnostics=lambda: {}, set_targets=Mock())
        with TemporaryDirectory() as folder:
            result = dict(hand_target_updates=0)
            session = HandSession(hands, np.tile([-1., 1.], (2, 20, 1)), Path(folder), result)
            q[1, 0] = .21
            with self.assertRaisesRegex(RuntimeError, 'right_finger1_joint1'):
                session.tick(np.zeros((2, 20)), 'prepare')
            session.log.close()
            hands.set_targets.assert_not_called()
            self.assertEqual(result['last_hand_feedback']['tracking_error_rad'], .21)

    def test_expected_contact_logs_tracking_warning_and_continues_replay(self):
        q = np.zeros((2, 20))
        effort = [0.]*20; effort[0] = -.45
        velocity = [0.]*20; velocity[0] = .03
        diagnostics = dict(left={}, right={'joint_state': {
            'effort': effort, 'velocity': velocity}})
        hands = SimpleNamespace(
            feedback=lambda: dict(left=q[0].tolist(), right=q[1].tolist()),
            diagnostics=lambda: diagnostics, set_targets=Mock(),
            node=SimpleNamespace(count_publishers=lambda _: 1))
        with TemporaryDirectory() as folder, \
                patch('hand_only.time.monotonic', side_effect=[0., .01]):
            result = dict(hand_target_updates=0)
            session = HandSession(
                hands, np.tile([-1., 1.], (2, 20, 1)), Path(folder), result,
                tracking_log_only_stages={'synchronized_replay'})
            q[1, 0] = .21
            session.tick(np.zeros((2, 20)), 'synchronized_replay', .5)
            session.log.close()
            row = json.loads((Path(folder)/'hand_feedback.jsonl').read_text())
        hands.set_targets.assert_called_once()
        self.assertEqual(result['hand_target_updates'], 1)
        self.assertEqual(result['hand_tracking_warnings']['count'], 1)
        self.assertEqual(result['hand_tracking_warnings']['worst']['worst_joint'],
                         'right_finger1_joint1')
        self.assertEqual(result['hand_tracking_warnings']['worst']['worst_joint_effort'], -.45)
        self.assertEqual(row['tracking_policy'], 'log_only')
        self.assertIn('tracking_warning', row)

    def test_log_only_contact_policy_still_rejects_measured_joint_limits(self):
        q = np.zeros((2, 20))
        hands = SimpleNamespace(
            feedback=lambda: dict(left=q[0].tolist(), right=q[1].tolist()),
            diagnostics=lambda: {}, set_targets=Mock(),
            node=SimpleNamespace(count_publishers=lambda _: 1))
        with TemporaryDirectory() as folder, \
                patch('hand_only.time.monotonic', side_effect=[0., .01]):
            result = dict(hand_target_updates=0)
            session = HandSession(
                hands, np.tile([-1., 1.], (2, 20, 1)), Path(folder), result,
                tracking_log_only_stages={'synchronized_replay'})
            q[1, 0] = 1.01
            with self.assertRaisesRegex(ValueError, 'Measured Wuji joints'):
                session.tick(np.zeros((2, 20)), 'synchronized_replay', .5)
            session.log.close()
        hands.set_targets.assert_not_called()
        self.assertNotIn('hand_tracking_warnings', result)

    def test_hand_readonly_creates_no_arm_transport_and_never_enables(self):
        import hand_only
        hands = Mock(touched=False)
        hands.feedback.return_value = dict(left=[0.]*20, right=[0.]*20)
        hands.diagnostics.return_value = {}
        hands.node.count_publishers.return_value = 1
        ros = SimpleNamespace(init=Mock(), shutdown=Mock())
        args = SimpleNamespace(episode=None, sample_hz=50., start_seconds=0., end_seconds=None,
                               speed=.05, rate=50., check=True, execute=False, prepare_only=False)
        with TemporaryDirectory() as folder, patch.dict(sys.modules, rclpy=ros), \
                patch('hand_only.load_hands', return_value=(np.array([0., .02]), np.zeros((2, 2, 20)), {})), \
                patch('hand_only.limits_from_install', return_value=(np.tile([-1., 1.], (2, 20, 1)), {})), \
                patch('hand_only.observed_hands_type', return_value=lambda target: hands), \
                patch('robot_home.process_commands', return_value=[]), \
                patch('robot_home.backend', side_effect=AssertionError('arm backend must not be called')), \
                patch('transport.replay_backend', side_effect=AssertionError('arm transport must not be created')), \
                patch('hand_only.subprocess.Popen', side_effect=AssertionError('check must not start drivers')):
            hand_only.run(args, Path(folder), {})
        hands.start.assert_not_called()
        hands.stop.assert_not_called()
        hands.close.assert_called_once()

    def test_hand_prepare_and_replay_complete_with_bounded_targets(self):
        import hand_only
        clock = SimpleNamespace(now=0.)
        def now():
            clock.now += 1e-6
            return clock.now
        q = np.zeros((2, 20))
        sent = []
        def set_targets(target):
            q[:] = [target[s] for s in ('left', 'right')]
            sent.append((clock.now, q.copy()))
        hands = SimpleNamespace(feedback=lambda: dict(left=q[0].tolist(), right=q[1].tolist()),
                                diagnostics=lambda: {}, set_targets=set_targets,
                                node=SimpleNamespace(count_publishers=lambda _: 1))
        with TemporaryDirectory() as folder, patch('hand_only.time.monotonic', side_effect=now), \
                patch('hand_only.time.sleep', side_effect=lambda dt: setattr(clock, 'now', clock.now+dt)):
            result = dict(hand_target_updates=0)
            session = HandSession(hands, np.tile([-1., 1.], (2, 20, 1)), Path(folder), result)
            goal = np.full((2, 20), .1)
            session.prepare(goal, 50.)
            t = np.array([0., .02])
            values = np.stack([goal, goal+.01])
            session.stream(t, values, hand_timeline(t, values, .05), 50.)
            session.log.close()
            self.assertTrue(np.allclose(q, goal+.01))
            changes = np.max(abs(np.diff(np.array([x[1] for x in sent]), axis=0)), axis=(1, 2))
            intervals = np.diff([x[0] for x in sent])
            self.assertLessEqual(float(np.max(changes/intervals)), 1.+1e-5)
            self.assertEqual(result['hand_target_updates'], len(sent))

    def test_hand_failure_stops_hands_only(self):
        import hand_only
        hands = Mock(touched=True)
        hands.feedback.return_value = dict(left=[0.]*20, right=[0.]*20)
        hands.diagnostics.return_value = {}
        hands.node.count_publishers.return_value = 1
        session = Mock()
        session.prepare.side_effect = RuntimeError('hand test fault')
        ros = SimpleNamespace(init=Mock(), shutdown=Mock())
        args = SimpleNamespace(episode=None, sample_hz=50., start_seconds=0., end_seconds=None,
                               speed=.05, rate=50., check=False, execute=True, prepare_only=True)
        with TemporaryDirectory() as folder, patch.dict(sys.modules, rclpy=ros), \
                patch('hand_only.load_hands', return_value=(np.array([0., .02]), np.zeros((2, 2, 20)), {})), \
                patch('hand_only.limits_from_install', return_value=(np.tile([-1., 1.], (2, 20, 1)), {})), \
                patch('hand_only.observed_hands_type', return_value=lambda target: hands), \
                patch('robot_home.process_commands', return_value=['wuji-control']), \
                patch('robot_home.backend', side_effect=AssertionError('no arm backend')), \
                patch('transport.replay_backend', side_effect=AssertionError('no arm transport')), \
                patch('hand_only.HandSession', return_value=session):
            result = {}
            with self.assertRaisesRegex(RuntimeError, 'hand test fault'):
                hand_only.run(args, Path(folder), result)
            self.assertEqual(result['hands_stop'], 'confirmed')
        hands.stop.assert_called_once()
        session.stream.assert_not_called()

    def test_arm_hold_failure_stops_arms_without_ros_or_hands(self):
        import arm_only
        q = np.zeros(16)
        poses = np.zeros((2, 7)); poses[:, 6] = 1.; poses[:, 0] = .5
        model = Mock(lower=np.full(16, -2.), upper=np.full(16, 2.))
        model.poses.return_value = poses
        robot = Mock(armed=False, commands_sent=0)
        robot.snapshot.return_value = dict(q16=q.tolist(), robot_info={}, eef_base_xyz_xyzw=poses.tolist())
        robot.healthy.return_value = (q.copy(), q.copy())
        def fail(*args, **kwargs):
            robot.commands_sent = 8
            # Mimic inherited executor disarming before propagating an error.
            robot.armed = False
            raise RuntimeError('arm test fault')
        args = SimpleNamespace(hold_seconds=.5, check=False, execute=True, prepare_only=False, rate=300.)
        with TemporaryDirectory() as folder, \
                patch('transport.replay_backend', return_value=(None, lambda: model, object)), \
                patch('arm_only.servo_transport', return_value=lambda *a: robot), \
                patch('arm_only.attach_trace'), patch('arm_only.save_trace'), \
                patch('home_policy.require_status'), patch('replay.certify_or_record', return_value={}), \
                patch('robot_home.process_commands', return_value=[]), \
                patch('robot_home.Hands', side_effect=AssertionError('no Hands allowed')), \
                patch.dict(sys.modules, rclpy=SimpleNamespace(init=Mock(side_effect=AssertionError('no ROS init')))), \
                patch('arm_only.measured_segment', return_value=(q, q, .02, 1)), \
                patch('arm_only.stage', side_effect=fail):
            result = {}
            with self.assertRaisesRegex(RuntimeError, 'arm test fault'):
                arm_only.run(args, Path(folder), result)
            self.assertEqual(result['arm_stop'], 'sent')
            self.assertEqual(result['arm_motion_commands_sent'], 8)
        robot.emergency_stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
