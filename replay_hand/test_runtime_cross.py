from types import SimpleNamespace
import unittest

from runtime_cross import stage_guard, stream_with_stop


class RuntimeTests(unittest.TestCase):
    def guard(self, speed, deviation=0., held=0., indices=range(14)):
        q, dq, zero = [0.] * 16, [0.] * 16, [0.] * 16
        q[0], q[15], dq[list(indices)[0]] = deviation, held, speed
        return stage_guard(q, dq, zero, zero, indices)

    def test_reported_speed_and_new_boundary_pass(self):
        self.assertEqual(self.guard(.4, .0169)['max_speed_rad_s'], .4)
        self.guard(1.)
        self.guard(1.02)
        self.guard(1.2)

    def test_excessive_speed_still_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'speed=1.2100'):
            self.guard(1.21)

    def test_position_and_held_checks_unchanged(self):
        for kwargs in [dict(deviation=.06), dict(held=.04)]:
            with self.assertRaises(RuntimeError):
                self.guard(.4, **kwargs)

    def test_head_speed_unchanged(self):
        with self.assertRaises(RuntimeError):
            self.guard(.3, indices=[14, 15])

    def test_invalid_feedback_rejected(self):
        with self.assertRaises(ValueError):
            self.guard(float('nan'))

    def exercise_stream(self, failure, sent=5, stop_fails=False):
        robot = SimpleNamespace(commands_sent=1, armed=False, stops=0)
        result = {}

        def stop():
            robot.stops += 1
            robot.armed = False
            if stop_fails:
                raise RuntimeError('transport disconnected')

        robot.emergency_stop = stop

        def stream(*args):
            robot.commands_sent += sent
            robot.armed = False  # Reproduce the inherited finally block.
            if failure:
                raise failure
            return {'done': True}

        args = (stream, SimpleNamespace(robot=robot), None, None, None, 300, result)
        if failure:
            with self.assertRaises(type(failure)) as raised:
                stream_with_stop(*args)
            self.assertIs(raised.exception, failure)
        else:
            self.assertTrue(stream_with_stop(*args)['done'])
        return robot, result

    def test_exception_and_interrupt_stop_after_disarm(self):
        for failure in [RuntimeError('tracking'), KeyboardInterrupt()]:
            robot, result = self.exercise_stream(failure)
            self.assertEqual(robot.stops, 1)
            self.assertEqual(result['arm_stop'], 'sent')
            self.assertEqual(result['servop_failure']['commands_sent'], 5)

    def test_failed_stop_is_reported_and_original_error_preserved(self):
        _, result = self.exercise_stream(RuntimeError('tracking'), stop_fails=True)
        self.assertEqual(result['arm_stop'], 'failed')
        self.assertIn('transport disconnected', result['arm_stop_error'])

    def test_no_stop_for_precommand_failure(self):
        robot, result = self.exercise_stream(RuntimeError('start offset'), sent=0)
        self.assertEqual(robot.stops, 0)
        self.assertNotIn('arm_stop', result)

    def test_success_unchanged(self):
        robot, result = self.exercise_stream(None)
        self.assertEqual(robot.stops, 0)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
