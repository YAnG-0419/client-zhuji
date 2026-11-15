"""Certified MoveJ preparation with test_ik's settle-before-ServoP contract.

Adapted from test_ik/servo_test.py::movej and approach_cross.py. Retains the
replay watchdog, live clearance, head and speed guards; no ServoJ is used.
"""
import time
import numpy as np

from servo_stages import measured_segment, duration_for

# Same boxes checked by test_ik/servo_test.py; keep this constraint explicit.
EEF_BOX = np.array([[[.250, .732], [-.213, .900], [-.673, .5]],
                    [[.250, .732], [-.900, .213], [-.673, .5]]])


def check_servop_box(poses):
    xyz = np.asarray(poses)[..., :3]
    if xyz.shape != (2, 3) or not np.isfinite(xyz).all():
        raise ValueError('Expected two finite ServoP EEF positions')
    if np.any(xyz < EEF_BOX[..., 0]) or np.any(xyz > EEF_BOX[..., 1]):
        raise ValueError('Measured/target EEF outside test_ik ServoP workspace; use certified MoveJ preparation first')


def approach_movej(session, old_start, goal, minimum_duration, out, result, record_accuracy_only=False):
    robot = session.robot
    current, dq = robot.healthy()
    if np.max(abs(current-old_start)) > .002 or np.max(abs(dq)) > .02:
        raise RuntimeError('Arm start moved before MoveJ preparation')
    start, target, bound, attempts = measured_segment(session, goal)
    duration = duration_for(start, target, minimum_duration)
    stats = dict(backend='MoveJ', duration_s=duration, settle_allowance_s=4.,
                 position_tolerance_rad=None if record_accuracy_only else .02,
                 accuracy_policy='record only' if record_accuracy_only else 'require nominal ready',
                 rest_speed_rad_s=.01, stable_seconds=.3,
                 measured_start_q16=start.tolist(), target_q16=target.tolist(),
                 clearance_bound_m=bound, certification_attempts=attempts)
    result['movej_preparation'] = stats
    print('ARM ONLY: MoveJ preparation %.1fs; %s; stationary <0.01rad/s' %
          (duration, 'position error recorded only' if record_accuracy_only else 'position error <0.02rad'), flush=True)
    robot.arm(backend='supervised_movej')
    robot.commands_sent += 1
    began = time.monotonic()
    try:
        reply = robot.query('request_movej', timeout=.08,
                            data={'joint': target[:14].tolist(), 'time': duration})
        if reply.get('result') != 'success':
            raise RuntimeError('MoveJ rejected: '+str(reply))
        stats['ack_latency_s'] = time.monotonic()-began
        settled = None
        while True:
            q, dq, _, _ = session.tick(None, start, target)
            now = time.monotonic()
            elapsed = now-began
            error = float(np.max(abs(q[:14]-target[:14])))
            speed = float(np.max(abs(dq)))
            stats['last_feedback'] = dict(elapsed_s=elapsed, q16=q.tolist(), dq16=dq.tolist(),
                                          max_error_rad=error, max_speed_rad_s=speed)
            ready = ((elapsed >= duration and speed < .01) if record_accuracy_only else
                     (elapsed >= .8*duration and error < .02 and speed < .01))
            settled = (now if settled is None else settled) if ready else None
            if settled is not None and now-settled >= .3:
                stats['completed'] = True
                stats['wall_duration_s'] = elapsed
                robot.armed = False
                return stats
            if elapsed > duration+4.:
                raise TimeoutError('MoveJ did not settle: error %.4frad, speed %.4frad/s' % (error, speed))
            if (elapsed > max(4., .2*duration) and np.max(abs(target[:14]-start[:14])) > .05
                    and np.max(abs(q[:14]-start[:14])) < .002):
                raise RuntimeError('MoveJ acknowledged but no measured motion')
            time.sleep(.01)
    except BaseException as exc:
        stats['error'] = type(exc).__name__+': '+str(exc)
        stats['wall_duration_s'] = time.monotonic()-began
        raise
