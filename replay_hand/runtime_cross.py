"""Replay-specific runtime guard and exceptional ServoP cleanup.

stage_guard is adapted from pengyang/tools/home_policy.py. Only the arm
feedback speed ceiling changes; position and held-joint checks are retained.
"""
import math

ARM_FEEDBACK_SPEED_LIMIT = 1.2  # rad/s, operator request 2026-11-11; target cap stays 1.0.


def stage_guard(q, dq, start, goal, indices):
    from home_policy import finite_vector
    q, dq = finite_vector(q, 16), finite_vector(dq, 16)
    start, goal = finite_vector(start, 16), finite_vector(goal, 16)
    active = set(indices)
    distance = sum((goal[i] - start[i]) ** 2 for i in active)
    alpha = min(1., max(0., sum((q[i] - start[i]) * (goal[i] - start[i])
                                for i in active) / max(distance, 1e-12)))
    deviation = max(abs(q[i] - (start[i] + alpha * (goal[i] - start[i]))) for i in active)
    held_error = max([abs(q[i] - start[i]) for i in range(16) if i not in active] or [0.])
    speed = max(abs(dq[i]) for i in active)
    limit = .25 if active == {14, 15} else ARM_FEEDBACK_SPEED_LIMIT
    if deviation > math.radians(3) or held_error > math.radians(2) or speed > limit:
        raise RuntimeError('Tracking guard: deviation=%.4f held_error=%.4f speed=%.4f limit=%.4f'
                           % (deviation, held_error, speed, limit))
    return dict(progress=alpha, path_deviation_rad=deviation,
                held_error_rad=held_error, max_speed_rad_s=speed)


def stream_with_stop(stream, session, T, t, qs, rate, result):
    """Stop on failure even if the inherited stream already cleared armed."""
    robot = session.robot
    before = robot.commands_sent
    try:
        return stream(session, T, t, qs, rate)
    except BaseException as exc:
        result['servop_failure'] = dict(error=type(exc).__name__ + ': ' + str(exc),
                                       commands_sent=robot.commands_sent - before)
        if robot.commands_sent > before or robot.armed:
            try:
                robot.emergency_stop()
                result['arm_stop'] = 'sent'  # Delivery is not a physical stop confirmation.
            except Exception as stop_error:
                result['arm_stop'] = 'failed'
                result['arm_stop_error'] = type(stop_error).__name__ + ': ' + str(stop_error)
        raise
