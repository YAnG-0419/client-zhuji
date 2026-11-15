"""Bounded accuracy experiment: log tracking error, retain device protection."""
import json
import time
import numpy as np


def accuracy_guard(q, dq, start, goal):
    from home_policy import finite_vector
    from runtime_cross import ARM_FEEDBACK_SPEED_LIMIT
    q, dq = finite_vector(q.tolist(), 16), finite_vector(dq.tolist(), 16)
    start, goal = finite_vector(start.tolist(), 16), finite_vector(goal.tolist(), 16)
    delta = np.array(goal[:14])-start[:14]
    alpha = float(np.clip(np.dot(np.array(q[:14])-start[:14], delta)/max(np.dot(delta, delta), 1e-12), 0., 1.))
    speed = max(abs(v) for v in dq[:14])
    if speed > ARM_FEEDBACK_SPEED_LIMIT:
        raise RuntimeError('Arm speed protection: %.4f > %.4f rad/s' % (speed, ARM_FEEDBACK_SPEED_LIMIT))
    return dict(progress=alpha, max_speed_rad_s=speed,
                path_deviation_rad=float(np.max(abs(np.array(q[:14])-(np.array(start[:14])+alpha*delta)))),
                accuracy_policy='record only')


def record_hold(session, start, duration, rate, out, result):
    """Exactly one constant-pose stream, <=2s; stop explicitly at completion."""
    from episode import pose_error
    from replay_servop import QuatTrack
    from movej_prepare import check_servop_box
    if not .2 <= duration <= 2. or not 50 <= rate <= 500:
        raise ValueError('Accuracy experiment requires duration 0.2..2s and rate 50..500Hz')
    robot, model = session.robot, session.model
    model.set_q(start)
    target = model.poses().copy()
    check_servop_box(target)
    left, right = QuatTrack()(target)
    payload = dict(left_pos=left.tolist(), right_pos=right.tolist())
    q, dq = robot.healthy()
    if np.max(abs(q-start)) > .002 or np.max(abs(dq)) > .02:
        raise RuntimeError('Measured start moved before accuracy hold')
    stats = dict(duration_s=duration, requested_rate_hz=rate, accuracy_policy='record only',
                 target_eef_xyz_xyzw=target.tolist(), first_command=payload, commands_sent=0,
                 max_eef_position_error_m=[0., 0.], max_eef_rotation_error_rad=[0., 0.],
                 max_joint_error_rad=0., max_speed_rad_s=0., completed=False)
    result['accuracy_hold'] = stats
    stamps = []
    began = time.monotonic()
    last_feedback = -float('inf')
    period = 1./rate
    tick = 0
    with (out/'accuracy_hold.jsonl').open('w') as log:
        robot.arm_servop()
        try:
            while time.monotonic()-began < duration:
                now = time.monotonic()
                if now-last_feedback >= .02:
                    q, dq, _, _ = session.tick(None, start, start, 0.)
                    fb = robot.pose
                    measured = np.array([fb[s+'_position']+fb[s+'_quat'][1:]+fb[s+'_quat'][:1]
                                         for s in ('left', 'right')])
                    ep, er = pose_error(measured, target)
                    row = dict(elapsed_s=now-began, q16=q.tolist(), dq16=dq.tolist(),
                               tau=getattr(robot, 'qstate', {}).get('tau'),
                               joint_error_rad=(q-start).tolist(),
                               feedback_eef_xyz_xyzw=measured.tolist(),
                               eef_position_error_m=ep.tolist(), eef_rotation_error_rad=er.tolist(),
                               q_feedback_age_s=time.monotonic()-robot.q_at,
                               eef_feedback_age_s=time.monotonic()-robot.pose_at)
                    stats['last_feedback'] = row
                    stats['max_eef_position_error_m'] = np.maximum(stats['max_eef_position_error_m'], ep).tolist()
                    stats['max_eef_rotation_error_rad'] = np.maximum(stats['max_eef_rotation_error_rad'], er).tolist()
                    stats['max_joint_error_rad'] = max(stats['max_joint_error_rad'], float(np.max(abs(q[:14]-start[:14]))))
                    stats['max_speed_rad_s'] = max(stats['max_speed_rad_s'], float(np.max(abs(dq[:14]))))
                    log.write(json.dumps(row)+'\n'); log.flush()
                    last_feedback = now
                # Fixed deadline also applies after a potentially slow supervision tick.
                if time.monotonic()-began >= duration:
                    break
                robot.send('request_servop', payload)
                robot.commands_sent += 1
                robot.lease_at = time.monotonic()
                stamps.append(time.monotonic())
                stats['commands_sent'] += 1
                tick += 1
                deadline = began+tick*period
                if time.monotonic() > deadline+period:
                    tick = int((time.monotonic()-began)/period)
                    continue
                while time.monotonic() < deadline:
                    time.sleep(min(.0005, max(0., deadline-time.monotonic())))
            robot.emergency_stop()
            result['arm_stop'] = 'sent'
            result['arm_stop_reason'] = 'fixed-duration accuracy experiment complete'
            stats['completed'] = True
        except BaseException as exc:
            stats['error'] = type(exc).__name__+': '+str(exc)
            stats['fault_feedback'] = getattr(robot, 'qstate', None)
            raise
        finally:
            stats['wall_duration_s'] = time.monotonic()-began
            if len(stamps) > 1:
                intervals = np.diff(stamps)
                stats.update(actual_rate_hz=float(1./intervals.mean()),
                             command_interval_max_ms=float(intervals.max()*1000))
            log.write(json.dumps(dict(event='end', **stats))+'\n')
    return stats
