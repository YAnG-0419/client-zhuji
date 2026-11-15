"""Bounded ServoP trajectory experiment with accuracy logged, not gated.

Reuses the same certified IK/FK path and timing as normal arm replay. It does
not relocate the recording and retains ArmSession device protections.
"""
import json
import time
import numpy as np


def validate_window(T, t, max_source_seconds=.5, max_motion_seconds=15.):
    if (len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0)
            or not 0 < t[-1]-t[0] <= max_source_seconds+1e-6):
        raise ValueError('Accuracy replay exceeds its source-duration bound or has an invalid time axis')
    if len(T) != len(t) or not np.isfinite(T).all() or np.any(np.diff(T) <= 0) or T[0] != 0 or T[-1] > max_motion_seconds:
        raise ValueError('Accuracy replay exceeds its motion-duration bound or has an invalid time axis')


def run_window(session, reference, duration, rate, out, result, label):
    from episode import pose_error
    from replay_servop import QuatTrack
    from movej_prepare import check_servop_box
    robot, model = session.robot, session.model
    stats = dict(completed=False, duration_s=duration, requested_rate_hz=rate, commands_sent=0,
                 max_eef_position_error_m=[0., 0.], max_eef_rotation_error_rad=[0., 0.],
                 max_joint_error_rad=0., max_speed_rad_s=0., accuracy_policy='record only')
    result[label] = stats
    quat = QuatTrack()
    stamps = []
    period = 1./rate
    with (out/(label+'.jsonl')).open('w') as log:
        robot.arm_servop()
        began, n, last_feedback = time.monotonic(), 0, -float('inf')
        try:
            while True:
                now = time.monotonic(); elapsed = now-began
                if elapsed >= duration:
                    break
                qref, source = reference(elapsed)
                model.set_q(qref); target = model.poses().copy()
                check_servop_box(target)
                if now-last_feedback >= .02:
                    q, dq, _, _ = session.tick(None, qref, qref, source)
                    fb = robot.pose
                    measured = np.array([fb[s+'_position']+fb[s+'_quat'][1:]+fb[s+'_quat'][:1]
                                         for s in ('left', 'right')])
                    ep, er = pose_error(measured, target)
                    error = float(np.max(abs(q[:14]-qref[:14])))
                    row = dict(elapsed_s=elapsed, source_time_s=source, q16=q.tolist(), dq16=dq.tolist(),
                               reference_q16=qref.tolist(), target_eef_xyz_xyzw=target.tolist(),
                               feedback_eef_xyz_xyzw=measured.tolist(), eef_position_error_m=ep.tolist(),
                               eef_rotation_error_rad=er.tolist(), max_joint_error_rad=error,
                               q_feedback_age_s=time.monotonic()-robot.q_at,
                               eef_feedback_age_s=time.monotonic()-robot.pose_at)
                    stats['last_feedback'] = row
                    stats['max_eef_position_error_m'] = np.maximum(stats['max_eef_position_error_m'], ep).tolist()
                    stats['max_eef_rotation_error_rad'] = np.maximum(stats['max_eef_rotation_error_rad'], er).tolist()
                    stats['max_joint_error_rad'] = max(stats['max_joint_error_rad'], error)
                    stats['max_speed_rad_s'] = max(stats['max_speed_rad_s'], float(np.max(abs(dq[:14]))))
                    log.write(json.dumps(row)+'\n');log.flush()
                    last_feedback = now
                if time.monotonic()-began >= duration:
                    break
                left, right = quat(target)
                payload = dict(left_pos=left.tolist(), right_pos=right.tolist())
                robot.send('request_servop', payload)
                robot.commands_sent += 1; stats['commands_sent'] += 1
                robot.lease_at = time.monotonic();stamps.append(time.monotonic())
                stats['last_command_source_time_s'] = source
                if len(stamps) == 1:
                    stats['first_command'] = payload
                n += 1; deadline = began+n*period
                if time.monotonic() > deadline+period:
                    n = int((time.monotonic()-began)/period)
                    continue
                while time.monotonic() < deadline:
                    time.sleep(min(.0005, max(0., deadline-time.monotonic())))
            stats['completed'] = True
        except BaseException as exc:
            stats['error'] = type(exc).__name__+': '+str(exc)
            stats['fault_feedback'] = getattr(robot, 'qstate', None)
            raise
        finally:
            stats['wall_duration_s'] = time.monotonic()-began
            if len(stamps) > 1:
                delta = np.diff(stamps)
                stats.update(actual_rate_hz=float(1./delta.mean()), command_interval_max_ms=float(delta.max()*1000))
            log.write(json.dumps(dict(event='end', **stats))+'\n')
    return stats


def record_replay(session, T, t, qs, rate, out, result, max_source_seconds=.5,
                  max_motion_seconds=15., emergency_stop_on_complete=True):
    from servo_stages import measured_segment, duration_for, min_jerk
    from joint_stream import source_phase
    from replay_servop import interp_rows
    from movej_prepare import check_servop_box
    validate_window(T, t, max_source_seconds, max_motion_seconds)
    start, target, bound, attempts = measured_segment(session, qs[0])
    blend = duration_for(start, target, 3.)
    if blend > 10.:
        raise ValueError('Measured-start takeover exceeds the bounded experiment duration')
    # Validate the entire measured-to-nominal transition, including its workspace.
    for alpha in np.linspace(0., 1., 101):
        session.model.set_q(start+alpha*(target-start));check_servop_box(session.model.poses())
        if time.monotonic()-session.last_tick >= .02:
            session.tick(None, start, start)
    q, dq = session.robot.healthy()
    if np.max(abs(q-start)) > .002 or np.max(abs(dq)) > .02:
        raise RuntimeError('Measured start moved during takeover validation')
    result['accuracy_takeover_path'] = dict(clearance_bound_m=bound, attempts=attempts,
                                          measured_start_q16=start.tolist(), target_q16=target.tolist())
    def takeover(elapsed):
        alpha = min_jerk((elapsed-1.)/blend)
        return start+alpha*(target-start), 0.
    print('SERVOP measured takeover: %.1fs; accuracy recorded only' % (blend+1.), flush=True)
    run_window(session, takeover, blend+1.+.1, rate, out, result, 'accuracy_takeover')
    if not session.prepare_only:
        wall = float(T[-1]+min(1., T[-1]))
        def source_reference(elapsed):
            tau = source_phase(min(elapsed, wall), float(T[-1]))
            return interp_rows(T, qs, tau), float(np.interp(tau, T, t))
        print('SERVOP ACTUAL SOURCE REPLAY %.3fs source / %.1fs motion' % (t[-1], wall), flush=True)
        run_window(session, source_reference, wall+.3, rate, out, result, 'accuracy_source_replay')
        if abs(result['accuracy_source_replay']['last_command_source_time_s']-t[-1]) > 1e-9:
            raise RuntimeError('Source final frame was not sent before the experiment deadline')
        result['source_replayed'] = True
    else:
        result['source_replayed'] = False
    # Most callers retain the historical completion behavior.  Full synchronized
    # replay explicitly defers this stop so it can verify the controller's
    # ServoP command-timeout hold while preserving developer mode.
    if emergency_stop_on_complete:
        session.robot.emergency_stop()
        result['arm_stop'] = 'sent'
        result['arm_stop_reason'] = 'bounded accuracy trajectory experiment complete'
