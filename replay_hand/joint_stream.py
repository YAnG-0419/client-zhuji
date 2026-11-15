"""Pinocchio-planned joint trajectory execution. No ServoP commands."""
import json
import time

import numpy as np

from servo_stages import duration_for, measured_segment, stage


def source_phase(wall_time, total):
    """Monotonic ease-in/out, including trajectories shorter than one second."""
    ramp = min(1., total)
    wall_time = min(max(0., wall_time), total+ramp)
    def integral(x):
        u = min(max(x/ramp, 0.), 1.)
        return ramp*(u**3-.5*u**4)
    if wall_time <= ramp:
        return integral(wall_time)
    if wall_time >= total:
        return total-integral(total+ramp-wall_time)
    return wall_time-ramp/2


def joint_only_transport(Transport):
    from servo_stages import servo_transport
    class JointOnly(servo_transport(Transport)):
        def arm_servop(self):
            raise RuntimeError('ServoP is disabled in the Pinocchio ServoJ entry')

        def send(self, title, data, guid=None):
            if title in ('request_servop', 'request_movej', 'request_movep'):
                raise RuntimeError('Only ServoJ motion is allowed in this entry')
            return super().send(title, data, guid)
    return JointOnly


def stream(session, T, t, qs, rate):
    start, target, bound, attempts = measured_segment(session, qs[0])
    # Source clock remains stopped until alignment. Same first-frame target;
    # no reanchoring, no controller-side Cartesian IK and no protocol switch.
    stage(session, start, target, duration_for(start, target, 3.), rate, 'servoj',
          session.replay_out, session.replay_result, bound, attempts,
          joint_tolerance=.02, label='servoj_alignment')
    if session.prepare_only:
        return dict(prepare_only=True, source_replayed=False)
    return source_stream(session, T, t, qs, rate)


def source_stream(session, T, t, qs, rate):
    from episode import pose_error, sample
    from replay_servop import interp_rows
    robot, model = session.robot, session.model
    q, dq = robot.healthy()
    if np.max(abs(q[:14]-qs[0, :14])) > .02 or np.max(abs(dq)) > .02:
        raise RuntimeError('ServoJ replay start is not stationary at frame zero')
    head = q[14:].copy()
    total, period = float(T[-1]), 1./rate
    wall = total+min(1., total)
    stats = dict(commands_sent=0, requested_rate_hz=rate,
                 max_eef_error_m=0., max_eef_error_rad=0., max_joint_error_rad=0.,
                 max_joint_speed_rad_s=0., source_replayed=True)
    session.replay_result['servoj_stream'] = stats
    times = []
    settled = None
    with (session.replay_out/'servoj_stream.jsonl').open('w') as log:
        robot.arm_servoj()
        began, last_feedback, n = time.monotonic(), -float('inf'), 0
        print('PINOCCHIO SERVOJ REPLAY START %.1fs at %.0f Hz' % (wall, rate), flush=True)
        try:
            while True:
                now = time.monotonic()
                elapsed = now-began
                tau = source_phase(elapsed, total)
                reference = interp_rows(T, qs, tau)
                source = float(np.interp(tau, T, t))
                if now-last_feedback >= .02:
                    _, hands = sample(session.t, session.eef, session.hand_data, source)
                    q, dq, _, _ = session.tick(hands, reference, reference, source)
                    if np.max(abs(q[14:]-head)) > .02 or np.max(abs(dq[14:])) > .20:
                        raise RuntimeError('Head moved during ServoJ source replay')
                    model.set_q(reference)
                    desired = model.poses().copy()
                    fb = robot.pose
                    measured = np.array([fb[s+'_position']+fb[s+'_quat'][1:]+fb[s+'_quat'][:1]
                                         for s in ('left', 'right')])
                    ep, er = pose_error(measured, desired)
                    joint_error = float(np.max(abs(q[:14]-reference[:14])))
                    stats['max_eef_error_m'] = max(stats['max_eef_error_m'], float(max(ep)))
                    stats['max_eef_error_rad'] = max(stats['max_eef_error_rad'], float(max(er)))
                    stats['max_joint_error_rad'] = max(stats['max_joint_error_rad'], joint_error)
                    stats['max_joint_speed_rad_s'] = max(stats['max_joint_speed_rad_s'], float(max(abs(dq[:14]))))
                    row = dict(elapsed_s=elapsed, source_time_s=source, q16=q.tolist(), dq16=dq.tolist(),
                               reference_q16=reference.tolist(), desired_eef_xyz_xyzw=desired.tolist(),
                               feedback_eef_xyz_xyzw=measured.tolist(), eef_position_error_m=ep.tolist(),
                               eef_rotation_error_rad=er.tolist())
                    stats['last_feedback'] = row
                    log.write(json.dumps(row)+'\n'); log.flush()
                    if max(ep) > .04 or max(er) > .20:
                        raise RuntimeError('ServoJ EEF tracking error %.4f m / %.4f rad' % (max(ep), max(er)))
                    ready = elapsed >= wall and joint_error < .02 and np.max(abs(dq)) < .02
                    settled = (now if settled is None else settled) if ready else None
                    last_feedback = now
                if settled is not None and now-settled >= .3:
                    stats['settled'] = True
                    robot.armed = False
                    return stats
                if elapsed > wall+3.:
                    raise TimeoutError('ServoJ final frame did not settle')
                current, _ = robot.healthy()
                reference[14:] = current[14:]
                robot.send('request_servoj', dict(q=reference.tolist(), filter_ratio=1.))
                robot.commands_sent += 1
                robot.lease_at = time.monotonic()
                times.append(time.monotonic())
                stats['commands_sent'] += 1
                n += 1
                deadline = began+n*period
                if time.monotonic() > deadline+period:
                    n = int((time.monotonic()-began)/period)
                    continue
                while time.monotonic() < deadline:
                    time.sleep(min(.0005, max(0., deadline-time.monotonic())))
        except BaseException as exc:
            stats['error'] = type(exc).__name__+': '+str(exc)
            # Keep armed for executor cleanup, including failures before first send.
            raise
        finally:
            stats['wall_duration_s'] = time.monotonic()-began
            if len(times) > 1:
                dt = np.diff(times)
                stats.update(actual_rate_hz=float(1./np.mean(dt)),
                             command_interval_p99_ms=float(np.percentile(dt, 99)*1000),
                             command_interval_max_ms=float(max(dt)*1000))
