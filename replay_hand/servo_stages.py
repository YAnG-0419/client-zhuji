"""Certified ServoJ preparation and measured-start ServoP takeover.

Protocol format and live head pass-through follow test_ik/servo_test.py.
The first sent reference is the measured start, followed by a quintic scalar
along a certified joint segment. ServoP still relies on controller kinematics;
joint and EEF feedback remain supervised throughout takeover.
"""
import json
import math
import time
import uuid

import numpy as np

from approach_cross import ARM_FIRST_FRAME_TOL_RAD


def min_jerk(s):
    s = min(1., max(0., float(s)))
    return 10*s**3 - 15*s**4 + 6*s**5


def duration_for(start, goal, minimum):
    travel = float(np.max(abs(goal[:14] - start[:14])))
    return max(minimum, 1.875*travel/.30, math.sqrt(5.774*travel/.5))


def servo_transport(Transport):
    from replay_servop import servo_transport as servop_transport
    from home_policy import require_status

    class PreparationTransport(servop_transport(Transport)):
        def arm_servoj(self):
            self.healthy()
            require_status(self.info)
            self.backend = 'replay_hand_servoj'
            self.lease_at = time.monotonic()
            self.armed = True

        def send(self, title, data, guid=None):
            if title != 'request_servoj':
                return super().send(title, data, guid)
            q = np.asarray(data['q'], dtype=float)
            if q.shape != (16,) or not np.isfinite(q).all() or data.get('filter_ratio') != 1.:
                raise ValueError('ServoJ requires finite q16 and filter_ratio=1')
            if np.any(q[:14] < self.joint_lower) or np.any(q[:14] > self.joint_upper):
                raise ValueError('ServoJ arm target outside executable limits')
            current, _ = self.healthy()
            if np.max(abs(q[14:] - current[14:])) > .002:
                raise ValueError('ServoJ head target must pass through current feedback')
            with self.write_lock:
                if (not self.armed or self.backend != 'replay_hand_servoj' or
                        self.fault or self.closed.is_set()):
                    raise RuntimeError('ServoJ preparation transport not healthy and armed')
                self.ws.send(json.dumps(dict(accid=self.info['accid'], title=title,
                                            timestamp=int(time.time()*1000), guid=guid or uuid.uuid4().hex,
                                            data=dict(q=q.tolist(), filter_ratio=1.)), allow_nan=False))
    return PreparationTransport


def measured_segment(session, goal):
    from replay_movej_path import check_current_segment
    robot = session.robot
    start, dq = robot.healthy()
    start = start.copy()
    if np.max(abs(dq)) > .02:
        raise RuntimeError('Servo stage requires a stationary measured start')
    target = goal.copy()
    target[14:] = start[14:]

    def pump():
        session.tick(session.hand_data[0], start, start)

    actual, dq, bound, attempts = check_current_segment(session.model, robot.healthy, target, pump)
    if np.max(abs(actual - start)) > .002 or np.max(abs(dq)) > .02:
        raise RuntimeError('Servo stage start changed during collision certification')
    # Keep the head fixed in the certified reference. ServoJ passes live head
    # feedback through and separately rejects head displacement during streaming.
    target[14:] = start[14:]
    return actual.copy(), target, bound, attempts


def stage(session, start, target, duration, rate, mode, out, result,
          bound, attempts, pos_limit=.04, rot_limit=.20, joint_tolerance=None, label=None):
    from episode import pose_error
    from replay_servop import QuatTrack
    robot, model = session.robot, session.model
    key = label or ('servoj_approach' if mode == 'servoj' else 'servop_takeover')
    # ServoP holds its measured initial pose before approaching the original
    # first frame. Source phase and hand target stay at zero throughout.
    initial_hold = 0. if mode == 'servoj' else 1.
    settle_allowance = 3.
    tolerance = ARM_FIRST_FRAME_TOL_RAD if mode == 'servoj' else .02
    if joint_tolerance is not None:
        tolerance = joint_tolerance
    stats = dict(mode=mode, requested_rate_hz=rate, duration_s=duration,
                 initial_hold_s=initial_hold, settle_allowance_s=settle_allowance,
                 position_tolerance_rad=tolerance, commands_sent=0,
                 measured_start_q16=start.tolist(), target_q16=target.tolist(),
                 clearance_bound_m=bound, certification_attempts=attempts)
    result[key] = stats
    command_times = []
    quat = QuatTrack()
    model.set_q(start)
    stats['measured_start_fk_xyz_xyzw'] = model.poses().copy().tolist()
    last_feedback, settled = -float('inf'), None
    q, dq = robot.healthy()
    if np.max(abs(q-start)) > .002 or np.max(abs(dq)) > .02:
        raise RuntimeError('Servo stage start changed before arming')
    period = 1./rate
    with (out/(key+'.jsonl')).open('w') as log:
        if mode == 'servoj':
            robot.arm_servoj()
        else:
            robot.arm_servop()
        began, tick = time.monotonic(), 0
        try:
            while True:
                now = time.monotonic()
                elapsed = now-began
                # Command 0 is exactly the measured start, independent of loop latency.
                phase = 0. if not command_times else min_jerk((elapsed-initial_hold)/duration)
                reference = start + phase*(target-start)
                reference[14:] = start[14:]
                model.set_q(reference)
                target_pose = model.poses().copy()
                if now-last_feedback >= .02:
                    q, dq, _, _ = session.tick(session.hand_data[0], reference, reference, 0.)
                    if np.max(abs(q[14:]-start[14:])) > .02 or np.max(abs(dq[14:])) > .20:
                        raise RuntimeError('Head moved during arm servo preparation')
                    feedback = robot.pose
                    measured_pose = np.array([feedback[s+'_position']+feedback[s+'_quat'][1:]+
                                              feedback[s+'_quat'][:1] for s in ('left', 'right')])
                    ep, er = pose_error(measured_pose, target_pose)
                    error = float(np.max(abs(q[:14]-target[:14])))
                    sample = dict(elapsed_s=elapsed, source_time_s=0., q16=q.tolist(),
                                  dq16=dq.tolist(), reference_q16=reference.tolist(),
                                  target_eef_xyz_xyzw=target_pose.tolist(),
                                  feedback_eef_xyz_xyzw=measured_pose.tolist(),
                                  eef_position_error_m=ep.tolist(), eef_rotation_error_rad=er.tolist(),
                                  max_target_joint_error_rad=error,
                                  q_feedback_age_s=time.monotonic()-robot.q_at,
                                  eef_feedback_age_s=time.monotonic()-robot.pose_at)
                    stats['last_feedback'] = sample
                    log.write(json.dumps(sample)+'\n')
                    log.flush()
                    if mode == 'servop' and (max(ep) > pos_limit or max(er) > rot_limit):
                        raise RuntimeError('ServoP takeover EEF error %.4f m / %.4f rad' % (max(ep), max(er)))
                    ready = (elapsed >= initial_hold+duration and error < tolerance and
                             np.max(abs(dq)) < .02)
                    settled = (now if settled is None else settled) if ready else None
                    last_feedback = now
                if settled is not None and now-settled >= .3:
                    stats['settled'] = True
                    robot.armed = False
                    break
                if elapsed > initial_hold+duration+settle_allowance:
                    raise TimeoutError(key+' did not settle; see '+key+'.jsonl')
                if mode == 'servoj':
                    current, _ = robot.healthy()
                    reference[14:] = current[14:]
                    payload = dict(q=reference.tolist(), filter_ratio=1.)
                    title = 'request_servoj'
                else:
                    left, right = quat(target_pose)
                    payload = dict(left_pos=left.tolist(), right_pos=right.tolist())
                    title = 'request_servop'
                robot.send(title, payload)
                robot.commands_sent += 1
                robot.lease_at = time.monotonic()
                if not command_times:
                    stats['first_command'] = payload
                command_times.append(time.monotonic())
                stats['commands_sent'] += 1
                tick += 1
                deadline = began+tick*period
                if time.monotonic() > deadline+period:
                    tick = int((time.monotonic()-began)/period)
                    continue
                while time.monotonic() < deadline:
                    time.sleep(min(.0005, max(0., deadline-time.monotonic())))
        except BaseException as exc:
            stats['error'] = type(exc).__name__+': '+str(exc)
            stats['fault_feedback'] = dict(joint_state=getattr(robot, 'qstate', None),
                                           eef=getattr(robot, 'pose', None),
                                           q_at=getattr(robot, 'q_at', None),
                                           pose_at=getattr(robot, 'pose_at', None))
            log.write(json.dumps(dict(event='abort', error=stats['error'],
                                      fault_feedback=stats['fault_feedback'],
                                      elapsed_s=time.monotonic()-began))+'\n')
            log.flush()
            # Keep armed true for the executor's exception cleanup. The wrapper
            # also stops after commands even when an inherited stream disarms.
            raise
        finally:
            stats['wall_duration_s'] = time.monotonic()-began
            if len(command_times) > 1:
                dt = np.diff(command_times)
                stats.update(actual_rate_hz=float(1./np.mean(dt)),
                             command_interval_p99_ms=float(np.percentile(dt, 99)*1000),
                             command_interval_max_ms=float(max(dt)*1000))
    return stats


def approach_servoj(session, old_start, goal, duration, rate, out, result):
    q, _ = session.robot.healthy()
    if np.max(abs(q-old_start)) > .002:
        raise RuntimeError('Arm approach start moved after hand preparation')
    start, target, bound, attempts = measured_segment(session, goal)
    duration = duration_for(start, target, duration)
    return stage(session, start, target, duration, rate, 'servoj', out, result, bound, attempts)


def takeover_servop(session, goal, rate, out, result, pos_limit, rot_limit):
    start, target, bound, attempts = measured_segment(session, goal)
    duration = duration_for(start, target, 3.)
    return stage(session, start, target, duration, rate, 'servop', out, result,
                 bound, attempts, pos_limit, rot_limit)
