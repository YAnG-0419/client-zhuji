"""First-frame MoveJ adapted from pengyang replay/replay.py::execute_approach.

Only replay_hand uses this duration-aware startup observation window. The
20% window is an observation policy, not a claim about firmware interpolation.
Collision, feedback, envelope, speed, settling and cleanup remain with the
existing replay safety machinery. No robot APIs are invoked on import.
"""

import json
import math
import time

import numpy as np


# Operator-requested first-frame arm tolerance; head hold remains at 0.01 rad.
ARM_FIRST_FRAME_TOL_RAD = 0.1


def startup_grace(duration):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("MoveJ duration must be finite and positive")
    return min(duration, max(1.5, 0.20 * duration))


def execute_approach(base, session, start, goal, duration, out, result):
    grace = startup_grace(duration)
    robot = session.robot
    findings = base.findings
    q, dq = robot.healthy()
    drift = float(np.max(abs(q - start)))
    if drift > .002 or np.max(abs(dq)) > .02:
        detail = 'Approach start moved %.5f rad after the hand approach' % drift
        if not findings.permit('approach_start_moved', detail, value=drift):
            raise RuntimeError('Arm approach start moved')
    from replay_movej_path import check_current_segment

    def pump():
        session.tick(session.hand_data[0], start, start)

    try:
        actual, dq, bound, attempts = check_current_segment(session.model, robot.healthy, goal, pump)
    except (ValueError, RuntimeError) as exc:
        if not findings.permit('approach_uncertified', str(exc)):
            raise
        actual, dq = robot.healthy()
        actual, bound, attempts = actual.copy(), None, 0
    moved = float(np.max(abs(actual - start)))
    if moved > .002 or np.max(abs(dq)) > .02:
        detail = 'Approach start changed %.5f rad during certification' % moved
        if not findings.permit('approach_start_changed', detail, value=moved):
            raise RuntimeError('Arm approach start changed during certification')
    target = findings.clip_to_envelope(session.model.lower[:14], session.model.upper[:14],
                                       goal[:14], 'first-frame approach', base.ROBOT_NAMES)
    report = dict(duration_s=duration, startup_grace_s=grace,
                  arm_position_tolerance_rad=ARM_FIRST_FRAME_TOL_RAD,
                  settle_allowance_s=base.SETTLE_ALLOWANCE_S,
                  movement_threshold_rad=base.NO_MOTION_PROGRESS_RAD,
                  actual_start_q16=actual.tolist(), target_q14=target.tolist(),
                  clearance_bound_m=bound, certification_attempts=attempts,
                  log='arm_approach.jsonl')
    result['arm_approach_watchdog'] = report
    # Open the diagnostic file before sending any motion command.
    with (out / 'arm_approach.jsonl').open('w') as log:
        robot.arm(backend='supervised_movej')
        robot.commands_sent += 1
        report['request_unix_s'] = time.time()
        request_started = time.monotonic()
        robot.query('request_movej', timeout=.08, data={'joint': target.tolist(), 'time': duration})
        began = time.monotonic()
        report['ack_unix_s'] = time.time()
        report['ack_latency_s'] = began - request_started
        log.write(json.dumps(dict(event='movej_ack', **report)) + '\n')
        log.flush()
        print('FIRST FRAME MoveJ startup observation=%.1fs, threshold=%.4frad' %
              (grace, base.NO_MOTION_PROGRESS_RAD), flush=True)
        print('FIRST FRAME arm tolerance=%.3frad, extra settle wait=%.1fs' %
              (ARM_FIRST_FRAME_TOL_RAD, base.SETTLE_ALLOWANCE_S), flush=True)
        settled, last_log = None, -float('inf')
        delta = goal[:14] - actual[:14]
        travel = float(np.max(abs(delta)))
        while True:
            q, dq, _, _ = session.tick(session.hand_data[0], actual, goal)
            now = time.monotonic()
            elapsed = now - began
            movement = float(np.max(abs(q[:14] - actual[:14])))
            error = float(np.max(abs(q[:14] - goal[:14])))
            progress = float(np.dot(q[:14] - actual[:14], delta) /
                             max(float(np.dot(delta, delta)), 1e-12))
            ready = (error < ARM_FIRST_FRAME_TOL_RAD and
                     np.max(abs(q[14:] - goal[14:])) < .01 and np.max(abs(dq)) < .02)
            settled = (now if settled is None else settled) if ready else None
            stalled = (elapsed > grace and travel > base.NO_MOTION_MIN_TRAVEL_RAD and
                       movement < base.NO_MOTION_PROGRESS_RAD)
            reached = elapsed >= duration and settled is not None and now - settled > .3
            timed_out = elapsed > duration + base.SETTLE_ALLOWANCE_S
            sample = dict(elapsed_s=elapsed, unix_s=time.time(), q16=q.tolist(),
                          max_movement_rad=movement, target_progress=progress,
                          max_target_error_rad=error, max_speed_rad_s=float(np.max(abs(dq))),
                          stalled=bool(stalled))
            report['last_feedback'] = sample
            if now - last_log >= .1 or stalled or reached or timed_out:
                log.write(json.dumps(dict(event='feedback', **sample)) + '\n')
                log.flush()
                last_log = now
            if reached:
                robot.armed = False
                return dict(clearance_bound_m=bound, certification_attempts=attempts,
                            duration_s=elapsed, startup_grace_s=grace)
            if stalled:
                detail = ('First-frame MoveJ insufficient startup motion: %.6f rad after %.2f s '
                          '(threshold %.6f rad, command duration %.2f s); '
                          'controller acknowledgement does not prove execution' %
                          (movement, elapsed, base.NO_MOTION_PROGRESS_RAD, duration))
                if not findings.permit('movej_not_executed', detail):
                    raise RuntimeError(detail)
            if timed_out or (stalled and elapsed > grace + 1.):
                detail = ('First-frame approach stopped %.4f rad short after %.1f s; starting the replay'
                          % (error, elapsed))
                if not findings.permit('approach_did_not_settle', detail, value=error):
                    raise TimeoutError('Arm first-frame approach did not settle')
                robot.armed = False
                return dict(clearance_bound_m=bound, certification_attempts=attempts,
                            duration_s=elapsed, settled=False, final_error_rad=error)
            time.sleep(.01)
