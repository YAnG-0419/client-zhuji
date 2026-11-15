"""Arm-only supervision/execution. Never constructs Hands or initializes ROS.

Adapted from replay/replay.py Session and execution_cross.py. The dummy-shaped
hand_data is solely the legacy sampling interface; no hand feedback is invented.
"""
import json
import time
import numpy as np
from replay_thresholds import LIVE_CLEARANCE_M

from isolated_cli import save, select_window
from runtime_cross import stage_guard, stream_with_stop
from servo_stages import stage, measured_segment
from replay_servop import servo_transport
from movej_prepare import approach_movej, check_servop_box
from wire_trace import attach_trace, save_trace


def test_ik_ready(q16, side):
    """Match test_ik READY_ARM while preserving the other arm and head."""
    if side not in ('left', 'right', 'both'):
        raise ValueError('Expected left, right or both ready arms')
    q = np.array(q16, dtype=float, copy=True)
    if q.shape != (16,) or not np.isfinite(q).all():
        raise ValueError('Ready preparation requires finite q16')
    offsets = (0, 7) if side == 'both' else ((0,) if side == 'left' else (7,))
    for offset in offsets:
        q[offset:offset+7] = [0., 0., 0., -1.6, 0., 0., 0.]
    return q


class ArmSession:
    def __init__(self, robot, model, t, eef, out, feedback_limits=None):
        self.robot, self.model, self.t, self.eef = robot, model, t, eef
        self.hand_data = np.zeros((len(t), 2, 20))  # Sampling compatibility only.
        self.feedback_log = (out/'arm_feedback.jsonl').open('w')
        self.stop_file = out/'stop.request'
        self.last_tick = time.monotonic()
        self.feedback_limits = None
        if feedback_limits is not None:
            bounds = np.asarray(feedback_limits, dtype=float)
            if (bounds.shape != (16, 2) or not np.isfinite(bounds).all()
                    or np.any(bounds[:, 0] >= bounds[:, 1])):
                raise ValueError('Expected finite mechanical feedback bounds for 16 joints')
            self.feedback_limits = bounds.copy()

    def tick(self, ignored_hand_target, start, goal, source_time=None):
        from home_policy import require_status
        now = time.monotonic()
        dt, self.last_tick = now-self.last_tick, now
        if self.stop_file.exists():
            raise RuntimeError('stop.request received')
        if not 0 < dt <= .1:
            raise RuntimeError('Arm supervision stalled: %.4f s' % dt)
        q, dq = self.robot.healthy()
        # Record before guards, including the actual frame that causes a stop.
        row = dict(monotonic_s=now, source_time_s=source_time, q16=q.tolist(),
                   dq16=dq.tolist(), start_q16=start.tolist(), goal_q16=goal.tolist())
        try:
            require_status(self.robot.info)
            # Controller zero velocities may decode to a NumPy int64 array.
            # The shared validator intentionally accepts Python scalars only.
            if getattr(self, 'record_accuracy_only', False):
                from accuracy_hold import accuracy_guard
                guard = accuracy_guard(q, dq, start, goal)
                row['accuracy_guard'] = guard
            else:
                guard = stage_guard(q.tolist(), dq.tolist(), start.tolist(), goal.tolist(), range(14))
            if np.max(abs(q[14:]-start[14:])) > .02 or np.max(abs(dq[14:])) > .20:
                raise RuntimeError('Head moved during arm-only replay')
            self.model.set_q(q)
            margin, pair = self.model.clearance()
            row.update(clearance_score_m=margin, closest_bodies=pair)
            if margin < LIVE_CLEARANCE_M:
                raise RuntimeError('Arm collision clearance guard: %.6fm, %s' % (margin, pair))
            bounds = self.feedback_limits
            if bounds is None:
                bounds = np.column_stack([self.model.lower, self.model.upper])
            invalid = np.flatnonzero((q < bounds[:, 0]) | (q > bounds[:, 1]))
            if len(invalid):
                row['joint_range_violations'] = [dict(index=int(j), measured_rad=float(q[j]),
                    lower_rad=float(bounds[j, 0]), upper_rad=float(bounds[j, 1])) for j in invalid]
                raise RuntimeError('Arm feedback joint range guard: '+str(row['joint_range_violations']))
            self.robot.lease_at = now
            return q, dq, guard['progress'], None
        except BaseException as exc:
            row['error'] = type(exc).__name__+': '+str(exc)
            raise
        finally:
            self.feedback_log.write(json.dumps(row)+'\n')
            self.feedback_log.flush()


def run(args, out, result):
    import replay as base
    import replay_hand as combined
    import execution_cross as execution
    from robot_home import process_commands
    from home_policy import require_status
    from episode import pose_error
    from transport import replay_backend
    robot = session = None
    hold = args.hold_seconds
    ready_arm = getattr(args, 'ready_arm', None)
    record_accuracy_only = getattr(args, 'record_accuracy_only', False)
    if record_accuracy_only and hold is not None and not .2 <= hold <= 2.:
        raise ValueError('Record-only accuracy hold is limited to2s')
    try:
        _, Model, Transport = replay_backend()
        model = Model()
        if hold is None:
            t, eef, unused_hands, seed, meta = combined.cross_loader(args.episode, args.sample_hz, out)
            t, (eef, seed), window = select_window(t, [eef, seed], args.start_seconds, args.end_seconds)
            meta['selected_source_window_s'] = window
            result['source'] = meta
            # No hand trajectory influences arm-only planning or timing.
            qs, _, _, plan = base.make_plan(model, t, eef, np.zeros((len(t), 2, 20)), seed, meta, args.speed, out)
            for reference in qs:
                model.set_q(reference)
                check_servop_box(model.poses())
            result['dense_collision'] = base.certify_or_record(model, qs, 'arm-only dense path')
            T = execution.make_timeline(t, qs, args.speed)
            if record_accuracy_only:
                from accuracy_replay import validate_window
                validate_window(T, t)
            plan.update(component='arms', hand_control=False, requested_rate_hz=args.rate,
                        backend='MoveJ preparation + measured ServoP takeover + ServoP replay',
                        servoj_allowed=False)
            plan['accuracy_policy'] = 'record only' if record_accuracy_only else 'enforce tracking tolerance'
            save(out/'plan.json', plan)
            np.savez_compressed(out/'plan_arm_only.npz', t=t, q16=qs, nominal_time_s=T)
            result['plan'] = plan
        else:
            result['plan'] = dict(component='arms', mode='measured_pose_hold', seconds=hold,
                                  backend='ServoP', source_replayed=False, ik_required=False,
                                  servoj_allowed=False,
                                  hand_control=False, requested_rate_hz=args.rate)
            if ready_arm is not None:
                result['plan'].update(mode='test_ik_ready_then_measured_hold', ready_arm=ready_arm,
                                      ready_joint_target=[0., 0., 0., -1.6, 0., 0., 0.],
                                      movej_minimum_s=20., automatic_return=False)
            result['plan']['accuracy_policy'] = 'record only' if record_accuracy_only else 'enforce tracking tolerance'
            save(out/'plan.json', result['plan'])
        if not (args.check or args.execute):
            result['offline_validated'] = True
            return
        owners = ('wuji_teleop.switch', 'wuji-teleop-switch', 'tools/manus_teleop.sh',
                  'robot_home.py', 'axis_replay.py', 'replay.py', 'servo_test.py',
                  'replay_hand.py', 'replay_hand_pinocchio.py', 'replay_servop.py')
        if any(owner in c for c in process_commands() for owner in owners):
            raise RuntimeError('Another robot controller is running; exit it before isolated testing')
        robot = servo_transport(Transport)(model.lower[:14], model.upper[:14])
        attach_trace(robot)
        state = robot.snapshot()
        save(out/'start_state.json', state)
        require_status(state['robot_info'])
        q0 = np.array(state['q16'])
        model.set_q(q0)
        ep, er = pose_error(model.poses(), np.array(state['eef_base_xyz_xyzw']))
        if max(ep) > .001 or max(er) > .0035:
            raise RuntimeError('Live model FK and SDK EEF disagree')
        if hold is None:
            qs[:, 14:] = q0[14:]
            for reference in qs:
                model.set_q(reference)
                check_servop_box(model.poses())
            result['live_head_collision'] = base.certify_or_record(model, qs, 'arm-only live-head path')
            result['approach_clearance'] = base.certify_or_record(model, np.array([q0, qs[0]]), 'arm-only approach')
        else:
            t = np.array([0., 1.])
            hold_goal = test_ik_ready(q0, ready_arm) if ready_arm else q0.copy()
            model.set_q(hold_goal)
            check_servop_box(model.poses())
            eef = np.stack([model.poses().copy()]*2)
            result['hold_clearance'] = base.certify_or_record(model, np.stack([q0, hold_goal]), 'arm-only ready/hold')
            result['hold_preparation_target_q16'] = hold_goal.tolist()
        robot.start_feedback()
        q, dq = robot.healthy()
        if np.max(abs(q-q0)) > .002 or np.max(abs(dq)) > .02:
            raise RuntimeError('Robot moved during arm-only preflight')
        result['live_preflight_passed'] = True
        if not args.execute:
            return
        session = ArmSession(robot, model, t, eef, out)
        session.record_accuracy_only = record_accuracy_only
        session.replay_out, session.replay_result = out, result
        session.prepare_only = args.prepare_only
        if hold is not None:
            if ready_arm:
                print('ARM ONLY: test_ik %s ready pose via certified MoveJ' % ready_arm, flush=True)
                approach_movej(session, q0, hold_goal, 20., out, result,
                               record_accuracy_only=record_accuracy_only)
                q0, _ = robot.healthy()
                model.set_q(q0)
                check_servop_box(model.poses())
            print('ARM ONLY: ServoP hold measured pose %.1fs; hands disconnected' % hold, flush=True)
            start, target, bound, attempts = measured_segment(session, q0)
            target = start.copy()  # Hold precisely the last certified measured pose.
            if record_accuracy_only:
                from accuracy_hold import record_hold
                record_hold(session, start, hold, args.rate, out, result)
            else:
                stage(session, start, target, hold, args.rate, 'servop', out, result, bound, attempts,
                      joint_tolerance=.02, label='arm_hold')
            result['source_replayed'] = False
        else:
            print('ARM ONLY: preparing first frame; hands disconnected', flush=True)
            approach_movej(session, q0, qs[0], 20., out, result, record_accuracy_only=record_accuracy_only)
            measured, _ = robot.healthy()
            model.set_q(measured)
            check_servop_box(model.poses())
            if record_accuracy_only:
                from accuracy_replay import record_replay
                record_replay(session, T, t, qs, args.rate, out, result)
            else:
                result['servop'] = stream_with_stop(execution.stream, session, T, t, qs, args.rate, result)
            result['source_replayed'] = not args.prepare_only
        result['execution_completed'] = True
    finally:
        errors = []
        if robot:
            if (robot.armed or (robot.commands_sent and not result.get('execution_completed'))) and result.get('arm_stop') != 'sent':
                try:
                    robot.emergency_stop()
                    result['arm_stop'] = 'sent'
                except Exception as exc:
                    errors.append('arm stop: '+str(exc))
            result['arm_motion_commands_sent'] = robot.commands_sent
            try:
                robot.close()
            except Exception as exc:
                errors.append('arm transport close: '+str(exc))
            finally:
                save_trace(robot, out, result)
        if session:
            session.feedback_log.close()
        if errors:
            result['cleanup_errors'] = errors
            raise RuntimeError('; '.join(errors))
