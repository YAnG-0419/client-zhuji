"""Synchronized full-episode ServoP/Wuji execution with a shared limited clock.

Uses tested MoveJ preparation and measured ServoP takeover. Arm accuracy is
recorded; device protections and the existing Wuji tracking guard remain active.
"""
import time
import subprocess
import numpy as np
from replay_thresholds import LIVE_CLEARANCE_M

from arm_only import ArmSession
from hand_only import (HandSession, HandTrackingError, limits_from_install, validate_targets,
                       observed_hands_type, stop_hand_stack)
from isolated_cli import save
from accuracy_replay import record_replay, validate_window
from movej_prepare import approach_movej, check_servop_box
from wire_trace import attach_trace, save_trace
from ik_cache import load_full_episode, save_validated_plan

ARM_TARGET_SPEED_RAD_S = 1.0
HAND_TARGET_SPEED_RAD_S = 1.0


def latch_servop_hold(robot, result, reason, timeout_s=1.0):
    """Stop ServoP streaming and verify the controller's command-timeout hold.

    TRON2 latches the current arm pose after its ServoP command timeout. Disarm
    the local 100 ms watchdog first so it does not replace that hold with
    request_emgy_stop. Failure to become stationary or retain developer mode
    leaves the ordinary emergency-stop cleanup active.
    """
    if getattr(robot, 'backend', None) != 'supervised_servop':
        raise RuntimeError('Controller hold is only available during supervised ServoP')
    began = time.monotonic()
    stable_since = None
    report = dict(reason=reason, command_timeout_s=.2, required_stable_s=.2,
                  timeout_s=timeout_s, emergency_stop_sent=False)
    result['controller_hold'] = report
    robot.armed = False
    while True:
        q, dq = robot.healthy()
        now = time.monotonic()
        speed = float(np.max(abs(dq[:14])))
        report['last_max_arm_speed_rad_s'] = speed
        report['max_arm_speed_rad_s'] = max(report.get('max_arm_speed_rad_s', 0.), speed)
        stable_since = now if speed < .02 and stable_since is None else stable_since
        if speed >= .02:
            stable_since = None
        if now-began >= .25 and stable_since is not None and now-stable_since >= .2:
            mode = robot.info['data'].get('working_mode')
            if mode != 'developer_mode':
                raise RuntimeError('Controller hold left developer_mode: '+str(mode))
            report.update(completed=True, elapsed_s=now-began, working_mode=mode,
                          held_q16=np.asarray(q).tolist(), held_eef_feedback=robot.pose)
            result['arm_stop'] = 'controller_hold'
            result['arm_stop_reason'] = reason
            return report
        if now-began > timeout_s:
            raise RuntimeError('Controller hold did not become stationary within %.1f s' % timeout_s)
        time.sleep(.01)


def synchronized_timeline(t, qs, hands, speed):
    t, qs, hands = map(np.asarray, (t, qs, hands))
    if (not np.isfinite(speed) or not 0 < speed <= 1 or len(t) < 2 or t.ndim != 1
            or qs.shape != (len(t), 16) or hands.shape != (len(t), 2, 20)
            or not all(np.isfinite(a).all() for a in (t, qs, hands))
            or np.any(np.diff(t) <= 0)):
        raise ValueError('Invalid synchronized trajectory')
    # Operator-selected target speed caps, on ONE clock for arms and hands.
    dt = np.maximum.reduce([np.diff(t)/speed,
                           np.max(abs(np.diff(qs[:, :14], axis=0)), axis=1)/ARM_TARGET_SPEED_RAD_S,
                           np.max(abs(np.diff(hands, axis=0)), axis=(1, 2))/HAND_TARGET_SPEED_RAD_S])
    return np.r_[0., np.cumsum(dt)]


def refresh_arm_start(session, previous, result):
    """Re-measure after hands preparation; certify approach from this new pose.

    Arms have not been commanded yet. Hand loading can change their resting
    pose during the >=10s preparation. Never reuse that stale preflight start.
    """
    began = time.monotonic()
    stable_since, anchor = None, None
    report = dict(previous_preflight_q16=previous.tolist(), stable_seconds=.3,
                  rest_speed_rad_s=.01, stable_position_span_rad=.002)
    result['post_hand_arm_start'] = report
    while True:
        q, dq, _, _ = session.tick(None, previous, previous)
        now = time.monotonic()
        if np.max(abs(dq)) >= .01:
            stable_since, anchor = None, None
        elif anchor is None or np.max(abs(q-anchor)) > .002:
            stable_since, anchor = now, q.copy()
        elif now-stable_since >= .3:
            report.update(refreshed_q16=q.tolist(),
                          preflight_shift_rad=float(np.max(abs(q-previous))),
                          elapsed_s=now-began, completed=True,
                          next_step='MoveJ approach re-certifies swept path from refreshed measured state')
            return q.copy()
        if now-began > 2.:
            raise RuntimeError('Arms did not become stationary after hand preparation')
        time.sleep(.01)


class CombinedSession(ArmSession):
    def __init__(self, robot, model, t, eef, values, hand_session, out):
        # SDK MoveJ envelope limits COMMANDS; measured feedback is constrained
        # by physical URDF ranges. Keep target planning/transport limits intact.
        super().__init__(robot, model, t, eef, out,
                         feedback_limits=model.model.jnt_range[model.jids])
        self.hand_data, self.hand_session = values, hand_session
        self.record_accuracy_only = True
        self.prepare_only = False
        self.hand_stage = 'prepare'

    def tick(self, desired, start, goal, source_time=None):
        q, dq, progress, _ = super().tick(None, start, goal, source_time)
        if desired is None:
            phase = 0. if source_time is None else source_time
            desired = np.array([np.interp(phase, self.t, self.hand_data[:, s, j])
                                for s in range(2) for j in range(20)]).reshape(2, 20)
        measured = self.hand_session.tick(desired, self.hand_stage, source_time)
        checker = getattr(self, 'hand_checker', None)
        if checker is not None:
            checker.set_q(q, measured)
            clearance, pair = checker.clearance()
            self.hand_clearance_last = dict(clearance_m=clearance, pair=pair, source_time_s=source_time)
            if clearance < LIVE_CLEARANCE_M:
                raise RuntimeError('Hand/body live clearance %.6fm %s' % (clearance, pair))
        return q, dq, progress, measured

    def prepare_hands(self, q0, result):
        initial, goal = self.hand_session.last.copy(), self.hand_data[0]
        duration = max(10., 1.875*float(np.max(abs(goal-initial)))/.3)
        result['hand_preparation'] = dict(duration_s=duration, tolerance_rad=.20)
        began, settled = time.monotonic(), None
        print('FULL REPLAY: hands preparation %.1fs' % duration, flush=True)
        while True:
            elapsed = time.monotonic()-began
            u = min(1., elapsed/duration)
            desired = initial+(10*u**3-15*u**4+6*u**5)*(goal-initial)
            _, _, _, measured = self.tick(desired, q0, q0)
            error = float(np.max(abs(measured-goal)))
            result['hand_preparation']['last_error_rad'] = error
            ready = elapsed >= duration and error < .20
            settled = (time.monotonic() if settled is None else settled) if ready else None
            if settled is not None and time.monotonic()-settled >= .3:
                result['hand_preparation']['completed'] = True
                return
            if elapsed > duration+8.:
                raise TimeoutError('Hands did not reach first frame: %.4f rad' % error)
            time.sleep(.01)


def run(args, out, result):
    import replay as base
    from transport import replay_backend
    from replay_servop import servo_transport
    from robot_home import Hands, process_commands
    from home_policy import require_status
    from episode import pose_error, SIDES, hand_names
    from isolated_cli import ROOT
    robot = hands = stack = stack_log = session = hand_session = None
    ros_initialized = False
    try:
        recovering = getattr(args, 'ik_recover', False)
        if recovering:
            from replay_robot.ik_recover.planner import load_recovered_episode
            t, eef, values, seed, meta = load_recovered_episode(args, ROOT, out, result)
        else:
            t, eef, values, seed, meta = load_full_episode(args.episode, 50., out)
        result['source'] = meta
        _, Model, Transport = replay_backend()
        model = Model()
        limits, provenance = limits_from_install()
        validate_targets(values, limits)
        result['hand_limits'] = provenance
        qs, _, _, plan = base.make_plan(model, t, eef, values, seed, meta, args.speed, out, reuse_valid_seed=True)
        if recovering and not np.array_equal(qs, seed):
            raise ValueError('Replay planner changed the verified recovery path')
        for q in qs:
            model.set_q(q);check_servop_box(model.poses())
        plan['dense_collision'] = base.certify_or_record(model, qs, 'full synchronized path')
        T = synchronized_timeline(t, qs, values, args.speed)
        validate_window(T, t, 60., 120.)
        plan.update(backend='MoveJ + measured ServoP takeover + synchronized ServoP/Wuji',
                    requested_speed=args.speed, nominal_motion_seconds=float(T[-1]),
                    playback_seconds=float(T[-1]+1.3), requested_arm_rate_hz=args.rate,
                    hand_publish_rate_hz=50., arm_target_speed_limit_rad_s=ARM_TARGET_SPEED_RAD_S,
                    hand_target_speed_limit_rad_s=HAND_TARGET_SPEED_RAD_S, arm_accuracy_policy='record only',
                    hand_tracking_limit_rad=.20,
                    hand_tracking_policy=('0.20 rad hard stop during hand preparation and MoveJ; '
                                          'log only during synchronized_replay for expected object contact'),
                    arm_completion_stop_policy=('Stop ServoP streaming, verify controller command-timeout hold '
                                                'and developer_mode; no request_emgy_stop on success'),
                    arm_fault_stop_policy=('request_emgy_stop except a verified supervised-ServoP '
                                           'hand-tracking/controller hold'),
                    arm_feedback_range_policy='URDF mechanical ranges; target range remains SDK/URDF intersection',
                    timing_policy='One source clock; both components slow together at either speed cap')
        result['plan'] = plan
        save(out/'plan.json', plan)
        np.savez_compressed(out/'plan_full.npz', t=t, q16=qs, hands=values, nominal_time_s=T)
        hand_checker = None
        if meta.get('placement_policy') == 'preserve_common_frame_v1':
            if not meta.get('task_placement'):
                raise RuntimeError('Aligned episode requires a validated common task placement')
            from hand_body_clearance import HandBodyClearance
            hand_checker = HandBodyClearance(ROOT/'replay_robot/models/tron2_wuji_board_v1_20261110/robot.urdf')
            for index in (0, len(qs)//2, len(qs)-1):
                hand_checker.set_q(qs[index], values[index]);model.set_q(qs[index])
                p, r = pose_error(hand_checker.poses(), model.poses())
                if max(p)>1e-6 or max(r)>1e-6:
                    raise RuntimeError('Hand assembly and replay arm FK disagree')
            result['hand_body_path'] = hand_checker.check_path(qs, values)
            save(out/'hand_body_path.json', result['hand_body_path'])
        if not recovering:
            save_validated_plan(ROOT, model, t, eef, values, qs, meta, 50., out)
        print('FULL PLAN READY %.3fs source / %.2fs playback; requested speed %.2f' %
              (t[-1], T[-1]+1.3, args.speed), flush=True)
        if not (args.check or args.execute):
            result['offline_validated'] = True
            return
        commands = process_commands()
        owners = ('wuji_teleop.switch', 'wuji-teleop-switch', 'tools/manus_teleop.sh',
                  'robot_home.py', 'axis_replay.py', 'replay.py', 'servo_test.py',
                  'replay_hand.py', 'replay_hand_pinocchio.py', 'replay_servop.py',
                  'replay_arm_only.py', 'replay_hands_only.py')
        if any(owner in c for c in commands for owner in owners):
            raise RuntimeError('Another controller owns the robot')
        robot = servo_transport(Transport)(model.lower[:14], model.upper[:14])
        attach_trace(robot)
        state = robot.snapshot();save(out/'start_state.json', state)
        require_status(state['robot_info'])
        q0 = np.array(state['q16'])
        model.set_q(q0)
        ep, er = pose_error(model.poses(), np.array(state['eef_base_xyz_xyzw']))
        if max(ep) > .001 or max(er) > .0035:
            raise RuntimeError('Live FK differs from SDK feedback')
        qs[:, 14:] = q0[14:]
        result['live_head_collision'] = base.certify_or_record(model, qs, 'full live-head path')
        if hand_checker is not None:
            result['live_head_hand_body'] = hand_checker.check_path(qs, values)
        if hand_checker is None:
            result['approach_clearance'] = base.certify_or_record(model, np.stack([q0, qs[0]]), 'full approach')
        for q in qs:
            model.set_q(q);check_servop_box(model.poses())
        import rclpy
        rclpy.init();ros_initialized = True
        hands = observed_hands_type(Hands)({s+'_hand': dict(zip(hand_names(s), values[0, i]))
                                          for i, s in enumerate(SIDES)})
        existing = any('wuji-hand-2-driver' in c or 'wuji-control' in c for c in commands)
        if args.execute and not existing:
            stack_log = (out/'stack.log').open('w')
            stack = subprocess.Popen(['ros2', 'launch', str(ROOT/'tools/home_bimanual.launch.py')],
                                     stdout=stack_log, stderr=subprocess.STDOUT, start_new_session=True)
        hands.feedback(timeout=20 if stack else 5)
        hands.validate_owner()
        for side in SIDES:
            if hands.node.count_publishers('/wuji/'+side+'/command') != 1:
                raise RuntimeError('Expected one Wuji control publisher per hand')
        robot.start_feedback()
        q, dq = robot.healthy()
        if np.max(abs(q-q0)) > .002 or np.max(abs(dq)) > .02:
            raise RuntimeError('Robot moved during full preflight')
        result['live_preflight_passed'] = True
        if hand_checker is not None:
            initial_hands = np.array([hands.feedback()[side] for side in SIDES])
            result['hand_preparation_clearance'] = hand_checker.check_path(
                np.stack([q0,q0]), np.stack([initial_hands,values[0]]))
            from hand_approach import plan_approach
            preparation_route, result['preparation_route'] = plan_approach(
                model,hand_checker,q0,qs[0],values[0])
            save(out/'preparation_route.json',result['preparation_route'])
        if not args.execute:
            return
        hands.start()
        hand_session = HandSession(hands, limits, out, result,
                                   tracking_log_only_stages={'synchronized_replay'})
        session = CombinedSession(robot, model, t, eef, values, hand_session, out)
        session.hand_checker = hand_checker
        session.replay_out, session.replay_result = out, result
        session.prepare_hands(q0, result)
        session.hand_stage = 'arm_preparation_hold'
        refreshed_start = refresh_arm_start(session, q0, result)
        if hand_checker is None:
            approach_movej(session, refreshed_start, qs[0], getattr(args, 'prepare_seconds', 20.),
                           out, result, record_accuracy_only=True)
        else:
            result['preparation_legs'] = []
            for index,target in enumerate(preparation_route[1:]):
                if index:
                    refreshed_start = refresh_arm_start(session,preparation_route[index],result)
                leg = dict(index=index+1,target_q16=target.tolist())
                result['preparation_legs'].append(leg)
                leg['hand_clearance'] = hand_checker.check_path(
                    np.stack([refreshed_start,target]), np.stack([values[0],values[0]]),
                    heartbeat=lambda: session.tick(None,refreshed_start,refreshed_start))
                print('CERTIFIED PREPARATION LEG %d/%d'%(index+1,len(preparation_route)-1),flush=True)
                approach_movej(session,refreshed_start,target,getattr(args, 'prepare_seconds', 20.),
                               out,leg,record_accuracy_only=True)
        session.hand_stage = 'synchronized_replay'
        try:
            record_replay(session, T, t, qs, args.rate, out, result,
                          max_source_seconds=60., max_motion_seconds=120.,
                          emergency_stop_on_complete=False)
            latch_servop_hold(robot, result, 'synchronized replay complete')
        except HandTrackingError as exc:
            if getattr(robot, 'backend', None) == 'supervised_servop' and robot.commands_sent:
                latch_servop_hold(robot, result, str(exc))
            raise
        result['execution_completed'] = True
    finally:
        errors = []
        if (robot and (robot.armed or robot.commands_sent)
                and result.get('arm_stop') not in ('sent', 'controller_hold')):
            try:
                robot.emergency_stop();result['arm_stop'] = 'sent'
            except Exception as exc:errors.append('arm stop: '+str(exc))
        if hands:
            if hands.touched:
                try:
                    hands.stop();result['hands_stop'] = 'confirmed'
                except Exception as exc:errors.append('hand stop: '+str(exc))
            try:hands.close()
            except Exception as exc:errors.append('hand close: '+str(exc))
        if session:session.feedback_log.close()
        if hand_session:hand_session.log.close()
        if stack:
            try:stop_hand_stack(stack)
            except Exception as exc:errors.append('stack stop: '+str(exc))
        if stack_log:stack_log.close()
        if robot:
            result['arm_motion_commands_sent'] = robot.commands_sent
            try:robot.close()
            except Exception as exc:errors.append('robot close: '+str(exc))
            finally:save_trace(robot, out, result)
        if ros_initialized:
            import rclpy
            rclpy.shutdown()
        if errors:
            result['cleanup_errors'] = errors
            raise RuntimeError('; '.join(errors))
