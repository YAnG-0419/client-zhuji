"""Parallel Pinocchio IK + ServoJ executor, adapted from execution_cross.py.

Only the new entry imports this module. No ServoP motion is allowed by its transport.
"""
import subprocess
import numpy as np
import replay as base
from replay_servop import (load_episode, make_timeline, SERVO_SPEED_LIMIT, RAMP_S,
                          process_commands, require_status, pose_error, Hands, hand_names, SIDES, ROOT)
from transport import replay_backend as backend
import findings
from servo_stages import approach_servoj
from joint_stream import joint_only_transport, stream
from wire_trace import attach_trace, save_trace


def run(args, out, result):
    robot = hands = stack = stack_log = session = None
    ros_initialized = False
    try:
        t, eef, hand_data, measured_q, meta = load_episode(args.episode, args.sample_hz)
        end = t[-1] if args.end_seconds is None else args.end_seconds
        if not np.isfinite(args.start_seconds) or not np.isfinite(end) or not 0 <= args.start_seconds < end <= t[-1]:
            raise ValueError('Requested source window must be finite and inside the recording')
        mask = (t >= args.start_seconds) & (t <= end)
        if np.count_nonzero(mask) < 2:
            raise ValueError('Source window contains fewer than two EEF samples')
        origin = float(t[mask][0])
        t, eef, hand_data, measured_q = t[mask]-origin, eef[mask], hand_data[mask], measured_q[mask]
        meta['selected_source_window_s'] = [origin, float(origin+t[-1])]
        result['source'] = meta
        _, Model, Transport = backend()
        model = Model()
        qs, idx, durations, plan = base.make_plan(model, t, eef, hand_data, measured_q, meta, args.speed, out)
        # ServoJ follows the dense joint path, not the waypoint chords: certify it directly.
        print('Checking swept clearance of the dense %d-frame ServoJ path' % len(qs), flush=True)
        result['dense_collision'] = base.certify_or_record(model, qs, 'dense ServoJ path')
        T = make_timeline(t, qs, args.speed)
        ramp_s = min(RAMP_S, float(T[-1]))
        plan.update(backend='EEF -> Pinocchio IK -> certified joint path -> direct ServoJ',
                    servo_rate_hz=args.rate, servo_speed_limit_rad_s=SERVO_SPEED_LIMIT,
                    servo_motion_duration_s=float(T[-1]+ramp_s), servo_ramp_s=ramp_s, first_frame_arm_backend='servoj',
                    takeover='measured joint start, certified ServoJ alignment to original frame zero',
                    prepare_only=args.prepare_only,
                    dense_collision=result['dense_collision'])
        base.save(out/'plan.json', plan)
        np.savez_compressed(out/'plan_servoj.npz', t=t, q16=qs, nominal_time_s=T)
        result['plan'] = plan
        print('PLAN READY source=%.2fs frames=%d servoj motion=%.1fs; preparation=ServoJ approach + alignment'
              % (t[-1], len(t), T[-1]+ramp_s), flush=True)
        if not (args.check or args.execute):
            result['offline_validated'] = True
            return
        commands = process_commands()
        owners = ('wuji_teleop.switch', 'wuji-teleop-switch', 'tools/manus_teleop.sh',
                  'robot_home.py', 'axis_replay.py', 'replay.py')
        if any(owner in c for owner in owners for c in commands if 'replay_servoj.py' not in c):
            raise RuntimeError('Exit teleop/home/replay before ServoJ replay; another controller owns the robot')
        robot = joint_only_transport(Transport)(model.lower[:14], model.upper[:14])
        attach_trace(robot)
        state = robot.snapshot()
        base.save(out/'start_state.json', state)
        try:
            require_status(state['robot_info'])
        except RuntimeError as exc:
            if not findings.permit('robot_status_unexpected', str(exc)):
                raise
        q0 = np.array(state['q16'])
        qs[:, 14:] = q0[14:]
        result['live_head_collision'] = base.certify_or_record(model, qs, 'live head recertification (dense)')
        model.set_q(q0)
        ep, er = pose_error(model.poses(), np.array(state['eef_base_xyz_xyzw']))
        if max(ep) > .001 or max(er) > .0035:
            detail = 'Live model FK and SDK EEF feedback differ by %.5f m / %.5f rad' % (max(ep), max(er))
            if not findings.permit('live_fk_disagrees', detail, value=max(ep)):
                raise RuntimeError('Live model FK and SDK EEF feedback disagree')
        result['arm_approach'] = base.certify_or_record(model, np.array([q0, qs[0]]), 'first-frame ServoJ approach')
        approach_seconds = max(20., 1.875*float(np.max(abs(q0[:14]-qs[0, :14])))/.3)
        result['arm_approach_duration_s'] = approach_seconds
        result['arm_approach_backend'] = 'servoj'
        import rclpy
        rclpy.init()
        ros_initialized = True
        target = {s+'_hand': dict(zip(hand_names(s), hand_data[0, i])) for i, s in enumerate(SIDES)}
        hands = Hands(target)
        existing = any('wuji-hand-2-driver' in c or 'wuji-control' in c for c in commands)
        if args.execute and not existing:
            stack_log = (out/'stack.log').open('w')
            stack = subprocess.Popen(['ros2', 'launch', str(ROOT/'tools/home_bimanual.launch.py')],
                                     stdout=stack_log, stderr=subprocess.STDOUT, start_new_session=True)
        current_hands = np.array([hands.feedback(timeout=20 if stack else 5)[s] for s in SIDES])
        hands.validate_owner()
        for s in SIDES:
            if hands.node.count_publishers('/wuji/'+s+'/command') != 1:
                raise RuntimeError('Expected exactly one wuji_control output publisher for '+s)
        robot.start_feedback()
        result['live_preflight_passed'] = True
        if not args.execute:
            return
        q, dq = robot.healthy()
        shift = float(np.max(abs(q-q0)))
        if shift > .002 or np.max(abs(dq)) > .02:
            detail = 'Robot moved %.5f rad during preflight' % shift
            if not findings.permit('moved_during_preflight', detail, value=shift):
                raise RuntimeError('Robot moved during preflight')
        hands.start()
        session = base.Session(robot, hands, model, t, eef, hand_data, out)
        session.replay_out, session.replay_result = out, result
        session.prepare_only = args.prepare_only
        hand_approach_seconds = max(10., 1.875*float(np.max(abs(hand_data[0]-current_hands)))/.3)
        print('MOVING FIRST FRAME hands duration=%.1fs' % hand_approach_seconds, flush=True)
        session.approach_hands(hand_approach_seconds, q0)
        if np.max(abs(q0[:14]-qs[0, :14])) > .002:
            print('MOVING FIRST FRAME arms ServoJ duration=%.1fs' % approach_seconds, flush=True)
            result['approach_execution'] = approach_servoj(session, q0, qs[0], approach_seconds, args.rate, out, result)
        print('REACHED FIRST FRAME', flush=True)
        print('SERVOP TAKEOVER from measured pose at %.0f Hz' % args.rate, flush=True)
        result['servoj'] = stream(session, T, t, qs, args.rate)
        print('PREPARATION DONE' if args.prepare_only else 'SERVOP REPLAY DONE', result['servoj'], flush=True)
        result['replayed'] = not args.prepare_only
        result['preparation_completed'] = True
    finally:
        cleanup_errors = []
        if robot and (robot.armed or (robot.commands_sent and not result.get('preparation_completed'))) and result.get('arm_stop') != 'sent':
            try:
                robot.emergency_stop()
                result['arm_stop'] = 'sent'
            except Exception as exc:
                cleanup_errors.append('arm stop: '+str(exc))
        if hands:
            if hands.touched:
                try:
                    hands.stop()
                    result['hands_stop'] = 'confirmed'
                except Exception as exc:
                    cleanup_errors.append('hand stop: '+str(exc))
            try:
                hands.close()
            except Exception as exc:
                cleanup_errors.append('hand close: '+str(exc))
        if session:
            session.feedback_log.close()
        if stack:
            try:
                base.stop_stack(stack)
            except Exception as exc:
                cleanup_errors.append('stack shutdown: '+str(exc))
        if stack_log:
            stack_log.close()
        if robot:
            result['motion_commands_sent'] = robot.commands_sent
            try:
                robot.close()
            finally:
                save_trace(robot, out, result)
        if ros_initialized:
            import rclpy
            rclpy.shutdown()
        if cleanup_errors:
            result['cleanup_errors'] = cleanup_errors
            raise RuntimeError('; '.join(cleanup_errors))
