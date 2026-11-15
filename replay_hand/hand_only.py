"""Wuji-only replay through wuji_control; no Tron2 transport or arm model."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

from episode_cross import SIDES, SCHEMA, hand_names
from isolated_cli import ROOT, save, select_window

TRACK_LIMIT = .20
TARGET_SPEED = 1.0


class HandTrackingError(RuntimeError):
    """A measured Wuji joint exceeded the supervised target-error limit."""


def load_hands(path, hz):
    """Read only time and Wuji arrays; no EEF, arm URDF, FK or IK dependency."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        t, hands = data['t'].astype(float), data['hands'].astype(float)
        meta = json.loads(str(data['metadata_json']))
        for side in SIDES:
            if data[side+'_hand_joint_names'].tolist() != hand_names(side):
                raise ValueError('Unexpected '+side+' joint order')
    if meta.get('schema') != SCHEMA:
        raise ValueError('Unsupported episode schema')
    if t.ndim != 1 or len(t) < 2 or hands.shape != (len(t), 2, 20):
        raise ValueError('Expected time axis and (N,2,20) hands')
    if not np.isfinite(t).all() or not np.isfinite(hands).all():
        raise ValueError('Nonfinite time or hand target')
    if abs(t[0]) > 1e-9 or np.any(np.diff(t) <= 0) or np.max(np.diff(t)) > .1:
        raise ValueError('Invalid source time axis')
    query = np.unique(np.r_[np.arange(0., t[-1], 1./hz), t[-1]])
    values = np.stack([np.interp(query, t, hands[:, s, j])
                       for s in range(2) for j in range(20)], axis=1).reshape(-1, 2, 20)
    return query, values, dict(episode=str(path.resolve()), episode_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               hand_source=meta.get('hand_source'), arm_arrays_read=False)


def limits_from_install():
    from ament_index_python.packages import get_package_share_directory
    path = Path(get_package_share_directory('wuji_control'))/'config/joint_limits.json'
    data = json.loads(path.read_text())
    limits = np.array([data[s] for s in SIDES], dtype=float)
    if limits.shape != (2, 20, 2) or not np.isfinite(limits).all() or np.any(limits[..., 0] >= limits[..., 1]):
        raise ValueError('Invalid installed Wuji limits')
    return limits, dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def validate_targets(hands, limits):
    if not np.isfinite(hands).all() or np.any(hands < limits[..., 0]) or np.any(hands > limits[..., 1]):
        raise ValueError('Wuji targets exceed installed joint limits')


def hand_timeline(t, hands, speed):
    travel = np.max(abs(np.diff(hands, axis=0)), axis=(1, 2))
    return np.r_[0., np.cumsum(np.maximum(np.diff(t)/speed, travel/TARGET_SPEED))]


def phase(w, total):
    ramp = min(1., total)
    wall = total+ramp
    def ease(x):
        u = min(1., max(0., x/ramp))
        return ramp*(u**3-u**4/2)
    if w <= ramp:
        return ease(w)
    if w >= wall-ramp:
        return total-ease(wall-w)
    return w-ramp/2


def observed_hands_type(Hands):
    """Add passive diagnostics to the existing, unchanged Wuji control client."""
    class ObservedHands(Hands):
        def __init__(self, target):
            self.extra = {s: {} for s in SIDES}
            super().__init__(target)
            from rclpy.qos import qos_profile_sensor_data
            for side in SIDES:
                self.node.create_subscription(self.message, '/wuji/'+side+'/command',
                    lambda msg, side=side: self.observe_command(side, msg), qos_profile_sensor_data)

        def receive(self, side, message):
            super().receive(side, message)
            if len(message.name) != 20 or set(message.name) != set(self.names[side]):
                return
            indices = [list(message.name).index(n) for n in self.names[side]]
            extra = dict(received_monotonic_s=time.monotonic(),
                         stamp_s=message.header.stamp.sec+message.header.stamp.nanosec/1e9)
            for key in ('velocity', 'effort'):
                values = getattr(message, key)
                extra[key] = ([float(values[i]) for i in indices]
                              if len(values) == 20 and all(math.isfinite(x) for x in values) else None)
            with self.mutex:
                self.extra[side]['joint_state'] = extra

        def observe_command(self, side, message):
            if len(message.name) != 20 or set(message.name) != set(self.names[side]) or len(message.position) != 20:
                return
            values = dict(zip(message.name, message.position))
            if not all(math.isfinite(v) for v in values.values()):
                return
            with self.mutex:
                self.extra[side]['control_output'] = dict(received_monotonic_s=time.monotonic(),
                    stamp_s=message.header.stamp.sec+message.header.stamp.nanosec/1e9,
                    q20=[float(values[n]) for n in self.names[side]])

        def diagnostics(self):
            with self.mutex:
                return json.loads(json.dumps(self.extra))
    return ObservedHands


def stop_hand_stack(stack):
    """Adapted from replay.py: reap the entire owned driver process group."""
    import os
    import signal
    for sig, seconds in ((signal.SIGINT, 10), (signal.SIGTERM, 3), (signal.SIGKILL, 2)):
        try:
            os.killpg(stack.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            stack.poll()
            try:
                os.killpg(stack.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(.1)
    stack.wait(timeout=2)


class HandSession:
    def __init__(self, hands, limits, out, result, tracking_log_only_stages=()):
        self.hands, self.limits, self.result = hands, limits, result
        if isinstance(tracking_log_only_stages, (str, bytes)):
            raise ValueError('tracking_log_only_stages must be a collection of stage names')
        self.tracking_log_only_stages = frozenset(tracking_log_only_stages)
        if not all(isinstance(stage, str) and stage for stage in self.tracking_log_only_stages):
            raise ValueError('tracking_log_only_stages must contain nonempty strings')
        self.last = np.array([hands.feedback()[s] for s in SIDES])
        validate_targets(self.last, limits)
        self.last_tick = time.monotonic()
        self.stop_file = out/'stop.request'
        self.log = (out/'hand_feedback.jsonl').open('w')

    def record_tracking_warning(self, row):
        event = {key: row.get(key) for key in (
            'monotonic_s', 'stage', 'source_time_s', 'tracking_error_rad',
            'per_side_error_rad', 'worst_side', 'worst_joint',
            'worst_joint_velocity_rad_s', 'worst_joint_effort')}
        report = self.result.setdefault('hand_tracking_warnings', dict(
            policy='log_only_for_expected_object_contact', limit_rad=TRACK_LIMIT,
            count=0, per_side_count={side: 0 for side in SIDES}, max_error_rad=0.))
        report['count'] += 1
        report['per_side_count'][row['worst_side']] += 1
        report.setdefault('first', event)
        report['last'] = event
        if row['tracking_error_rad'] >= report['max_error_rad']:
            report['max_error_rad'] = row['tracking_error_rad']
            report['worst'] = event

    def tick(self, desired, stage_name, source_time=None):
        now = time.monotonic()
        dt, self.last_tick = now-self.last_tick, now
        if self.stop_file.exists():
            raise RuntimeError('stop.request received')
        if not 0 < dt <= .1:
            raise RuntimeError('Hand supervision stalled: %.4f s' % dt)
        measured = np.array([self.hands.feedback()[s] for s in SIDES])
        err = abs(measured-self.last)
        side, joint = np.unravel_index(int(np.argmax(err)), err.shape)
        policy = ('log_only' if stage_name in self.tracking_log_only_stages else 'hard_stop')
        driver = self.hands.diagnostics()
        row = dict(monotonic_s=now, stage=stage_name, source_time_s=source_time,
                   measured_q20=measured.tolist(), previous_target_q20=self.last.tolist(),
                   desired_q20=desired.tolist(), tracking_error_rad=float(err.max()),
                   per_side_error_rad=err.max(axis=1).tolist(), worst_joint=hand_names(SIDES[side])[joint],
                   worst_side=SIDES[side], tracking_limit_rad=TRACK_LIMIT,
                   tracking_policy=policy, driver=driver)
        joint_state = driver.get(SIDES[side], {}).get('joint_state', {})
        for field, output in (('velocity', 'worst_joint_velocity_rad_s'),
                              ('effort', 'worst_joint_effort')):
            values = joint_state.get(field)
            if isinstance(values, list) and len(values) == 20:
                row[output] = values[joint]
        try:
            if (not np.isfinite(measured).all() or np.any(measured < self.limits[..., 0])
                    or np.any(measured > self.limits[..., 1])):
                raise ValueError('Measured Wuji joints exceed installed joint limits')
            if err.max() > TRACK_LIMIT:
                detail = ('Hand tracking error %.4f rad at %s exceeds 0.20 rad'
                          % (err.max(), row['worst_joint']))
                if policy == 'hard_stop':
                    raise HandTrackingError(detail)
                row['tracking_warning'] = detail
                self.record_tracking_warning(row)
            for s in SIDES:
                for suffix in ('/target/program', '/joint_states', '/command'):
                    if self.hands.node.count_publishers('/wuji/'+s+suffix) != 1:
                        raise RuntimeError('Missing/conflicting Wuji publisher: '+s+suffix)
            validate_targets(desired, self.limits)
            self.last += np.clip(desired-self.last, -TARGET_SPEED*dt, TARGET_SPEED*dt)
            self.hands.set_targets({s: self.last[i].tolist() for i, s in enumerate(SIDES)})
            self.result['hand_target_updates'] += 1
            row['commanded_q20'] = self.last.tolist()
            self.result['last_hand_feedback'] = row
            return measured
        except BaseException as exc:
            row['error'] = type(exc).__name__+': '+str(exc)
            self.result['last_hand_feedback'] = row
            raise
        finally:
            self.log.write(json.dumps(row, allow_nan=False)+'\n')
            self.log.flush()

    def prepare(self, goal, rate):
        initial = self.last.copy()
        duration = max(10., 1.875*float(np.max(abs(goal-initial)))/.3)
        self.result['hand_preparation'] = dict(duration_s=duration, tolerance_rad=.20, settle_allowance_s=8.)
        print('HANDS ONLY: first frame %.1fs; no arm connection' % duration, flush=True)
        began, settled = time.monotonic(), None
        while True:
            elapsed = time.monotonic()-began
            u = min(1., elapsed/duration)
            measured = self.tick(initial+(10*u**3-15*u**4+6*u**5)*(goal-initial), 'prepare')
            error = float(np.max(abs(measured-goal)))
            self.result['hand_preparation']['last_error_rad'] = error
            ready = elapsed >= duration and error < .20
            settled = (time.monotonic() if settled is None else settled) if ready else None
            if settled is not None and time.monotonic()-settled >= .3:
                return
            if elapsed > duration+8.:
                raise TimeoutError('Hands did not reach first frame; error %.4f rad' % error)
            time.sleep(1./rate)

    def stream(self, t, hands, T, rate):
        total = float(T[-1]); wall = total+min(1., total)
        began, settled = time.monotonic(), None
        print('HANDS ONLY: source replay %.1fs at supervision %.0f Hz' % (wall, rate), flush=True)
        while True:
            elapsed = time.monotonic()-began
            tau = phase(min(elapsed, wall), total)
            desired = np.array([np.interp(tau, T, hands[:, s, j])
                                for s in range(2) for j in range(20)]).reshape(2, 20)
            measured = self.tick(desired, 'replay', float(np.interp(tau, T, t)))
            ready = elapsed >= wall and np.max(abs(measured-hands[-1])) < .20
            settled = (time.monotonic() if settled is None else settled) if ready else None
            if settled is not None and time.monotonic()-settled >= .3:
                return
            if elapsed > wall+8.:
                raise TimeoutError('Hands did not reach final frame')
            time.sleep(1./rate)


def run(args, out, result):
    t, values, meta = load_hands(args.episode, args.sample_hz)
    t, (values,), window = select_window(t, [values], args.start_seconds, args.end_seconds)
    limits, provenance = limits_from_install()
    validate_targets(values, limits)
    T = hand_timeline(t, values, args.speed)
    result.update(source=dict(meta, selected_source_window_s=window), limits=provenance,
                  plan=dict(component='hands', arm_connection=False, ik_required=False,
                            source_seconds=float(t[-1]), motion_seconds=float(T[-1]+min(1., T[-1])),
                            supervision_rate_hz=args.rate, program_publish_rate_hz=50.,
                            first_frame_tolerance_rad=.20, tracking_limit_rad=TRACK_LIMIT))
    save(out/'plan.json', result['plan'])
    np.savez_compressed(out/'plan_hands_only.npz', t=t, hands=values, nominal_time_s=T)
    if not (args.check or args.execute):
        result['offline_validated'] = True
        return
    import rclpy
    from robot_home import Hands, process_commands
    hands = stack = stack_log = session = None
    rclpy.init()
    try:
        commands = process_commands()
        owners = ('wuji_teleop.switch', 'wuji-teleop-switch', 'tools/manus_teleop.sh', 'robot_home.py',
                  'axis_replay.py', 'replay.py', 'replay_hand.py', 'replay_hand_pinocchio.py', 'servo_test.py')
        if any(owner in c for c in commands for owner in owners):
            raise RuntimeError('Another controller is running; exit it before isolated testing')
        target = {s+'_hand': dict(zip(hand_names(s), values[0, i])) for i, s in enumerate(SIDES)}
        hands = observed_hands_type(Hands)(target)
        existing = any('wuji-hand-2-driver' in c or 'wuji-control' in c for c in commands)
        if args.execute and not existing:
            stack_log = (out/'hand_stack.log').open('w')
            stack = subprocess.Popen(['ros2', 'launch', str(ROOT/'tools/home_bimanual.launch.py')],
                                     stdout=stack_log, stderr=subprocess.STDOUT, start_new_session=True)
        measured = hands.feedback(timeout=20 if stack else 5)
        validate_targets(np.array([measured[s] for s in SIDES]), limits)
        hands.validate_owner()
        for s in SIDES:
            if hands.node.count_publishers('/wuji/'+s+'/command') != 1:
                raise RuntimeError('Expected one wuji_control output per hand')
        save(out/'start_state.json', dict(hands=measured, driver=hands.diagnostics()))
        result['live_preflight_passed'] = True
        if not args.execute:
            return
        hands.start()
        session = HandSession(hands, limits, out, result)
        session.prepare(values[0], args.rate)
        if not args.prepare_only:
            session.stream(t, values, T, args.rate)
        result['source_replayed'] = not args.prepare_only
    finally:
        errors = []
        if hands:
            if hands.touched:
                try:
                    hands.stop()
                    result['hands_stop'] = 'confirmed'
                except Exception as exc:
                    errors.append('hand stop: '+str(exc))
            try:
                hands.close()
            except Exception as exc:
                errors.append('hand close: '+str(exc))
        if session:
            session.log.close()
        if stack:
            try:
                stop_hand_stack(stack)
            except Exception as exc:
                errors.append('hand stack shutdown: '+str(exc))
        if stack_log:
            stack_log.close()
        rclpy.shutdown()
        if errors:
            result['cleanup_errors'] = errors
            raise RuntimeError('; '.join(errors))
