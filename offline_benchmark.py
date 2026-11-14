#!/usr/bin/env python3
"""Offline timing of saved replay calculations; all socket connections are blocked.

No robot/client/Hands objects are constructed. Does not run full_sync.run or replay
entry points. This measures CPU-side calculation under current host load, not robot
control latency, scheduling guarantees, or equivalent Cartesian paths.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time


def deny_connection(*args, **kwargs):
    raise RuntimeError('Offline benchmark forbids network/device connections')


socket.socket.connect = deny_connection
socket.socket.connect_ex = deny_connection
socket.create_connection = deny_connection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/home/guest/pengyang'))
    parser.add_argument('--run', type=Path, required=True,
                        help='Existing completed-planning run with plan_full.npz and plan.npz')
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--ik-samples', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 10 <= args.samples <= 1000 or not 0 <= args.ik_samples <= 8:
        parser.error('samples must be 10..1000; ik-samples must be 0..8')
    root = args.root.resolve()
    run = args.run.resolve()
    output = args.output or root/'client'/('benchmark_'+time.strftime('%Y%m%d_%H%M%S')+'.json')
    output = output.resolve()
    if root/'client' not in output.parents:
        parser.error('Output must be inside pengyang/client')

    # A local live replay must not compete with this benchmark for CPU.
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            words = proc.read_bytes().decode(errors='replace').split('\0')
        except (OSError, ProcessLookupError):
            continue
        if '--execute' in words and any(
                Path(word).name in {'replay_full.py', 'replay_arm_only.py', 'servo_test.py',
                                    'replay_hand.py', 'replay_hand_pinocchio.py', 'robot_home.py'}
                for word in words):
            raise RuntimeError('A live replay is running; benchmark was not started')

    sys.path[:0] = [str(root/'replay_hand'), str(root/'replay'), str(root/'tools')]
    import numpy as np
    from scipy.spatial.transform import Rotation, Slerp
    import replay as base
    from transport import replay_backend
    from replay_servop import interp_rows, QuatTrack
    from movej_prepare import check_servop_box
    from episode import pose_error
    from hand_body_clearance import HandBodyClearance

    with np.load(run/'plan_full.npz', allow_pickle=False) as archive:
        t = archive['t'].copy()
        qs = archive['q16'].copy()
        hands = archive['hands'].copy()
    with np.load(run/'plan.npz', allow_pickle=False) as archive:
        np.testing.assert_array_equal(t, archive['t'])
        eef = archive['eef'].copy()

    report = dict(schema='offline_replay_benchmark_v1', run=str(run),
                  timestamp_local=time.strftime('%Y-%m-%d %H:%M:%S %z'),
                  network_connections_allowed=False, hardware_commands_sent=0,
                  source_frames=len(t), load_average=os.getloadavg(),
                  note='Current-load microbenchmark; excludes ROS, live feedback, disk logs, '
                       'WebSocket transport and real-time scheduling. Direct EEF interpolation '
                       'changes the target path and is not an execution-equivalence claim.',
                  timings={}, ik_cases=[])
    _, Model, _ = replay_backend()
    began = time.perf_counter()
    model = Model()
    report['arm_model_initialization_ms'] = 1000*(time.perf_counter()-began)
    began = time.perf_counter()
    hand_model = HandBodyClearance(root/'replay_robot/models/tron2_wuji_board_v1_20261110/robot.urdf')
    report['hand_model_initialization_ms'] = 1000*(time.perf_counter()-began)
    slerps = [Slerp(t, Rotation.from_quat(eef[:, side, 3:])) for side in (0, 1)]
    qtrack, direct_track = QuatTrack(), QuatTrack()
    phases = np.linspace(float(t[0]), float(t[-1]), args.samples)

    def encode(target, tracker):
        check_servop_box(target)
        left, right = tracker(target)
        return json.dumps(dict(left_pos=left.tolist(), right_pos=right.tolist()), allow_nan=False)

    def fk_payload(phase):
        model.set_q(interp_rows(t, qs, phase))
        return encode(model.poses(), qtrack)

    def direct_payload(phase):
        target = interp_rows(t, eef, phase).copy()
        for side in (0, 1):
            target[side, 3:] = slerps[side](phase).as_quat()
        return encode(target, direct_track)

    def arm_clearance(phase):
        model.set_q(interp_rows(t, qs, phase))
        return model.clearance()

    def hand_clearance(phase):
        hand_model.set_q(interp_rows(t, qs, phase), interp_rows(t, hands, phase))
        return hand_model.clearance()

    def measure(name, function, count):
        measured = []
        # Warm-up remains an offline model operation, never a motion command.
        function(phases[0])
        for phase in np.linspace(float(t[0]), float(t[-1]), count):
            began = time.perf_counter()
            function(phase)
            measured.append(1000*(time.perf_counter()-began))
        values = np.asarray(measured)
        result = dict(samples=len(values), mean_ms=float(values.mean()),
                      p50_ms=float(np.percentile(values, 50)), p95_ms=float(np.percentile(values, 95)),
                      p99_ms=float(np.percentile(values, 99)), max_ms=float(values.max()))
        report['timings'][name] = result
        print(name, json.dumps(result), flush=True)

    measure('joint_interpolation_fk_workspace_quaternion_json', fk_payload, args.samples)
    measure('eef_linear_slerp_workspace_quaternion_json', direct_payload, args.samples)
    measure('arm_model_update_and_clearance', arm_clearance, min(50, args.samples))
    measure('hand_model_update_and_clearance', hand_clearance, min(50, args.samples))
    if args.ik_samples:
        indices = np.linspace(1, len(t)-1, args.ik_samples).astype(int)
        for index in indices:
            case = dict(frame=int(index), source_time_s=float(t[index]), seed='previous saved planned joints')
            began = time.perf_counter()
            try:
                solved = base.solve_eef(model, qs[index-1].copy(), eef[index])
                case['elapsed_ms'] = 1000*(time.perf_counter()-began)
                model.set_q(solved)
                ep, er = pose_error(model.poses(), eef[index])
                case.update(position_error_m=float(max(ep)), rotation_error_rad=float(max(er)),
                            passes_residual_gate=bool(max(ep) <= base.IK_POSITION_TOL and
                                                     max(er) <= base.IK_ROTATION_TOL))
            except Exception as exc:
                case.update(elapsed_ms=1000*(time.perf_counter()-began), error=repr(exc))
            report['ik_cases'].append(case)
            print('ik_case', json.dumps(case), flush=True)
    paths = ['replay_hand/full_sync.py', 'replay_hand/accuracy_replay.py',
             'replay_hand/planner_cross.py', 'replay/replay.py', 'replay/replay_thresholds.py']
    report['source_sha256'] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('REPORT', output, flush=True)


if __name__ == '__main__':
    main()
