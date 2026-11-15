"""CLI and exclusive ownership for component-isolated replay entries."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def select_window(t, arrays, start, end):
    end = float(t[-1]) if end is None else end
    if not np.isfinite([start, end]).all() or not 0 <= start < end <= t[-1]:
        raise ValueError('Requested source window must lie inside recording')
    mask = (t >= start) & (t <= end)
    if np.count_nonzero(mask) < 2:
        raise ValueError('Source window must contain at least two samples')
    origin = float(t[mask][0])
    return t[mask]-origin, [a[mask] for a in arrays], [origin, float(t[mask][-1])]


def main(component, runner):
    parser = argparse.ArgumentParser(description='Isolated '+component+' replay; default OFFLINE, no device commands')
    parser.add_argument('--episode', type=Path, default=ROOT/'replay_hand/data/episode395_replay_hand.npz')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='read-only preflight, no drivers started')
    mode.add_argument('--execute', action='store_true', help='enable/control ONLY the selected component')
    parser.add_argument('--speed', type=float, default=.05)
    parser.add_argument('--rate', type=float, default=300. if component == 'arms' else 50.)
    parser.add_argument('--sample-hz', type=float, default=50.)
    parser.add_argument('--start-seconds', type=float, default=0.)
    parser.add_argument('--end-seconds', type=float)
    preparation = parser.add_mutually_exclusive_group()
    preparation.add_argument('--prepare-only', action='store_true', help='move selected component to frame zero; no source replay')
    if component == 'arms':
        preparation.add_argument('--hold-seconds', type=float, help='ServoP hold measured arm pose inside workspace; no episode load or IK')
        parser.add_argument('--ready-arm', choices=['left', 'right', 'both'],
                            help='with --hold-seconds: certified MoveJ to test_ik ready pose for selected arm before holding')
        parser.add_argument('--record-accuracy-only', action='store_true',
                            help='bounded hold or <=0.5s source experiment: log accuracy errors; retain device protections')
    args = parser.parse_args()
    if not np.isfinite(args.speed) or not 0 < args.speed <= .25:
        parser.error('--speed must be finite and within (0, 0.25]')
    low, high = (50., 500.) if component == 'arms' else (20., 50.)
    if not np.isfinite(args.rate) or not low <= args.rate <= high:
        parser.error('--rate must be within [%g, %g]' % (low, high))
    if not np.isfinite(args.sample_hz) or not 5 <= args.sample_hz <= 100:
        parser.error('--sample-hz must be within [5, 100]')
    hold = getattr(args, 'hold_seconds', None)
    if getattr(args, 'ready_arm', None) is not None and hold is None:
        parser.error('--ready-arm requires --hold-seconds')
    if getattr(args, 'record_accuracy_only', False):
        if hold is not None and not .2 <= hold <= 2.:
            parser.error('--record-accuracy-only hold must be within [0.2, 2]')
        if hold is None and (args.end_seconds is None or not np.isfinite([args.start_seconds, args.end_seconds]).all()
                             or not 0 < args.end_seconds-args.start_seconds <= .500001):
            parser.error('--record-accuracy-only trajectory requires an explicit source window of at most0.5s')
    if hold is not None and (not np.isfinite(hold) or not .2 <= hold <= 10.):
        parser.error('--hold-seconds must be within [0.2, 10]; it sends real ServoP commands with --execute')
    entry = 'replay_'+('arm' if component == 'arms' else 'hands')+'_only'
    out = ROOT/'replay_hand/runs'/(entry+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
    out.mkdir(parents=True)
    result = dict(completed=False, component=component, execute_requested=args.execute,
                  arm_motion_commands_sent=0, hand_target_updates=0,
                  isolation='No connection, enable, command or stop of the other component')
    def interrupt(signum, frame):
        raise KeyboardInterrupt()
    old = signal.signal(signal.SIGTERM, interrupt)
    lock = None
    try:
        # Shared with combined replay: isolation tests are deliberately sequential.
        # Offline planning needs neither this robot-host directory nor hardware.
        if args.check or args.execute:
            lock = open('/home/guest/wanchi/replay_tip_align_test_1030/.hands_replay.lock', 'a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner(args, out, result)
        result['completed'] = True
    except (Exception, KeyboardInterrupt) as exc:
        result['error'] = type(exc).__name__+': '+str(exc)
        print(entry.upper()+' FAILED:', result['error'], flush=True)
    finally:
        if lock is not None:
            lock.close()
        signal.signal(signal.SIGTERM, old)
        save(out/'result.json', result)
        print('REPORT', out/'result.json', flush=True)
    return 0 if result['completed'] else 1
