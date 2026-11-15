#!/usr/bin/env python3
"""Full dual-arm/dual-hand replay. Default offline; --execute moves hardware."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT/'replay'), str(ROOT/'tools'), str(ROOT)]
from isolated_cli import save
from full_sync import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=Path, default=ROOT/'replay_hand/data/episode395_board_placed_rx10.npz')
    p.add_argument('--speed', type=float, default=1.)
    p.add_argument('--rate', type=float, default=300.)
    p.add_argument('--prepare-seconds', type=float, default=30.,
                   help='Minimum MoveJ duration per certified preparation leg (20..120 seconds)')
    recovery = p.add_argument_group('IK recovery')
    recovery.add_argument('--ik-recover', action='store_true', help='Validate and bridge short IK gaps before replay')
    recovery.add_argument('--recover-position-mm', type=float, default=20., help='Maximum position error in a bridge')
    recovery.add_argument('--recover-rotation-deg', type=float, default=3., help='Maximum rotation error in a bridge')
    recovery.add_argument('--recover-max-gap-s', type=float, default=1.5, help='Maximum source duration between bridge anchors')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not np.isfinite(args.speed) or not 0 < args.speed <= 1:
        p.error('--speed must be within (0,1]')
    if not np.isfinite(args.rate) or not 50 <= args.rate <= 500:
        p.error('--rate must be within [50,500]')
    if not np.isfinite(args.prepare_seconds) or not 20 <= args.prepare_seconds <= 120:
        p.error('--prepare-seconds must be within [20,120]')
    if args.ik_recover and (not np.isfinite(args.recover_position_mm) or args.recover_position_mm <= 0
                            or not 0 < args.recover_rotation_deg <= 180
                            or not np.isfinite(args.recover_max_gap_s) or args.recover_max_gap_s <= 0):
        p.error('Recovery needs positive finite limits and a rotation bound within (0, 180]')
    out = ROOT/'replay_hand/runs'/('replay_full_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
    out.mkdir(parents=True)
    result = dict(completed=False, execute_requested=args.execute, arm_motion_commands_sent=0,
                  hand_target_updates=0, requested_speed=args.speed)
    def interrupt(signum, frame):raise KeyboardInterrupt()
    old = signal.signal(signal.SIGTERM, interrupt)
    lock = None
    try:
        if args.check or args.execute:
            lock = open('/home/guest/wanchi/replay_tip_align_test_1030/.hands_replay.lock', 'a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, out, result);result['completed'] = True
    except (Exception, KeyboardInterrupt) as exc:
        result['error'] = type(exc).__name__+': '+str(exc)
        print('REPLAY_FULL FAILED:', result['error'], flush=True)
    finally:
        if lock:lock.close()
        signal.signal(signal.SIGTERM, old)
        save(out/'result.json', result)
        print('REPORT', out/'result.json', flush=True)
    return 0 if result['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
