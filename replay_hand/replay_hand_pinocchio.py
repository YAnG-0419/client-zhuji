#!/usr/bin/env python3
"""Parallel entry: Pinocchio EEF IK and ServoJ arms with synchronized Wuji hands."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT/'replay'), str(ROOT/'tools')]

import replay as base
import execution_pinocchio as execution
from anchor_cross import anchor_episode
from episode_cross import load_cross_episode
from planner_cross import solve_seed_path
from pinocchio_cross import PinocchioSolver
from runtime_cross import stage_guard, ARM_FEEDBACK_SPEED_LIMIT
from transport import replay_backend


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def run(args, out, result):
    solver = PinocchioSolver(ROOT)
    original_solver, original_guard = base.solve_eef, base.stage_guard
    original_loader, original_hand = execution.load_episode, base.HAND_FIRST_FRAME_TOL

    def loader(path, hz):
        t, eef, hands, meta = load_cross_episode(path, hz)
        _, Model, _ = replay_backend()
        model = Model()
        eef, anchor, anchor_report = anchor_episode(model, eef)
        print('PINOCCHIO IK: source EEF -> direct ServoJ joint targets', flush=True)
        qs, report = solve_seed_path(base, model, ROOT, t, eef, out,
                                    preferred=('measured_anchor_pinocchio', anchor))
        meta.update(cross_anchor=anchor_report, cross_seed=report,
                    ik_backend='Pinocchio', command_backend='ServoJ',
                    pinocchio=solver.report())
        return t, eef, hands, qs, meta

    try:
        base.solve_eef, base.stage_guard = solver.solve, stage_guard
        base.HAND_FIRST_FRAME_TOL = .15
        execution.load_episode = loader
        execution.run(args, out, result)
    finally:
        base.solve_eef, base.stage_guard = original_solver, original_guard
        execution.load_episode, base.HAND_FIRST_FRAME_TOL = original_loader, original_hand
        result['pinocchio'] = solver.report()
        save(out/'pinocchio_ik.json', result['pinocchio'])
    if (out/'plan.json').is_file():
        plan = json.loads((out/'plan.json').read_text())
        plan.update(entry='replay_hand_pinocchio', ik_backend='Pinocchio',
                    command_backend='ServoJ', servop_commands_allowed=False,
                    pinocchio=solver.report(), arm_feedback_speed_limit_rad_s=ARM_FEEDBACK_SPEED_LIMIT)
        save(out/'plan.json', plan)
        result['plan'] = plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode', type=Path, default=ROOT/'replay_hand/data/episode395_replay_hand.npz')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true', help='read-only live preflight after offline planning')
    modes.add_argument('--execute', action='store_true', help='execute ServoJ arms and Wuji hands')
    parser.add_argument('--speed', type=float, default=.10)
    parser.add_argument('--rate', type=float, default=300.)
    parser.add_argument('--sample-hz', type=float, default=50.)
    parser.add_argument('--start-seconds', type=float, default=0.)
    parser.add_argument('--end-seconds', type=float)
    parser.add_argument('--prepare-only', action='store_true', help='only prepare hands and align arms via ServoJ')
    args = parser.parse_args()
    if not 0 < args.speed <= .25:
        parser.error('--speed must be within (0, 0.25]')
    if not 50 <= args.rate <= 500:
        parser.error('--rate must be within [50, 500]')
    out = ROOT/'replay_hand/runs'/('pinocchio_servoj_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid()))
    out.mkdir(parents=True)
    result = dict(completed=False, execute_requested=args.execute,
                  backend='pinocchio_ik_servoj_wuji', motion_commands_sent=0,
                  arm_feedback_speed_limit_rad_s=ARM_FEEDBACK_SPEED_LIMIT)
    def interrupt(signum, frame):
        raise KeyboardInterrupt()
    previous = signal.signal(signal.SIGTERM, interrupt)
    lock = open('/home/guest/wanchi/replay_tip_align_test_1030/.hands_replay.lock', 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, out, result)
        result['completed'] = True
    except (Exception, KeyboardInterrupt) as exc:
        result['error'] = type(exc).__name__+': '+str(exc)
        print('PINOCCHIO REPLAY FAILED:', result['error'], flush=True)
    finally:
        lock.close()
        signal.signal(signal.SIGTERM, previous)
        save(out/'result.json', result)
        print('REPORT', out/'result.json', flush=True)
    return 0 if result['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
