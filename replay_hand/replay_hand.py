#!/usr/bin/env python3
"""Cross-embodiment Franka/Sharpa -> Tron2/Wuji2 supervised ServoP replay.

Offline planning is the default. --check reads live state without commanding motion.
Only --execute may approach the first frame and stream commands.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPLAY = ROOT / "replay"
sys.path.insert(0, str(REPLAY))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import findings
import replay as base
import execution_cross as servo
from anchor_cross import anchor_episode
from episode_cross import load_cross_episode
from planner_cross import solve_seed_path
from runtime_cross import ARM_FEEDBACK_SPEED_LIMIT, stage_guard, stream_with_stop
from transport import replay_backend as backend


EEF_TRACK_POS_M = 0.04  # Matches the tested test_ik ServoP hard stop.
EEF_TRACK_ROT_RAD = 0.20
# Operator authorized addressing the 2026-11-10 0.180 rad preparation plateau.
# Match the existing runtime hand tracking ceiling; do not disable its guard.
HAND_FIRST_FRAME_TOL_RAD = 0.20
DEFAULT_EPISODE = ROOT / "replay_hand/data/episode395_board_aligned_v1.npz"


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def cross_loader(path: Path, hz: float, out: Path):
    t, eef, hands, meta = load_cross_episode(path, hz)
    _, Model, _ = backend()
    model = Model()
    placement = meta.get('placement_policy', 'legacy_per_arm_anchor')
    if placement == 'preserve_common_frame_v1':
        anchor_report = dict(policy=placement, absolute_camera_pose_preserved=True,
                             bimanual_relative_pose_preserved=True,
                             per_arm_reanchoring_applied=False)
        preferred = None
    elif placement == 'legacy_per_arm_anchor':
        eef, anchor_q, anchor_report = anchor_episode(model, eef)
        preferred = ('wanchi_relative_replay_anchor', anchor_q)
    else:
        raise ValueError('Unknown placement policy: ' + str(placement))
    seed_path, seed_report = solve_seed_path(
        base, model, ROOT, t, eef, out,
        preferred=preferred,
    )
    meta["cross_anchor"] = anchor_report
    meta["cross_seed"] = seed_report
    return t, eef, hands, seed_path, meta


def run(args, out: Path, result: dict) -> None:
    if args.execute:
        with np.load(args.episode, allow_pickle=False) as archive:
            alignment_meta = json.loads(str(archive['metadata_json']))
        if alignment_meta.get('placement_policy') == 'preserve_common_frame_v1' and not alignment_meta.get('execution_ready'):
            raise RuntimeError('Board-aligned conversion has not passed IK and hand-inclusive collision validation')
    original_loader = servo.load_episode
    original_position = servo.EEF_TRACK_POS
    original_rotation = servo.EEF_TRACK_ROT
    original_hand_first_frame = base.HAND_FIRST_FRAME_TOL
    original_guard = base.stage_guard
    original_stream = servo.stream
    result['arm_feedback_speed_limit_rad_s'] = ARM_FEEDBACK_SPEED_LIMIT
    try:
        servo.load_episode = lambda path, hz: cross_loader(path, hz, out)
        servo.EEF_TRACK_POS = EEF_TRACK_POS_M
        servo.EEF_TRACK_ROT = EEF_TRACK_ROT_RAD
        base.HAND_FIRST_FRAME_TOL = HAND_FIRST_FRAME_TOL_RAD
        base.stage_guard = stage_guard
        servo.stream = lambda session, T, t, qs, rate: stream_with_stop(
            original_stream, session, T, t, qs, rate, result)
        servo.run(args, out, result)
    finally:
        servo.load_episode = original_loader
        servo.EEF_TRACK_POS = original_position
        servo.EEF_TRACK_ROT = original_rotation
        base.HAND_FIRST_FRAME_TOL = original_hand_first_frame
        base.stage_guard = original_guard
        servo.stream = original_stream
    plan_path = out / "plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text())
        plan.update(
            entry="replay_hand",
            first_frame_arm_source="cross-embodiment IK seed; measured-start certified ServoJ approach",
            execution_basis=(
                "pengyang replay_servop planning/safety plus test_ik ServoP 4 cm EEF tracking stop"
            ),
            hand_first_frame_tolerance_rad=HAND_FIRST_FRAME_TOL_RAD,
            arm_feedback_speed_limit_rad_s=ARM_FEEDBACK_SPEED_LIMIT,
            hand_first_frame_tolerance_basis=(
                "Operator-authorized first-frame tolerance 0.20 rad after 2026-11-10 "
                "0.180 rad plateau; runtime 0.20 rad hand tracking guard unchanged"
            ),
            right_hand_eef_rotation_independently_validated=False,
        )
        save(plan_path, plan)
        result["plan"] = plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="offline plan plus read-only live preflight")
    mode.add_argument("--execute", action="store_true", help="approach and execute on hardware")
    parser.add_argument("--speed", type=float, default=0.10, help="maximum source-time speed")
    parser.add_argument("--rate", type=float, default=300.0, help="ServoP command rate")
    parser.add_argument("--sample-hz", type=float, default=50.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--end-seconds", type=float)
    parser.add_argument("--prepare-only", action="store_true",
                        help="with --execute: prepare hands, ServoJ approach and ServoP takeover; no source replay")
    parser.add_argument("--permissive", action="store_true")
    args = parser.parse_args()
    if not 0 < args.speed <= 0.25:
        parser.error("replay_hand --speed must be within (0, 0.25]")
    if not 50 <= args.rate <= 500:
        parser.error("--rate must be within [50, 500]")
    out = ROOT / "replay_hand/runs" / ("replay_hand_" + time.strftime("%Y%m%d_%H%M%S") + "_" + str(os.getpid()))
    out.mkdir(parents=True)
    result = {
        "completed": False,
        "execute_requested": args.execute,
        "backend": "replay_hand_supervised_servop",
        "permissive": args.permissive,
        "motion_commands_sent": 0,
    }

    def interrupt(signum, frame):
        raise KeyboardInterrupt()

    old = signal.signal(signal.SIGTERM, interrupt)
    lock = open("/home/guest/wanchi/replay_tip_align_test_1030/.hands_replay.lock", "a")
    if args.permissive:
        findings.begin(out / "violations.jsonl")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, out, result)
        result["completed"] = True
    except (Exception, KeyboardInterrupt) as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
        print("REPLAY_HAND FAILED:", result["error"], flush=True)
    finally:
        lock.close()
        signal.signal(signal.SIGTERM, old)
        record = findings.active()
        if record is not None:
            result["findings"] = record.summary()
            findings.end()
        save(out / "result.json", result)
        print("REPORT", out / "result.json", flush=True)
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
