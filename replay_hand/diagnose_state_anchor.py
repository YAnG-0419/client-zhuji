#!/usr/bin/env python3
"""Offline-certify episode395 relative motion from a saved Tron2 state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "replay"), str(ROOT / "tools"), str(Path(__file__).resolve().parent)]

import replay as base
from anchor_cross import reanchor_eef
from episode_cross import load_cross_episode
from planner_cross import solve_seed_path
from transport import replay_backend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    state = json.loads(args.state.read_text())
    q = np.asarray(state["q16"], dtype=float)
    if q.shape != (16,) or not np.isfinite(q).all():
        raise ValueError("saved state must contain finite q16")
    t, eef, _, meta = load_cross_episode(args.episode, 50.0)
    _, Model, _ = replay_backend()
    model = Model()
    if np.any(q < model.lower) or np.any(q > model.upper):
        raise ValueError("saved state exceeds executable limits")
    model.set_q(q)
    eef, anchor = reanchor_eef(eef, model.poses().copy())
    path, seed = solve_seed_path(
        base, model, ROOT, t, eef, args.output,
        preferred=("saved_live_state", q),
    )
    collision = base.certify_path(model, path)
    report = {
        "completed": True,
        "planning_only": True,
        "motion_commands_sent": 0,
        "state": str(args.state),
        "source": meta,
        "anchor": anchor,
        "seed": seed,
        "dense_collision": collision,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "completed": True,
        "frames": len(path),
        "max_position_error_m": seed["max_position_error_m"],
        "max_rotation_error_rad": seed["max_rotation_error_rad"],
        "max_joint_step_rad": seed["max_joint_step_rad"],
        "dense_collision": collision,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
