#!/usr/bin/env python3
"""Offline-only search for a reachable constant translation of frame zero."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "replay"), str(ROOT / "tools"), str(Path(__file__).resolve().parent)]

import replay as base
from episode import pose_error
from planner_cross import candidates
from transport import replay_backend


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.episode, allow_pickle=False) as archive:
        source = archive["eef"][0].copy()
    _, Model, _ = replay_backend()
    rows = []
    for back in (0.04, 0.06, 0.08, 0.10, 0.12, 0.14):
        for inward in (0.04, 0.06, 0.08, 0.10, 0.12, 0.14):
            target = source.copy()
            target[:, 0] -= back
            target[0, 1] -= inward
            target[1, 1] += inward
            best = None
            for seed_name, seed in candidates(ROOT, Model()):
                model = Model()
                try:
                    q = base.solve_eef(model, seed.copy(), target)
                    model.set_q(q)
                    position, rotation = pose_error(model.poses(), target)
                    clearance, pair = model.clearance()
                    margin = float(np.min(np.minimum(q - model.lower, model.upper - q)))
                    row = {
                        "back_m": back,
                        "inward_m": inward,
                        "seed": seed_name,
                        "position_error_m": list(map(float, position)),
                        "rotation_error_rad": list(map(float, rotation)),
                        "clearance_score_m": float(clearance),
                        "closest_bodies": pair,
                        "minimum_joint_margin_rad": margin,
                        "q16": q.tolist(),
                    }
                    score = max(position) * 100.0 + max(rotation) + max(0.0, 0.03 - margin) * 10.0
                    if best is None or score < best[0]:
                        best = (score, row)
                except Exception as exc:
                    best = best or (float("inf"), {
                        "back_m": back,
                        "inward_m": inward,
                        "seed": seed_name,
                        "error": type(exc).__name__ + ": " + str(exc),
                    })
            rows.append(best[1])
    feasible = [
        row for row in rows
        if max(row.get("position_error_m", [float("inf")])) <= base.IK_POSITION_TOL
        and max(row.get("rotation_error_rad", [float("inf")])) <= base.IK_ROTATION_TOL
        and row.get("clearance_score_m", -float("inf")) >= base.MODEL_CLEARANCE_M
        and row.get("minimum_joint_margin_rad", -float("inf")) >= 0.03
    ]
    feasible.sort(key=lambda row: (row["back_m"] ** 2 + row["inward_m"] ** 2,
                                   -row["minimum_joint_margin_rad"]))
    report = {"source": str(args.episode), "feasible_count": len(feasible),
              "selected": feasible[0] if feasible else None, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"feasible_count": len(feasible), "selected": report["selected"]}, indent=2))
    return 0 if feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())
