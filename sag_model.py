#!/usr/bin/env python3
"""Measure the arms' static gravity sag from runs that already happened.

ServoP streams EEF poses and the controller runs its own joint loop, so under
the hand's weight the arm settles a centimetre or two below the commanded pose.
The offset is static (measured spread < 0.5 mm) but grows as the arm extends,
so a single constant is not enough: this fits  sag = A @ [x, y, z, 1]  per side
in base_Link, from the commanded-vs-measured pairs the audit trail already
records.  No new robot motion is needed - replay runs work as well as live ones.

    python sag_model.py                       # fit from client/runs, print a report
    python sag_model.py --out sag_model.json  # ... and write the model
    python sag_model.py --check sag_model.json  # how well it predicts held-out runs

The live client applies it with --sag-comp on (default off).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

SIDES = ("left", "right")
MAX_COMPENSATION_M = 0.035          # never push the command further than this


def features(eef_xyz, arm_q):
    """Regressors for one side: the commanded position, the arm joints, and a bias.

    Droop is a gravity torque divided by the joint stiffness, so it is a function
    of the arm configuration; the commanded position alone explains only half of
    it (measured 2026-11-17 over 2269 samples: |sag| 12.0 mm -> 5.7 mm residual with
    position only, -> 3.5 mm once the joints are included).
    """
    return np.concatenate([np.asarray(eef_xyz, float).reshape(-1, 3),
                           np.asarray(arm_q, float).reshape(-1, 7),
                           np.ones((np.asarray(eef_xyz).reshape(-1, 3).shape[0], 1))], axis=1)


def samples(run_dirs, *, settled_tail_s=0.25):
    """Commanded pose, arm joints, and the sag measured against the command.

    Only the tail of each chunk is used: that is where the arm has stopped and
    the offset is the static droop rather than tracking lag.
    """
    target, joints, sag = [], [], []
    for run in run_dirs:
        for path in sorted((run / "audit_executions").glob("exec*.npz")):
            try:
                with np.load(path, allow_pickle=False) as data:
                    elapsed = data["feedback_elapsed_s"]
                    commanded = data["feedback_eef_target_xyzw"]
                    measured = data["feedback_eef_measured_xyzw"]
                    q16 = data["feedback_q16_measured_rad"]
            except (OSError, ValueError, KeyError):
                continue
            if len(elapsed) < 3:
                continue
            keep = elapsed >= elapsed[-1] - settled_tail_s
            if keep.sum() < 2:
                keep = np.zeros(len(elapsed), bool); keep[-2:] = True
            target.append(commanded[keep, :, :3])
            joints.append(np.stack([q16[keep, :7], q16[keep, 7:14]], axis=1))
            sag.append(measured[keep, :, :3] - commanded[keep, :, :3])
    if not target:
        raise SystemExit("no executed chunks found; run the client with --execute first")
    return np.concatenate(target), np.concatenate(joints), np.concatenate(sag)


def fit(target, joints, sag):
    """Least squares  sag = A @ features  per side, plus residual stats."""
    model = {"max_compensation_m": MAX_COMPENSATION_M, "sides": {}}
    for index, side in enumerate(SIDES):
        design = features(target[:, index], joints[:, index])
        coefficients, *_ = np.linalg.lstsq(design, sag[:, index], rcond=None)
        predicted = design @ coefficients
        residual = np.linalg.norm(sag[:, index] - predicted, axis=1)
        raw = np.linalg.norm(sag[:, index], axis=1)
        model["sides"][side] = {
            "coefficients": coefficients.T.tolist(),          # 3 x 11, metres
            "samples": int(len(design)),
            "mean_sag_mm": (1000 * sag[:, index].mean(axis=0)).tolist(),
            "raw_norm_mm": {"mean": float(1000 * raw.mean()), "p95": float(1000 * np.percentile(raw, 95))},
            "residual_mm": {"mean": float(1000 * residual.mean()),
                            "p95": float(1000 * np.percentile(residual, 95)),
                            "max": float(1000 * residual.max())},
        }
    return model


def predict(model, eef_xyzw, arm_q16):
    """Sag (2,3) in metres for a commanded (2,7) pose at q16, clipped to the cap."""
    cap = float(model.get("max_compensation_m", MAX_COMPENSATION_M))
    arm_q16 = np.asarray(arm_q16, float)
    out = np.zeros((2, 3))
    for index, side in enumerate(SIDES):
        coefficients = np.asarray(model["sides"][side]["coefficients"], float)
        row = features(np.asarray(eef_xyzw)[index, :3],
                       arm_q16[7 * index:7 * (index + 1)])[0]
        out[index] = coefficients @ row
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    scale = np.where(norm > cap, cap / np.maximum(norm, 1e-9), 1.0)
    return out * scale


def compensate(model, eef_xyzw, arm_q16):
    """The pose to stream so the arm settles on eef_xyzw: command = target - sag."""
    out = np.array(eef_xyzw, dtype=float, copy=True)
    out[:, :3] -= predict(model, eef_xyzw, arm_q16)
    return out


def report(model) -> str:
    lines = []
    for side in SIDES:
        entry = model["sides"][side]
        lines.append("%-5s n=%-5d  sag mean [%+6.1f %+6.1f %+6.1f] mm  |sag| mean %5.1f p95 %5.1f"
                     % (side, entry["samples"], *entry["mean_sag_mm"],
                        entry["raw_norm_mm"]["mean"], entry["raw_norm_mm"]["p95"]))
        lines.append("      residual after fit: mean %.1f  p95 %.1f  max %.1f mm"
                     % (entry["residual_mm"]["mean"], entry["residual_mm"]["p95"],
                        entry["residual_mm"]["max"]))
    return "\n".join(lines)


def main() -> int:
    client = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", type=Path,
                        help="run directories; default: every live-* under client/runs")
    parser.add_argument("--out", type=Path, help="write the fitted model here")
    parser.add_argument("--check", type=Path, help="score an existing model on these runs")
    args = parser.parse_args()

    run_dirs = args.runs or sorted(p for p in (client / "runs").glob("live-*") if p.is_dir())
    if not run_dirs:
        print("no runs found", file=sys.stderr)
        return 1
    target, joints, sag = samples(run_dirs)
    print("%d samples from %d runs" % (len(target), len(run_dirs)))

    if args.check:
        model = json.loads(args.check.read_text())
        for index, side in enumerate(SIDES):
            predicted = np.array([predict(model, np.concatenate(
                [target[row], np.zeros((2, 4))], axis=1),
                joints[row].reshape(-1))[index] for row in range(len(target))])
            residual = np.linalg.norm(sag[:, index] - predicted, axis=1)
            print("%-5s held-out residual: mean %.1f  p95 %.1f  max %.1f mm"
                  % (side, 1000 * residual.mean(), 1000 * np.percentile(residual, 95),
                     1000 * residual.max()))
        return 0

    model = fit(target, joints, sag)
    print(report(model))
    if args.out:
        args.out.write_text(json.dumps(model, indent=2) + "\n")
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
