#!/usr/bin/env python3
"""Offline benchmark for both streaming Wuji root modes."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from hand_retarget_wuji2 import MODES, Wuji2StreamingRetargeter


def percentile(values):
    values = np.asarray(values, dtype=float)
    return {"samples": len(values), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)), "max_ms": float(values.max()),
            "total_ms": float(values.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 8 <= args.frames <= 128:
        parser.error("frames must be within [8, 128]")
    report = {"schema": "wuji2_streaming_retarget_benchmark_v1",
              "hardware_commands_sent": 0, "execution_ready": False, "modes": {}}
    for mode in MODES:
        mode_report = {}
        for stride in (1, 4):
            side_times = []
            accepted = 0
            began = time.perf_counter()
            for side_index, side in enumerate(("left", "right")):
                solver = Wuji2StreamingRetargeter(side, mode, thresholds={
                    "max_joint_step_rad": 1.0,
                    "max_root_translation_step_m": 1.0,
                    "max_root_rotation_step_rad": 2.0,
                })
                q0 = np.clip(np.linspace(-0.15, 0.25, 20), solver.hand.lo + 0.05,
                             solver.hand.hi - 0.05)
                root = np.eye(4)
                root[:3, :3] = Rotation.from_euler("xyz", [0.15, -0.1, 0.25 * (1 - 2 * side_index)]).as_matrix()
                root[:3, 3] = [0.4, 0.18 * (1 - 2 * side_index), 0.65]
                solver.begin_chunk(q0, root)
                for frame in range(0, args.frames, stride):
                    q = np.clip(q0 + 0.08 * np.sin(frame / 12.0),
                                solver.hand.lo + 0.02, solver.hand.hi - 0.02)
                    points = solver.hand.fk(q) @ root[:3, :3].T + root[:3, 3]
                    answer = solver.retarget_frame(points, 1_000_000_000 + frame * 8_333_333,
                                                   normalize=False)
                    side_times.append(answer["diagnostic"]["elapsed_ms"])
                    accepted += int(answer["accepted"])
            key = "all_32_steps" if stride == 1 else "every_4th_step"
            mode_report[key] = dict(percentile(side_times), wall_ms=1000 * (time.perf_counter() - began),
                                    accepted=accepted, attempted=len(side_times))
        report["modes"][mode] = mode_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
