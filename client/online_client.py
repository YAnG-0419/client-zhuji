#!/usr/bin/env python3
"""Compile a model/recorded camera chunk for the online client. Offline only.

This CLI deliberately has no --execute option and never imports the robot or
Wuji hardware transports.  It verifies the deployment mapping, applies the
configured tool offset exactly once, resamples an accepted chunk, and emits an
auditable NPZ/report for the future online runtime.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from online_core import DeploymentMapping, convert_camera_npz, resample_chunk


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> int:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).resolve().with_name("deployment_config.json"))
    parser.add_argument("--chunk", type=Path, required=True)
    parser.add_argument("--camera-mapping", required=True,
                        choices=("live_zhuji_camera", "board_transfer"),
                        help="Use live_zhuji_camera for online zhuji observations; old replay/Orbbec chunks require board_transfer")
    parser.add_argument("--origin-ns", type=int,
                        help="Replace the recorded clock with an absolute monotonic planning origin")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    config = json.loads(args.config.read_text())
    output = args.output
    if output is None:
        output = root / "client/runs" / ("compile_" + time.strftime("%Y%m%d_%H%M%S"))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema": "pengyang_online_eef_compile_v1", "completed": False,
              "execution_ready": False, "hardware_commands_sent": 0,
              "input": str(args.chunk.resolve()), "config": str(args.config.resolve())}
    try:
        mapping = DeploymentMapping.load(root, args.config, args.camera_mapping)
        chunk = convert_camera_npz(args.chunk, mapping, origin_ns=args.origin_ns)
        period_ns = int(round(float(config["timing"]["planning_period_s"]) * 1e9))
        accepted = resample_chunk(chunk, period_ns)
        np.savez_compressed(output / "accepted_chunk.npz", times_ns=accepted.times_ns,
                            eef=accepted.eef, hands=accepted.hands,
                            metadata_json=np.asarray(json.dumps(accepted.metadata)))
        result.update(completed=True, source_frames=len(chunk.times_ns),
                      accepted_frames=len(accepted.times_ns),
                      duration_s=(int(accepted.times_ns[-1]) - int(accepted.times_ns[0])) / 1e9,
                      operator_adapter_length_m=mapping.operator_adapter_length_m,
                      calibrated_coordinate_origin_distance_m=mapping.coordinate_origin_distance_m,
                      mapping=mapping.report,
                      accepted_chunk=str(output / "accepted_chunk.npz"))
    except BaseException as error:
        result["error"] = type(error).__name__ + ": " + str(error)
    save_json(output / "report.json", result)
    print("REPORT", output / "report.json", flush=True)
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

