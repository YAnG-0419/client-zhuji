#!/usr/bin/env python3
"""Read-only XFlow shadow planner. It never publishes or sends robot commands.

The ROS snapshot is collected by the system-Python subscription-only helper;
all planning and inference then run in the locked pixi environment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from online_dryrun import _write_result, run, save_json
from timing_policy import EXECUTION_CLOCK, FRAME_PER_STEP_30HZ, SYNC_HOLD


def ros_snapshot_command(client_dir, snapshot_path, seconds=2.0):
    return ["/usr/bin/python3", str(client_dir / "ros_readonly_observer.py"),
            "--seconds", str(float(seconds)), "--snapshot-npz", str(snapshot_path),
            "--config", str(client_dir / "deployment_config.json")]


def ros_environment(base=None):
    environment = dict(os.environ if base is None else base)
    python_path = "/opt/ros/foxy/lib/python3.8/site-packages"
    library_path = "/opt/ros/foxy/lib"
    if environment.get("PYTHONPATH"):
        python_path += os.pathsep + environment["PYTHONPATH"]
    if environment.get("LD_LIBRARY_PATH"):
        library_path += os.pathsep + environment["LD_LIBRARY_PATH"]
    environment.update({
        "FASTRTPS_DEFAULT_PROFILES_FILE":
            "/opt/ros-bringup/fastdds_config_udp_only.xml",
        "ROS_DOMAIN_ID": "0", "ROS_LOCALHOST_ONLY": "0",
        "PYTHONPATH": python_path, "LD_LIBRARY_PATH": library_path,
    })
    return environment


def collect_snapshot(client_dir, snapshot_path, runner=subprocess.run):
    command = ros_snapshot_command(client_dir, snapshot_path)
    answer = runner(command, env=ros_environment(), text=True,
                    capture_output=True, timeout=10, check=False)
    if answer.returncode != 0:
        detail = (answer.stdout + "\n" + answer.stderr).strip()
        raise RuntimeError("read-only ROS startup/snapshot gate failed: " + detail)
    if not snapshot_path.is_file():
        raise RuntimeError("ROS helper reported success without a snapshot")
    return answer.stdout.strip()


def main():
    client = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=client.parent)
    parser.add_argument("--task", required=True)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--image-mode", choices=("virtual_cam0", "stretch_undistort"),
                        default="virtual_cam0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-responses", type=Path)
    args = parser.parse_args()
    if not 1 <= args.cycles <= 3:
        parser.error("--cycles must be within [1,3]")
    output = (args.output or client / "runs" /
              ("r2-live-dryrun-" + time.strftime("%Y%m%d-%H%M%S"))).resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {"schema": "xflow_wuji2_live_dryrun_r2", "completed": False,
               "cycles_requested": args.cycles, "cycles_completed": 0,
               "infer_requests": 0, "execution_ready": False,
               "hardware_commands_sent": 0, "publishers_created": 0,
               "cycles": []}
    try:
        for cycle in range(args.cycles):
            destination = output / ("cycle%02d" % cycle)
            destination.mkdir()
            source = destination / "observation.npz"
            ros_output = collect_snapshot(client, source)
            planning_args = SimpleNamespace(
                root=args.root.resolve(), config=client / "deployment_config.json",
                source=source, server="remote", root_mode="auto",
                image_mode=args.image_mode, time_scale=FRAME_PER_STEP_30HZ,
                latency_policy=SYNC_HOLD, history_clock=EXECUTION_CLOCK,
                anchor_mode="decay", decay_steps=12, smoothing_window=5,
                speed_policy="reject", steps=32, task=args.task,
                num_inference_steps=10, save_responses=args.save_responses,
                response_label="live-cycle%02d-%s" % (cycle, args.image_mode))
            began = time.perf_counter()
            result = run(planning_args)
            planning_ms = (time.perf_counter() - began) * 1000.0
            target = _write_result(destination, result)
            mode = next(iter(result["reports"].values()))
            item = {"cycle": cycle, "ros_observer": ros_output,
                    "round_trip_ms": result["rpc"].get("round_trip_ms"),
                    "planning_ms": planning_ms,
                    "anchor": mode.get("anchor"),
                    "hand_speed_within_1rad_s": mode.get("hand_speed_within_1rad_s"),
                    "arm_speed_within_1rad_s": mode.get("offline_pinocchio_ik", {}).get(
                        "arm_joint_speed_within_1rad_s"),
                    "targets_npz": str(target), "execution_ready": False,
                    "hardware_commands_sent": 0, "publishers_created": 0}
            save_json(destination / "report.json", item)
            summary["cycles"].append(item)
            summary["cycles_completed"] += 1
            summary["infer_requests"] += int(result["rpc"].get("infer_requests", 0))
            print("cycle=%d rtt_ms=%.1f planning_ms=%.1f hand_gate=%s arm_gate=%s" %
                  (cycle, float(item["round_trip_ms"]), planning_ms,
                   item["hand_speed_within_1rad_s"], item["arm_speed_within_1rad_s"]))
        summary["completed"] = summary["cycles_completed"] == args.cycles
    except BaseException as error:
        summary["error"] = type(error).__name__ + ": " + str(error)
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0 if summary["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
