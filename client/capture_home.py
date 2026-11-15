#!/usr/bin/env python3
"""Save the robot's current stationary pose as poses/home.json (read-only capture).

Subscribes to /joint_states and /wuji/{left,right}/joint_states only; never
publishes, enables, or commands anything.  The previous home.json is kept as
poses/home_<captured_at>.json before being replaced.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import json
import math
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

GROUPS = {
    "left_arm": ("/joint_states", ["abad_L_Joint", "hip_L_Joint", "yaw_L_Joint", "knee_L_Joint",
                                    "wrist_yaw_L_Joint", "wrist_pitch_L_Joint", "wrist_roll_L_Joint"]),
    "right_arm": ("/joint_states", ["abad_R_Joint", "hip_R_Joint", "yaw_R_Joint", "knee_R_Joint",
                                     "wrist_yaw_R_Joint", "wrist_pitch_R_Joint", "wrist_roll_R_Joint"]),
    "head": ("/joint_states", ["head_pitch_Joint", "head_yaw_Joint"]),
    "left_hand": ("/wuji/left/joint_states",
                  ["left_finger%d_joint%d" % (f, j) for f in range(1, 6) for j in range(1, 5)]),
    "right_hand": ("/wuji/right/joint_states",
                   ["right_finger%d_joint%d" % (f, j) for f in range(1, 6) for j in range(1, 5)]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", type=Path, default=ROOT / "poses/home.json")
    parser.add_argument("--seconds", type=float, default=1.0, help="stationary window")
    args = parser.parse_args()

    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node("capture_home_readonly")
    latest, history = {}, {topic: [] for topic, _ in GROUPS.values()}

    def receive(topic, msg):
        latest[topic] = dict(msg=msg, received_unix_ns=time.time_ns(),
                             received_monotonic_ns=time.monotonic_ns())
        history[topic].append((time.monotonic(), dict(zip(msg.name, msg.position))))

    for topic in history:
        node.create_subscription(JointState, topic, lambda m, t=topic: receive(t, m),
                                 qos_profile_sensor_data)
    try:
        deadline = time.monotonic() + max(3.0, args.seconds + 2.0)
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        missing = [t for t in history if t not in latest]
        if missing:
            raise RuntimeError("no feedback on %s (are the Wuji drivers running?)" % missing)
        # Stationary check over the last window on every captured joint.
        now = time.monotonic()
        for group, (topic, names) in GROUPS.items():
            window = [values for stamp, values in history[topic] if now - stamp <= args.seconds]
            if len(window) < 3:
                raise RuntimeError("too few %s samples in the stationary window" % topic)
            for name in names:
                series = [values[name] for values in window]
                if max(series) - min(series) > 0.01:
                    raise RuntimeError("%s is moving (%.4f rad span); hold still and retry"
                                       % (name, max(series) - min(series)))
        groups, raw = {}, {}
        for group, (topic, names) in GROUPS.items():
            item = latest[topic]; msg = item["msg"]
            values = dict(zip(msg.name, msg.position))
            positions = [float(values[name]) for name in names]
            if not all(math.isfinite(value) for value in positions):
                raise RuntimeError("non-finite feedback in " + group)
            stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            groups[group] = dict(source_topic=topic, source_stamp_ns=stamp, joint_names=names,
                                 positions_rad=positions,
                                 positions_deg=[math.degrees(value) for value in positions])
            raw[topic.strip("/").replace("/", "_")] = dict(
                topic=topic, source_stamp_ns=stamp, received_unix_ns=item["received_unix_ns"],
                received_monotonic_ns=item["received_monotonic_ns"], frame_id=msg.header.frame_id,
                joint_names=list(msg.name), positions_rad=[float(v) for v in msg.position])
        stamps = [g["source_stamp_ns"] for g in groups.values()]
        received = [v["received_monotonic_ns"] for v in raw.values()]
        captured = datetime.now(timezone(timedelta(hours=8)))
        document = {
            "schema_version": 1, "pose_name": "home", "kind": "measured_joint_feedback_snapshot",
            "captured_at": captured.isoformat(), "host": "10.192.1.4", "ros_distribution": "foxy",
            "position_unit": "rad", "display_position_unit": "deg",
            "source_stamp_span_ms": (max(stamps) - min(stamps)) / 1e6,
            "receipt_span_ms": (max(received) - min(received)) / 1e6,
            "groups": groups, "raw_feedback": raw,
            "notes": ("Read-only joint feedback snapshot captured by client/capture_home.py. "
                      "Capturing this file does not command motion or change controller settings."),
        }
        from home_trajectory import read_home
        read_home(document)  # same validation the home tool applies
        if args.home.exists():
            old = json.loads(args.home.read_text())
            stamp = old.get("captured_at", "unknown").replace("-", "").replace(":", "").replace("T", "_")
            backup = args.home.with_name("home_%s.json" % stamp.split("+")[0].replace(".", "_"))
            if not backup.exists():
                shutil.copy2(args.home, backup)
            print("previous home kept as", backup)
        args.home.write_text(json.dumps(document, indent=2) + "\n")
        print("saved", args.home)
        for group in groups:
            print(" %-10s" % group, [round(v, 3) for v in groups[group]["positions_rad"]])
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
