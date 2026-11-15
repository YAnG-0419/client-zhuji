#!/usr/bin/env python3
"""Passive ROS subscriber for controller target/state comparison.

Creates no publishers, service clients or robot SDK connections. Bridge stamps
may be publication times for cached SDK values, not original sensor timestamps.
"""
import argparse
import json
import math
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=3.)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or not 1 <= args.seconds <= 1200:
        parser.error('--seconds must be within [1,1200]')
    args.out.mkdir(parents=True, exist_ok=False)
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    rclpy.init()
    node = rclpy.create_node('read_only_arm_control_observer')
    counts = {'/joint_states': 0, '/joint_command': 0}
    last = {}
    began = time.monotonic()
    report = dict(read_only=True, motion_commands_sent=0,
                  timestamp_caveat='Bridge restamps cached SDK samples; freshness of source cannot be proved from ROS stamp alone.')
    try:
        with (args.out/'samples.jsonl').open('w') as log:
            def receive(topic, msg):
                data = dict(topic=topic, elapsed_s=time.monotonic()-began,
                            received_monotonic_s=time.monotonic(),
                            ros_stamp_ns=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec,
                            names=list(msg.name), position=list(msg.position),
                            velocity=list(msg.velocity), effort=list(msg.effort))
                for key in ('position', 'velocity', 'effort'):
                    if not all(math.isfinite(v) for v in data[key]):
                        data['nonfinite'] = True
                        data[key] = [v if math.isfinite(v) else None for v in data[key]]
                counts[topic] += 1
                last[topic] = data
                log.write(json.dumps(data, allow_nan=False)+'\n')
            subscriptions = [node.create_subscription(JointState, topic,
                             lambda msg, topic=topic: receive(topic, msg), qos_profile_sensor_data)
                             for topic in counts]
            while time.monotonic()-began < args.seconds:
                rclpy.spin_once(node, timeout_sec=.05)
            report['publishers'] = {topic: [dict(node_name=p.node_name, namespace=p.node_namespace,
                                               topic_type=p.topic_type)
                                           for p in node.get_publishers_info_by_topic(topic)] for topic in counts}
            report.update(duration_s=time.monotonic()-began, counts=counts, last=last)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        (args.out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(out=str(args.out), counts=counts, motion_commands_sent=0)))


if __name__ == '__main__':
    main()
