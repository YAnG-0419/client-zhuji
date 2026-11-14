#!/usr/bin/env python3
"""ROS2 read-only observer for zhuji.  This file never creates a publisher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState

from image_preprocess import (ImagePreprocessor, camera_info_dict,
                              camera_info_from_json, config_section,
                              validate_camera_info)
from observation_buffer import ObservationBuffer


TOPICS = {
    "image": "/camera/top/color/image_raw",
    "camera_info": "/camera/top/color/camera_info",
    "arm_joints": "/joint_states",
    "left_eef": "/left_arm/ee_pose",
    "right_eef": "/right_arm/ee_pose",
    "left_wuji": "/wuji/left/joint_states",
    "right_wuji": "/wuji/right/joint_states",
}


def stamp_ns(header):
    return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)


class ReadOnlyObserver(Node):
    def __init__(self):
        super().__init__("xflow_readonly_observer")
        self.buffer = ObservationBuffer(horizon_s=4.0)
        # All subscriptions use best-effort/volatile sensor QoS.  This is
        # compatible with current reliable publishers and historical Wuji QoS.
        self.create_subscription(Image, TOPICS["image"], self._image,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, TOPICS["camera_info"], self._camera_info,
                                 qos_profile_sensor_data)
        self.create_subscription(JointState, TOPICS["arm_joints"],
                                 lambda msg: self._joint("arm_joints", msg),
                                 qos_profile_sensor_data)
        self.create_subscription(PoseStamped, TOPICS["left_eef"],
                                 lambda msg: self._pose("left_eef", msg),
                                 qos_profile_sensor_data)
        self.create_subscription(PoseStamped, TOPICS["right_eef"],
                                 lambda msg: self._pose("right_eef", msg),
                                 qos_profile_sensor_data)
        self.create_subscription(JointState, TOPICS["left_wuji"],
                                 lambda msg: self._joint("left_wuji", msg),
                                 qos_profile_sensor_data)
        self.create_subscription(JointState, TOPICS["right_wuji"],
                                 lambda msg: self._joint("right_wuji", msg),
                                 qos_profile_sensor_data)

    def _guard(self, callback):
        try:
            callback()
        except BaseException as error:
            self.buffer.reader_error = error

    def _image(self, msg):
        self._guard(lambda: self.buffer.add_image(
            stamp_ns(msg.header), time.monotonic_ns(), msg.data,
            msg.width, msg.height, msg.step, msg.encoding, msg.header.frame_id))

    def _camera_info(self, msg):
        def add():
            value = camera_info_dict(msg.width, msg.height, msg.distortion_model,
                                     msg.k, msg.d, msg.header.frame_id)
            value["stamp_ns"] = stamp_ns(msg.header)
            self.buffer.set_camera_info(value)
        self._guard(add)

    def _joint(self, stream, msg):
        self._guard(lambda: self.buffer.add_state(
            stream, stamp_ns(msg.header), msg.position, msg.name))

    def _pose(self, stream, msg):
        pose = msg.pose
        value = [pose.position.x, pose.position.y, pose.position.z,
                 pose.orientation.x, pose.orientation.y,
                 pose.orientation.z, pose.orientation.w]
        self._guard(lambda: self.buffer.add_state(stream, stamp_ns(msg.header), value))

    def missing_topics(self):
        return [topic for topic in TOPICS.values()
                if self.count_publishers(topic) < 1]


def spin_for(node, seconds):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and node.buffer.reader_error is None:
        rclpy.spin_once(node, timeout_sec=min(0.05, deadline - time.monotonic()))


def save_image_samples(node, destination, config_path, archive_path):
    if not node.buffer.images or node.buffer.camera_info is None:
        raise RuntimeError("image/camera_info not received")
    import cv2
    destination.mkdir(parents=True, exist_ok=False)
    live = node.buffer.camera_info
    archived = camera_info_from_json(archive_path)
    gate = validate_camera_info(live, archived)
    current = node.buffer.images[-1]
    raw = np.frombuffer(current[2], np.uint8).reshape(480, 640, 3)
    config = config_section(config_path)
    virtual = ImagePreprocessor(live, config, "virtual_cam0")
    stretch = ImagePreprocessor(live, config, "stretch_undistort")
    outputs = {"raw_rgb.png": raw, "virtual_cam0_rgb320.png": virtual.process(raw),
               "stretch_undistort_rgb320.png": stretch.process(raw)}
    for name, rgb in outputs.items():
        if not cv2.imwrite(str(destination / name), rgb[:, :, ::-1]):
            raise RuntimeError("failed to write " + name)
    report = {
        "sample_ns": current[0], "receive_monotonic_ns": current[1],
        "camera_info_gate": gate, "live_camera_info": {
            "width": live["width"], "height": live["height"],
            "frame_id": live["frame_id"], "distortion_model": live["distortion_model"],
            "K": live["K"].tolist(), "D": live["D"].tolist()},
        "virtual_cam0": virtual.report(), "stretch_undistort": stretch.report(),
        "files": list(outputs), "hardware_commands_sent": 0,
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def save_snapshot_npz(node, destination, config_path, archive_path):
    """Serialize one read-only shadow sample; never creates a publisher."""
    missing = node.missing_topics()
    if missing:
        raise RuntimeError("missing topic publishers: " + ", ".join(missing))
    snapshot = node.buffer.snapshot(allow_missing_history=True,
                                    history_clock="execution")
    camera_gate = validate_camera_info(
        snapshot["camera_info"], camera_info_from_json(archive_path))
    states = snapshot["states"]; names = snapshot["state_names"]
    arrays = {
        "sample_ns": np.asarray(snapshot["sample_ns"], np.int64),
        "head_rgb": snapshot["head_rgb"],
        "arm_joint_names": np.asarray(names["arm_joints"]),
        "arm_joint_positions_rad": states["arm_joints"],
        "camera_K": np.asarray(snapshot["camera_info"]["K"], np.float64),
        "camera_D": np.asarray(snapshot["camera_info"]["D"], np.float64),
        "camera_width": np.asarray(snapshot["camera_info"]["width"]),
        "camera_height": np.asarray(snapshot["camera_info"]["height"]),
        "camera_frame_id": np.asarray(snapshot["camera_info"]["frame_id"]),
        "camera_distortion_model": np.asarray(
            snapshot["camera_info"]["distortion_model"]),
    }
    if snapshot["history_rgb"] is not None:
        arrays["history_rgb"] = snapshot["history_rgb"]
    for side in ("left", "right"):
        arrays[side + "_sdk_eef_xyzw"] = states[side + "_eef"]
        arrays[side + "_wuji_q20_rad"] = states[side + "_wuji"]
        arrays[side + "_wuji_joint_names"] = np.asarray(names[side + "_wuji"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return {"completed": True, "snapshot_npz": str(destination),
            "sample_ns": snapshot["sample_ns"],
            "history_clock": snapshot["history_clock"],
            "history_transmitted": snapshot["history_rgb"] is not None,
            "history_missing_reason": snapshot["history_missing_reason"],
            "measured_camera_fps": snapshot["measured_camera_fps"],
            "camera_info_gate": camera_gate,
            "alignment_error_ns": snapshot["alignment_error_ns"],
            "hardware_commands_sent": 0, "publishers_created": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--image-sample-dir", type=Path)
    parser.add_argument("--snapshot-report", type=Path)
    parser.add_argument("--snapshot-npz", type=Path)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("deployment_config.json"))
    parser.add_argument("--camera-archive", type=Path,
                        default=Path("/home/guest/pengyang/calibration/camera/color_camera_info_live.json"))
    args = parser.parse_args()
    rclpy.init()
    node = ReadOnlyObserver()
    exit_code = 0
    try:
        spin_for(node, args.seconds)
        if node.buffer.reader_error:
            raise RuntimeError(node.buffer.reader_error)
        # Attempt the synchronized snapshot before doing CPU-heavy image
        # preprocessing, so freshness diagnostics identify the actually
        # missing stream (currently Wuji state), not time spent saving PNGs.
        if args.snapshot_report:
            try:
                snapshot = node.buffer.snapshot(history_clock="wall")
                result = {"completed": True, "sample_ns": snapshot["sample_ns"],
                          "alignment_error_ns": snapshot["alignment_error_ns"],
                          "hardware_commands_sent": 0}
            except BaseException as error:
                result = {"completed": False,
                          "error": type(error).__name__ + ": " + str(error),
                          "hardware_commands_sent": 0}
                exit_code = 2
            args.snapshot_report.parent.mkdir(parents=True, exist_ok=True)
            args.snapshot_report.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
        if args.image_sample_dir:
            print(json.dumps(save_image_samples(node, args.image_sample_dir,
                                                args.config, args.camera_archive), indent=2))
        if args.snapshot_npz:
            try:
                result = save_snapshot_npz(node, args.snapshot_npz,
                                           args.config, args.camera_archive)
            except BaseException as error:
                result = {"completed": False,
                          "error": type(error).__name__ + ": " + str(error),
                          "missing_topics": node.missing_topics(),
                          "hardware_commands_sent": 0, "publishers_created": 0}
                exit_code = 2
            print(json.dumps(result, indent=2))
    finally:
        node.destroy_node(); rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

