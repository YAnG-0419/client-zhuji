#!/usr/bin/env python3
"""Extract one synchronized, read-only XFlow observation from a ROS2 sqlite bag.

This is an offline system-Python adapter.  It opens sqlite with ``mode=ro``;
it never plays a bag and never creates a ROS graph entity.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3

import cv2
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


TOPICS = {
    "image": "/camera/top/color/image_raw/compressed",
    "camera_info": "/camera/top/color/camera_info",
    "arm_joints": "/joint_states",
    "left_eef": "/left_arm/ee_pose",
    "right_eef": "/right_arm/ee_pose",
    "left_fk": "/record/left_arm/ee_state_fk",
    "right_fk": "/record/right_arm/ee_state_fk",
    "left_wuji": "/wuji/left/joint_states",
    "right_wuji": "/wuji/right/joint_states",
}


def header_ns(message):
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def runtime_names(side):
    return tuple("%s_finger%d_joint%d" % (side, finger, joint)
                 for finger in range(1, 6) for joint in range(1, 5))


def pose7(message):
    value = message.pose
    return np.asarray([value.position.x, value.position.y, value.position.z,
                       value.orientation.x, value.orientation.y,
                       value.orientation.z, value.orientation.w], dtype=np.float64)


def pose_difference(actual, reference):
    translation = float(np.linalg.norm(actual[:3] - reference[:3]))
    qa = actual[3:] / np.linalg.norm(actual[3:])
    qb = reference[3:] / np.linalg.norm(reference[3:])
    rotation = 2.0 * math.acos(float(np.clip(abs(np.dot(qa, qb)), 0.0, 1.0)))
    return {"translation_m": translation, "rotation_rad": rotation}


class ReadOnlyBag:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        uri = "file:%s?mode=ro" % self.path
        self.connection = sqlite3.connect(uri, uri=True)
        self.topics = {}
        for topic_id, name, type_name in self.connection.execute(
                "SELECT id,name,type FROM topics"):
            self.topics[name] = (topic_id, type_name, get_message(type_name))
        missing = [name for name in TOPICS.values() if name not in self.topics]
        if missing:
            raise ValueError("bag is missing required topics: " + ", ".join(missing))

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def count(self, topic):
        topic_id = self.topics[topic][0]
        return int(self.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE topic_id=?", (topic_id,)).fetchone()[0])

    def at_index(self, topic, index):
        topic_id, _, message_type = self.topics[topic]
        row = self.connection.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1 OFFSET ?",
            (topic_id, int(index))).fetchone()
        if row is None:
            raise IndexError("message index is outside topic")
        return int(row[0]), deserialize_message(row[1], message_type)

    def nearest(self, topic, timestamp):
        topic_id, _, message_type = self.topics[topic]
        before = self.connection.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? AND timestamp<=? ORDER BY timestamp DESC LIMIT 1",
            (topic_id, int(timestamp))).fetchone()
        after = self.connection.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? AND timestamp>=? ORDER BY timestamp ASC LIMIT 1",
            (topic_id, int(timestamp))).fetchone()
        candidates = [row for row in (before, after) if row is not None]
        if not candidates:
            raise RuntimeError("topic has no message")
        row = min(candidates, key=lambda value: abs(int(value[0]) - int(timestamp)))
        return int(row[0]), deserialize_message(row[1], message_type)

    def nearest_header(self, topic, target_header_ns, db_hint_ns,
                       search_window_ns=100_000_000):
        """Select by message header while using the DB clock only as a window."""
        topic_id, _, message_type = self.topics[topic]
        rows = self.connection.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? AND timestamp BETWEEN ? AND ? ORDER BY timestamp",
            (topic_id, int(db_hint_ns - search_window_ns),
             int(db_hint_ns + search_window_ns))).fetchall()
        if not rows:
            raise RuntimeError("no %s messages near image DB timestamp" % topic)
        decoded = [(int(row[0]), deserialize_message(row[1], message_type))
                   for row in rows]
        return min(decoded, key=lambda item: abs(header_ns(item[1]) - target_header_ns))


def decode_rgb(message):
    if "jpeg" not in message.format.lower():
        raise ValueError("offline image is not JPEG")
    bgr = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None or bgr.shape != (480, 640, 3):
        raise ValueError("offline JPEG did not decode as 640x480 color")
    return np.ascontiguousarray(bgr[:, :, ::-1])


def extract_snapshot(bag, image_index, history_offset_frames=16,
                     state_tolerance_ns=10_000_000):
    image_topic = TOPICS["image"]
    count = bag.count(image_topic)
    if not history_offset_frames <= image_index < count:
        raise ValueError("image_index must leave enough history and remain in bag")
    image_db_ns, image = bag.at_index(image_topic, image_index)
    sample_ns = header_ns(image)
    period_headers = []
    for index in range(max(0, image_index - 60), image_index + 1):
        _, period_image = bag.at_index(image_topic, index)
        period_headers.append(header_ns(period_image))
    median_period_ns = float(np.median(np.diff(period_headers)))
    measured_fps = 1e9 / median_period_ns
    history_tolerance_ns = int(np.clip(median_period_ns / 2.0,
                                       16_700_000, 50_000_000))
    history_target_ns = sample_ns - int(round(history_offset_frames / 30.0 * 1e9))
    # SQLite receive timestamps and sensor header timestamps have a fixed
    # transport offset.  Select around the expected image index and compare
    # the actual headers, rather than mixing the two clock fields.
    candidates = []
    expected_index = image_index - history_offset_frames
    for index in range(max(0, expected_index - 2),
                       min(count, expected_index + 3)):
        _, candidate = bag.at_index(image_topic, index)
        candidates.append(candidate)
    history = min(candidates, key=lambda value: abs(header_ns(value) - history_target_ns))
    history_ns = header_ns(history)
    if abs(history_ns - history_target_ns) > history_tolerance_ns:
        raise RuntimeError("offline history frame is outside dynamic tolerance")
    arrays = {"head_rgb": decode_rgb(image), "history_rgb": decode_rgb(history)}
    report = {
        "schema": "xflow_offline_bag_snapshot_v1",
        "source_bag": str(bag.path), "source_is_jpeg_lossy": True,
        "image_format": image.format, "image_index": image_index,
        "sample_ns": sample_ns, "history_sample_ns": history_ns,
        "history_offset_frames": history_offset_frames,
        "history_clock": "execution_equals_wall_for_offline_recording",
        "measured_camera_fps": measured_fps,
        "history_tolerance_ms": history_tolerance_ns / 1e6,
        "history_target_error_ms": abs(history_ns - history_target_ns) / 1e6,
        "alignment_error_ms": {}, "hardware_commands_sent": 0,
        "execution_ready": False,
    }
    camera_db, camera = bag.nearest_header(TOPICS["camera_info"], sample_ns,
                                           image_db_ns)
    arrays.update(camera_K=np.asarray(camera.k, np.float64).reshape(3, 3),
                  camera_D=np.asarray(camera.d, np.float64),
                  camera_width=np.asarray(camera.width),
                  camera_height=np.asarray(camera.height),
                  camera_frame_id=np.asarray(camera.header.frame_id),
                  camera_distortion_model=np.asarray(camera.distortion_model))
    report["camera_info"] = {"K": arrays["camera_K"].tolist(),
                             "D": arrays["camera_D"].tolist(),
                             "width": int(camera.width), "height": int(camera.height),
                             "frame_id": camera.header.frame_id,
                             "distortion_model": camera.distortion_model}
    messages = {}
    for key in ("arm_joints", "left_eef", "right_eef", "left_wuji", "right_wuji"):
        db_ns, message = bag.nearest_header(TOPICS[key], sample_ns, image_db_ns)
        stamp = header_ns(message)
        error = abs(stamp - sample_ns)
        if error > state_tolerance_ns:
            raise RuntimeError("%s is %.3f ms from image" % (key, error / 1e6))
        messages[key] = message
        report["alignment_error_ms"][key] = error / 1e6
    arm = messages["arm_joints"]
    arrays["arm_joint_names"] = np.asarray(arm.name)
    arrays["arm_joint_positions_rad"] = np.asarray(arm.position, np.float64)
    for side in ("left", "right"):
        state = messages[side + "_wuji"]
        if tuple(state.name) != runtime_names(side) or len(state.position) != 20:
            raise ValueError(side + " Wuji JointState contract mismatch")
        arrays[side + "_wuji_joint_names"] = np.asarray(state.name)
        arrays[side + "_wuji_q20_rad"] = np.asarray(state.position, np.float64)
        arrays[side + "_sdk_eef_xyzw"] = pose7(messages[side + "_eef"])
        _, fk = bag.nearest_header(TOPICS[side + "_fk"], sample_ns, image_db_ns)
        arrays[side + "_record_fk_xyzw"] = pose7(fk)
        report[side + "_sdk_vs_record_fk"] = pose_difference(
            arrays[side + "_sdk_eef_xyzw"], arrays[side + "_record_fk_xyzw"])
    # Ground-truth future samples for the offline fake model.  These timestamps
    # follow the service's hh_a12 native target geometry; execution scheduling
    # is tested separately and currently places the 32 returned points at 30Hz.
    future_time_s = np.arange(1, 33, dtype=np.float64) / 120.0
    arrays["future_model_time_s"] = future_time_s
    for side in ("left", "right"):
        eef = []
        q20 = []
        future_errors = []
        for offset_s in future_time_s:
            target_ns = sample_ns + int(round(offset_s * 1e9))
            hint_ns = image_db_ns + int(round(offset_s * 1e9))
            _, eef_message = bag.nearest_header(TOPICS[side + "_eef"],
                                                target_ns, hint_ns)
            _, q_message = bag.nearest_header(TOPICS[side + "_wuji"],
                                              target_ns, hint_ns)
            if tuple(q_message.name) != runtime_names(side):
                raise ValueError(side + " future Wuji names differ")
            eef.append(pose7(eef_message)); q20.append(q_message.position)
            future_errors.append(max(abs(header_ns(eef_message) - target_ns),
                                     abs(header_ns(q_message) - target_ns)) / 1e6)
        arrays["future_" + side + "_sdk_eef_xyzw"] = np.asarray(eef, np.float64)
        arrays["future_" + side + "_wuji_q20_rad"] = np.asarray(q20, np.float64)
        report[side + "_future_alignment_max_ms"] = float(max(future_errors))
    arrays["sample_ns"] = np.asarray(sample_ns, np.int64)
    arrays["history_sample_ns"] = np.asarray(history_ns, np.int64)
    return arrays, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--image-index", type=int, default=800)
    parser.add_argument("--history-offset-frames", type=int, default=16)
    parser.add_argument("--snapshot-fractions",
                        help="comma-separated image fractions, e.g. 0.2,0.4,0.6,0.8")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        with ReadOnlyBag(args.bag) as bag:
            if args.snapshot_fractions:
                count = bag.count(TOPICS["image"])
                reports = {}
                for text in args.snapshot_fractions.split(","):
                    fraction = float(text)
                    if not 0.0 < fraction < 1.0:
                        raise ValueError("snapshot fractions must be within (0,1)")
                    index = int(round((count - 1) * fraction))
                    label = "frac%03d" % int(round(fraction * 100))
                    destination = args.output / label
                    destination.mkdir(parents=True, exist_ok=False)
                    arrays, item = extract_snapshot(
                        bag, index, args.history_offset_frames)
                    np.savez_compressed(destination / "observation.npz", **arrays)
                    item["fraction"] = fraction
                    item["npz"] = str(destination / "observation.npz")
                    (destination / "report.json").write_text(
                        json.dumps(item, indent=2) + "\n")
                    reports[label] = item
                report = {"schema": "xflow_offline_bag_multi_snapshot_r2",
                          "snapshots": reports, "hardware_commands_sent": 0,
                          "execution_ready": False}
                (args.output / "summary.json").write_text(
                    json.dumps(report, indent=2) + "\n")
            else:
                arrays, report = extract_snapshot(
                    bag, args.image_index, args.history_offset_frames)
                np.savez_compressed(args.output / "observation.npz", **arrays)
                report["npz"] = str(args.output / "observation.npz")
                (args.output / "report.json").write_text(
                    json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0
    except BaseException as error:
        report = {"completed": False,
                  "error": type(error).__name__ + ": " + str(error),
                  "hardware_commands_sent": 0}
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

