#!/usr/bin/env python3
"""Serve a recorded TRON2 + Wuji trajectory through the XFlow wire_v4 API.

This process has no ROS, robot transport, or command publisher.  It converts
the episode's base-frame SDK EEF poses and Wuji q20 values to the exact
MediaPipe21/cam0 representation consumed by the live client.  Each inference
advances once through the recording; after the end it returns a stationary
final frame and never loops.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import socket
import socketserver
import sys
import threading
import time
import traceback
import uuid

import numpy as np
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PENGYANG_ROOT", HERE.parent)).resolve()
WIRE = ROOT / "client/vendor/xflow_deploy"
sys.path[:0] = [str(HERE), str(ROOT / "client"), str(WIRE)]

from online_core import DeploymentMapping, SIDES  # noqa: E402
from proprio import ProprioKinematics  # noqa: E402
from wire_v4 import recv_message, send_message  # noqa: E402


PROTOCOL = "xflow_kp21_chunk_v1"
MODEL_NAME = "replay_episode2"
TASKS = {
    "flower": {"instruction": "Replay simulator compatibility task: flower."},
    "whiteboard": {"instruction": "Replay simulator compatibility task: whiteboard."},
    "tidying": {"instruction": "Replay Episode 2: table tidying."},
    "lemon": {"instruction": "Replay simulator compatibility task: lemon."},
    "chili": {"instruction": "Replay simulator compatibility task: chili."},
}
PALM_INDICES = np.asarray([0, 1, 5, 9, 13, 17], dtype=int)


class ServiceError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def pose7_matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("EEF pose must be finite xyz+xyzw")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(value[3:]).as_matrix()
    result[:3, 3] = value[:3]
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_keypoint_path(root: Path, config_path: Path, episode_path: Path):
    with np.load(episode_path, allow_pickle=False) as archive:
        t = np.asarray(archive["t"], dtype=float)
        eef = np.asarray(archive["eef"], dtype=float)
        hands = np.asarray(archive["hands"], dtype=float)
    if (t.ndim != 1 or len(t) < 2 or eef.shape != (len(t), 2, 7)
            or hands.shape != (len(t), 2, 20)
            or not all(np.isfinite(value).all() for value in (t, eef, hands))
            or np.any(np.diff(t) <= 0)):
        raise ValueError("episode has invalid t/eef/hands arrays")
    t = t - t[0]
    mapping = DeploymentMapping.load(root, config_path, "live_zhuji_camera")
    kinematics = ProprioKinematics(mapping)
    keypoints = np.empty((len(t), 2, 21, 3), dtype=np.float32)
    for index in range(len(t)):
        base_eef = {side: pose7_matrix(eef[index, side_index])
                    for side_index, side in enumerate(SIDES)}
        q20 = {side: hands[index, side_index]
               for side_index, side in enumerate(SIDES)}
        keypoints[index] = kinematics.measured_keypoints(base_eef, q20)
    return t, keypoints


class ReplayService:
    def __init__(self, root: Path, config_path: Path, episode_path: Path):
        self.root = Path(root).resolve()
        self.config_path = Path(config_path).resolve()
        self.episode_path = Path(episode_path).resolve()
        self.t, self.keypoints = load_keypoint_path(
            self.root, self.config_path, self.episode_path)
        self.flat_keypoints = self.keypoints.reshape(len(self.t), -1)
        self.episode_sha256 = file_sha256(self.episode_path)
        self.deployment_id = "replay-" + self.episode_sha256[:16]
        self.sessions = OrderedDict()
        self.lock = threading.Lock()
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    @staticmethod
    def timing(speed_ratio):
        ratio = 4.0 if speed_ratio is None else float(speed_ratio)
        if not np.isfinite(ratio) or not 0.25 <= ratio <= 16.0:
            raise ServiceError("INVALID_REQUEST", "speed_ratio must be within [0.25,16]")
        times = np.arange(1, 33, dtype=np.float32) * np.float32(ratio / 120.0)
        return ratio, int(round(4.0 * ratio)), times

    def model_info(self):
        ratio, history, times = self.timing(None)
        return {
            "name": MODEL_NAME,
            "weights": str(self.episode_path),
            "deployment_id": self.deployment_id,
            "contract": "humanhand",
            "codec": "none",
            "speed_ratio": ratio,
            "training_history_offset_frames": 4,
            "history_offset_frames": history,
            "keypoint_time_s": times.tolist(),
            "dt_s": float(times[0]),
            "horizon_s": float(times[-1]),
            "replay_duration_s": float(self.t[-1]),
            "replay_frames": int(len(self.t)),
            "end_policy": "hold_final_frame_no_loop",
        }

    def info(self, _message):
        return {
            "protocol": PROTOCOL,
            "protocol_version": 4,
            "wire": "wire_v4",
            "default_model": MODEL_NAME,
            "models": {MODEL_NAME: self.model_info()},
            "registry": [MODEL_NAME],
            "tasks": TASKS,
            "horizon": 32,
            "timing": "recorded path sampled at response keypoint_time_s; one chunk per infer",
            "keypoint_layout": "[2,21,3] in / [32,2,21,3] out; hand order L,R; MediaPipe21; metres; cam0 optical frame",
            "image": "images are validated for protocol compatibility but ignored by the replay simulator",
            "simulator": {
                "episode": str(self.episode_path),
                "episode_sha256": self.episode_sha256,
                "end_policy": "hold_final_frame_no_loop",
                "started_at": self.started_at,
            },
        }

    def reset(self, message):
        session_id = uuid.uuid4().hex
        with self.lock:
            self.sessions[session_id] = {
                "last_ns": -1,
                "request_ids": OrderedDict(),
                "cursor_s": 0.0,
                "infer_count": 0,
            }
            while len(self.sessions) > 256:
                self.sessions.popitem(last=False)
        return {"session_id": session_id, "request_id": message.get("request_id")}

    def _session_for_infer(self, message):
        with self.lock:
            session = self.sessions.get(message.get("session_id"))
            if session is None:
                raise ServiceError("STALE_SESSION", "unknown session; send reset")
            request_id = message.get("request_id")
            sample_ns = message.get("sample_ns")
            if not request_id or request_id in session["request_ids"]:
                raise ServiceError("INVALID_REQUEST", "missing or duplicate request_id")
            if not isinstance(sample_ns, int) or sample_ns <= session["last_ns"]:
                raise ServiceError("INVALID_REQUEST", "sample_ns must be an increasing integer per session")
            session["last_ns"] = sample_ns
            session["request_ids"][request_id] = None
            while len(session["request_ids"]) > 1024:
                session["request_ids"].popitem(last=False)
            return session

    @staticmethod
    def _image(value, image_space, name):
        image = np.asarray(value)
        expected = (400, 640, 3) if image_space == "raw_cam0" else (320, 320, 3)
        if image_space not in ("raw_cam0", "rgb320"):
            raise ServiceError("INVALID_REQUEST", "unsupported image_space")
        if image.dtype != np.uint8 or image.shape != expected:
            raise ServiceError("INVALID_REQUEST", "%s must be uint8 %s" % (name, expected))
        return image

    def sample(self, query_t):
        query_t = np.clip(np.asarray(query_t, dtype=float), self.t[0], self.t[-1])
        output = np.empty((len(query_t), self.flat_keypoints.shape[1]), dtype=np.float32)
        for column in range(self.flat_keypoints.shape[1]):
            output[:, column] = np.interp(
                query_t, self.t, self.flat_keypoints[:, column]).astype(np.float32)
        return output.reshape(len(query_t), 2, 21, 3)

    @staticmethod
    def _rigid_fit(source, target):
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
        u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        translation = target_center - source_center @ rotation.T
        return rotation, translation

    def anchored_sample(self, start_s, query_t, proprio, valid):
        """Replay only relative motion, rooted at this request's measured hands."""
        reference = self.sample([start_s])[0]
        predicted = self.sample(query_t)
        reports = []
        for side in range(2):
            indices = PALM_INDICES[valid[side, PALM_INDICES]]
            if len(indices) < 3:
                indices = np.flatnonzero(valid[side])
            if len(indices) < 3:
                raise ServiceError("INVALID_REQUEST", "at least three valid keypoints per hand are required")
            rotation, translation = self._rigid_fit(
                reference[side, indices], proprio[side, indices])
            aligned_reference = reference[side] @ rotation.T + translation
            aligned_future = predicted[:, side] @ rotation.T + translation
            point_offset = np.where(valid[side, :, None],
                                    proprio[side] - aligned_reference, 0.0)
            predicted[:, side] = aligned_future + point_offset[None]
            reports.append({
                "rigid_translation_m": translation.tolist(),
                "point_offset_max_m": float(np.linalg.norm(point_offset, axis=-1).max()),
            })
        return predicted.astype(np.float32), reports

    def infer(self, message):
        began = time.perf_counter()
        session = self._session_for_infer(message)
        model = message.get("model") or MODEL_NAME
        if model != MODEL_NAME:
            raise ServiceError("INVALID_REQUEST", "model is not loaded: " + str(model))
        task = message.get("task")
        if task not in TASKS:
            raise ServiceError("INVALID_REQUEST", "unknown task: " + str(task))
        observation = message.get("observation") or {}
        image_space = message.get("image_space", "raw_cam0")
        if "head_rgb" not in observation or "keypoints" not in observation:
            raise ServiceError("INVALID_REQUEST", "observation requires head_rgb and keypoints")
        self._image(observation["head_rgb"], image_space, "head_rgb")
        if observation.get("history_rgb") is not None:
            self._image(observation["history_rgb"], image_space, "history_rgb")
        points = np.asarray(observation["keypoints"])
        if points.dtype != np.float32 or points.shape != (2, 21, 3):
            raise ServiceError("INVALID_REQUEST", "keypoints must be float32 (2,21,3)")
        valid = np.ones((2, 21), dtype=bool)
        if "keypoint_valid" in observation:
            raw_valid = np.asarray(observation["keypoint_valid"])
            if raw_valid.dtype != np.uint8 or raw_valid.shape != (2, 21):
                raise ServiceError("INVALID_REQUEST", "keypoint_valid must be uint8 (2,21)")
            valid = raw_valid.astype(bool)
        valid &= np.isfinite(points).all(axis=-1)
        proprio = np.where(valid[..., None], points, 0).astype(np.float32)
        ratio, history_frames, output_times = self.timing(message.get("speed_ratio"))

        with self.lock:
            start_s = float(session["cursor_s"])
            query_t = np.minimum(start_s + output_times.astype(float), self.t[-1])
            predicted, anchor_report = self.anchored_sample(
                start_s, query_t, proprio, valid)
            session["cursor_s"] = float(query_t[-1])
            session["infer_count"] += 1
            infer_count = int(session["infer_count"])
            end_s = float(session["cursor_s"])

        elapsed = time.perf_counter() - began
        return {
            "protocol": PROTOCOL,
            "protocol_version": 4,
            "request_id": message["request_id"],
            "session_id": message["session_id"],
            "sample_ns": message["sample_ns"],
            "model": MODEL_NAME,
            "deployment_id": self.deployment_id,
            "weights": str(self.episode_path),
            "contract": "humanhand",
            "task": task,
            "instruction": TASKS[task]["instruction"],
            "hand_order": ["left", "right"],
            "frame": "cam0_color_optical_frame",
            "units": "metres",
            "keypoints": predicted,
            "keypoint_time_s": output_times,
            "dt_s": float(output_times[0]),
            "speed_ratio": ratio,
            "history_offset_frames": history_frames,
            "history_rgb_sent": "history_rgb" in observation,
            "horizon": 32,
            "proprio_keypoints": proprio,
            "timings_ms": {
                "preprocess": 0.0,
                "t5_vlm": 0.0,
                "dino": 0.0,
                "denoise": 0.0,
                "total": elapsed * 1000.0,
            },
            "inference_seconds": elapsed,
            "replay_cursor_start_s": start_s,
            "replay_cursor_end_s": end_s,
            "replay_complete": bool(end_s >= self.t[-1]),
            "replay_infer_count": infer_count,
            "replay_request_anchor": anchor_report,
        }

    def handle(self, message):
        command = message.get("cmd")
        function = {"info": self.info, "reset": self.reset, "infer": self.infer}.get(command)
        if function is None:
            raise ServiceError("INVALID_REQUEST", "unknown command: " + str(command))
        return function(message)


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        sock = self.request
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            try:
                message = recv_message(sock)
            except (ConnectionError, OSError):
                return
            except ValueError as error:
                send_message(sock, {"error": {"code": "INVALID_REQUEST", "message": str(error)}})
                continue
            try:
                response = self.server.service.handle(message)
            except ServiceError as error:
                response = {"error": {"code": error.code, "message": str(error)},
                            "request_id": message.get("request_id")}
            except Exception as error:
                traceback.print_exc()
                response = {"error": {"code": "MODEL_ERROR", "message": repr(error)},
                            "request_id": message.get("request_id")}
            try:
                send_message(sock, response)
            except (ConnectionError, OSError):
                return


class ReplayServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28769)
    parser.add_argument("--episode", type=Path,
                        default=ROOT / "replay_robot/episode2_wuji2_a5da567e_20261116/replay_input_50hz.npz")
    parser.add_argument("--config", type=Path, default=ROOT / "client/deployment_config.json")
    parser.add_argument("--endpoint-file", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    service = ReplayService(ROOT, args.config, args.episode)
    server = ReplayServer((args.host, args.port), Handler)
    server.service = service
    endpoint = {
        "host": args.host,
        "port": args.port,
        "pid": os.getpid(),
        "protocol": PROTOCOL,
        "model": MODEL_NAME,
        "deployment_id": service.deployment_id,
        "episode": str(service.episode_path),
        "episode_sha256": service.episode_sha256,
        "frames": len(service.t),
        "duration_s": float(service.t[-1]),
        "end_policy": "hold_final_frame_no_loop",
        "started_at": service.started_at,
    }
    if args.endpoint_file:
        args.endpoint_file.parent.mkdir(parents=True, exist_ok=True)
        args.endpoint_file.write_text(json.dumps(endpoint, indent=2) + "\n")
    print("REPLAY SIM SERVER READY " + json.dumps(endpoint, sort_keys=True), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
