"""Strict, inference-only XFlow TCP client.

Only ``info``, ``reset`` and ``infer`` are exposed.  There are deliberately no
admin, load/unload, token, ROS, or robot-control APIs in this module.
"""
from __future__ import annotations

from pathlib import Path
import socket
import sys
import time
import uuid
from typing import Callable, Optional

import numpy as np


WIRE_DIR = Path(__file__).resolve().parent / "vendor/xflow_deploy"
if str(WIRE_DIR) not in sys.path:
    sys.path.insert(0, str(WIRE_DIR))
from wire_v4 import recv_message, send_message  # noqa: E402


PROTOCOL = "xflow_kp21_chunk_v1"
PROTOCOL_VERSION = 4
IMAGE_SPACES = ("raw_cam0", "rgb320")


class XFlowError(RuntimeError):
    pass


class XFlowProtocolError(XFlowError):
    pass


class XFlowRemoteError(XFlowError):
    def __init__(self, code, message):
        self.code = str(code)
        self.remote_message = str(message)
        super().__init__(self.code + ": " + self.remote_message)


def _array(value, dtype, shape, name):
    value = np.asarray(value)
    if value.dtype != np.dtype(dtype) or value.shape != shape:
        raise ValueError("%s must be %s %s" % (name, np.dtype(dtype), shape))
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError(name + " must be finite")
    return np.ascontiguousarray(value)


class XFlowClient:
    def __init__(self, host="127.0.0.1", port=28765, timeout_s=60.0, *,
                 model="hh_a12", deployment_id=None, sock=None,
                 infer_budget=None):
        self.sock = sock
        if self.sock is None:
            self.sock = socket.create_connection((host, int(port)), timeout=10.0)
        self.sock.settimeout(float(timeout_s))
        try:
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:  # AF_UNIX/socketpair in tests.
            pass
        # None: use the service's default_model (one model per service).
        self.model = None if model is None else str(model)
        self.expected_deployment_id = (None if deployment_id is None
                                       else str(deployment_id))
        self.infer_budget = None if infer_budget is None else int(infer_budget)
        if self.infer_budget is not None and self.infer_budget < 0:
            raise ValueError("infer_budget must be nonnegative")
        self.infer_count = 0
        self.session_id = None
        self.last_attempted_sample_ns = -1
        self.info_response = None
        self.model_info = None

    def _call(self, message):
        send_message(self.sock, message)
        response = recv_message(self.sock)
        request_id = message.get("request_id")
        if "error" in response:
            error = response.get("error")
            if (not isinstance(error, dict) or not isinstance(error.get("code"), str)
                    or not isinstance(error.get("message"), str)):
                raise XFlowProtocolError("malformed error response")
            if request_id is not None and response.get("request_id") != request_id:
                raise XFlowProtocolError("error response request_id mismatch")
            raise XFlowRemoteError(error["code"], error["message"])
        if request_id is not None and response.get("request_id") != request_id:
            raise XFlowProtocolError("response request_id mismatch")
        return response

    def info(self):
        response = self._call({"cmd": "info"})
        if (response.get("protocol") != PROTOCOL
                or response.get("protocol_version") != PROTOCOL_VERSION
                or response.get("wire") != "wire_v4"
                or response.get("horizon") != 32):
            raise XFlowProtocolError("unsupported XFlow protocol identity")
        models = response.get("models")
        tasks = response.get("tasks")
        if self.model is None and isinstance(models, dict):
            self.model = response.get("default_model")
        if not isinstance(models, dict) or self.model not in models:
            raise XFlowProtocolError("configured model is not loaded: %s (loaded: %s)"
                                     % (self.model, sorted(models) if isinstance(models, dict) else None))
        if not isinstance(tasks, dict) or not tasks:
            raise XFlowProtocolError("info has no tasks")
        model = models[self.model]
        required = ("deployment_id", "contract", "history_offset_frames",
                    "keypoint_time_s")
        if not isinstance(model, dict) or any(key not in model for key in required):
            raise XFlowProtocolError("model info is incomplete")
        times = np.asarray(model["keypoint_time_s"], dtype=float)
        if (times.shape != (32,) or not np.isfinite(times).all()
                or times[0] <= 0 or np.any(np.diff(times) <= 0)):
            raise XFlowProtocolError("model keypoint_time_s is invalid")
        deployment = str(model["deployment_id"])
        if self.expected_deployment_id is not None and deployment != self.expected_deployment_id:
            raise XFlowProtocolError("deployment_id mismatch: expected %s, got %s"
                                     % (self.expected_deployment_id, deployment))
        self.info_response = response
        self.model_info = model
        return response

    def reset(self):
        request_id = uuid.uuid4().hex
        response = self._call({"cmd": "reset", "request_id": request_id})
        session_id = response.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise XFlowProtocolError("reset response has no session_id")
        self.session_id = session_id
        self.last_attempted_sample_ns = -1
        return session_id

    def _validate_image(self, image, image_space, name):
        shape = (400, 640, 3) if image_space == "raw_cam0" else (320, 320, 3)
        return _array(image, np.uint8, shape, name)

    def infer(self, *, task, head_rgb, history_rgb, keypoints, sample_ns,
              image_space="rgb320", keypoint_valid=None, seed=0,
              num_inference_steps=10, speed_ratio=None):
        if self.info_response is None:
            raise XFlowError("info() must succeed before infer()")
        if self.session_id is None:
            raise XFlowError("reset() must succeed before infer()")
        if image_space not in IMAGE_SPACES:
            raise ValueError("unsupported image_space")
        if task not in self.info_response["tasks"]:
            raise ValueError("task is not advertised by the service")
        if type(sample_ns) is not int or sample_ns <= self.last_attempted_sample_ns:
            raise ValueError("sample_ns must strictly increase across attempts")
        if self.infer_budget is not None and self.infer_count >= self.infer_budget:
            raise XFlowError("local inference-call budget exhausted")
        head = self._validate_image(head_rgb, image_space, "head_rgb")
        history = (None if history_rgb is None else
                   self._validate_image(history_rgb, image_space, "history_rgb"))
        points = _array(keypoints, np.float32, (2, 21, 3), "keypoints")
        valid = None
        if keypoint_valid is not None:
            valid = _array(keypoint_valid, np.uint8, (2, 21), "keypoint_valid")
            if np.any((valid != 0) & (valid != 1)):
                raise ValueError("keypoint_valid must contain only 0 or 1")
        request_id = uuid.uuid4().hex
        observation = {"head_rgb": head, "keypoints": points}
        if history is not None:
            observation["history_rgb"] = history
        if valid is not None:
            observation["keypoint_valid"] = valid
        message = {
            "cmd": "infer", "request_id": request_id,
            "session_id": self.session_id, "sample_ns": sample_ns,
            "task": task, "model": self.model, "image_space": image_space,
            "seed": int(seed), "num_inference_steps": int(num_inference_steps),
            **({} if speed_ratio is None else {"speed_ratio": float(speed_ratio)}),
            "observation": observation,
        }
        # The server consumes request_id/sample_ns before validating later
        # fields.  Advance local state before the call, including error paths.
        self.last_attempted_sample_ns = sample_ns
        self.infer_count += 1
        began = time.perf_counter()
        try:
            response = self._call(message)
        except XFlowRemoteError as error:
            if error.code == "STALE_SESSION":
                self.session_id = None
            raise
        self._validate_infer_response(response, message, points, valid)
        response["round_trip_ms"] = (time.perf_counter() - began) * 1000.0
        return response

    def _validate_infer_response(self, response, request, points, valid):
        exact = {
            "protocol": PROTOCOL, "protocol_version": PROTOCOL_VERSION,
            "request_id": request["request_id"],
            "session_id": request["session_id"], "sample_ns": request["sample_ns"],
            "model": self.model, "deployment_id": self.model_info["deployment_id"],
            "contract": self.model_info["contract"], "task": request["task"],
            "horizon": 32, "hand_order": ["left", "right"],
            "frame": "cam0_color_optical_frame", "units": "metres",
        }
        for key, expected in exact.items():
            if response.get(key) != expected:
                raise XFlowProtocolError("infer response %s mismatch" % key)
        predicted = _array(response.get("keypoints"), np.float32,
                           (32, 2, 21, 3), "response keypoints")
        output_times = _array(response.get("keypoint_time_s"), np.float32,
                              (32,), "response keypoint_time_s")
        if "speed_ratio" in request:
            # Server >= 2026-11-15: times are robot time for the requested ratio.
            if not np.isclose(float(response.get("speed_ratio", np.nan)), request["speed_ratio"]):
                raise XFlowProtocolError("response speed_ratio differs from request")
            if output_times[0] <= 0 or np.any(np.diff(output_times) <= 0):
                raise XFlowProtocolError("response keypoint_time_s is not increasing")
        else:
            expected_times = np.asarray(self.model_info["keypoint_time_s"], np.float32)
            if not np.allclose(output_times, expected_times, atol=1e-6, rtol=0):
                raise XFlowProtocolError("response keypoint_time_s differs from info")
        proprio = _array(response.get("proprio_keypoints"), np.float32,
                         (2, 21, 3), "response proprio_keypoints")
        expected_proprio = points.copy()
        if valid is not None:
            expected_proprio[valid == 0] = 0
        if not np.array_equal(proprio, expected_proprio):
            raise XFlowProtocolError("server proprio_keypoints differs from request")
        if not isinstance(response.get("timings_ms"), dict):
            raise XFlowProtocolError("response has no timings_ms")
        if not np.isfinite(float(response.get("inference_seconds", np.nan))):
            raise XFlowProtocolError("response inference_seconds is invalid")
        return predicted

    def infer_fresh(self, sample_factory: Callable[[], dict], *, max_stale_resets=1):
        """Retry only STALE_SESSION, and only after obtaining a fresh sample."""
        for attempt in range(max_stale_resets + 1):
            if self.session_id is None:
                self.reset()
            sample = sample_factory()
            try:
                return self.infer(**sample)
            except XFlowRemoteError as error:
                if error.code != "STALE_SESSION" or attempt >= max_stale_resets:
                    raise
        raise AssertionError("unreachable")

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

