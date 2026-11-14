"""Transport-independent bounded buffer for synchronized XFlow observations."""
from __future__ import annotations

from collections import deque
import time

import numpy as np


STATE_STREAMS = ("arm_joints", "left_eef", "right_eef", "left_wuji", "right_wuji")


def _immutable(value, dtype=None):
    output = np.array(value, dtype=dtype, copy=True)
    if not np.isfinite(output).all():
        raise ValueError("state contains nonfinite values")
    output.setflags(write=False)
    return output


class ObservationBuffer:
    """Stores epoch-header timestamps and separate monotonic receive times."""
    def __init__(self, horizon_s=1.5, camera_fps=30.0):
        if horizon_s < 1.0 or camera_fps <= 0:
            raise ValueError("invalid observation buffer horizon/fps")
        self.horizon_ns = int(round(horizon_s * 1e9))
        self.camera_fps = float(camera_fps)
        self.images = deque()
        self.states = {name: deque() for name in STATE_STREAMS}
        self.camera_info = None
        self.reader_error = None

    def _trim(self, queue, newest_ns):
        cutoff = int(newest_ns) - self.horizon_ns
        while queue and queue[0][0] < cutoff:
            queue.popleft()

    def add_image(self, stamp_ns, receive_monotonic_ns, rgb_bytes,
                  width=640, height=480, step=1920,
                  encoding="rgb8", frame_id="top_color_optical_frame"):
        if (type(stamp_ns) is not int or type(receive_monotonic_ns) is not int
                or stamp_ns <= 0 or receive_monotonic_ns <= 0):
            raise ValueError("image timestamps must be positive integers")
        if (encoding != "rgb8" or (width, height, step) != (640, 480, 1920)
                or frame_id != "top_color_optical_frame"):
            raise ValueError("unexpected live image layout/frame")
        raw = bytes(rgb_bytes)
        if len(raw) != height * step:
            raise ValueError("image byte count differs from height*step")
        if self.images and stamp_ns <= self.images[-1][0]:
            raise ValueError("image header timestamp is not strictly increasing")
        self.images.append((stamp_ns, receive_monotonic_ns, raw))
        self._trim(self.images, stamp_ns)

    def add_state(self, stream, stamp_ns, value, names=()):
        if stream not in self.states or type(stamp_ns) is not int or stamp_ns <= 0:
            raise ValueError("invalid state stream/timestamp")
        queue = self.states[stream]
        if queue and stamp_ns < queue[-1][0]:
            raise ValueError(stream + " timestamp moved backward")
        queue.append((stamp_ns, _immutable(value), tuple(names)))
        self._trim(queue, stamp_ns)

    def set_camera_info(self, value):
        self.camera_info = value

    @staticmethod
    def _nearest(queue, target_ns):
        if not queue:
            raise RuntimeError("required observation stream has no samples")
        return min(queue, key=lambda item: abs(item[0] - target_ns))

    def snapshot(self, history_offset_frames=4, state_tolerance_ns=10_000_000,
                 history_tolerance_ns=20_000_000, max_receive_age_ns=150_000_000,
                 now_monotonic_ns=None):
        if self.reader_error is not None:
            raise RuntimeError("observation reader failed: " + str(self.reader_error))
        if self.camera_info is None or not self.images:
            raise RuntimeError("camera image/info not ready")
        if type(history_offset_frames) is not int or history_offset_frames <= 0:
            raise ValueError("history_offset_frames must be a positive integer")
        current = self.images[-1]
        now = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
        if now - current[1] > max_receive_age_ns:
            raise RuntimeError("latest image is stale by local monotonic receive time")
        target_history_ns = current[0] - int(round(history_offset_frames / self.camera_fps * 1e9))
        history = self._nearest(self.images, target_history_ns)
        if history[0] == current[0] or abs(history[0] - target_history_ns) > history_tolerance_ns:
            raise RuntimeError("required history image is unavailable")
        aligned = {}
        alignment_error_ns = {}
        names = {}
        for stream in STATE_STREAMS:
            item = self._nearest(self.states[stream], current[0])
            delta = abs(item[0] - current[0])
            if delta > state_tolerance_ns:
                raise RuntimeError(stream + " is not aligned to image")
            aligned[stream] = item[1]
            alignment_error_ns[stream] = delta
            names[stream] = item[2]
        current_rgb = np.frombuffer(current[2], dtype=np.uint8).reshape(480, 640, 3)
        history_rgb = np.frombuffer(history[2], dtype=np.uint8).reshape(480, 640, 3)
        current_rgb.setflags(write=False); history_rgb.setflags(write=False)
        return {
            "sample_ns": current[0],
            "receive_monotonic_ns": current[1],
            "head_rgb": current_rgb, "history_rgb": history_rgb,
            "history_sample_ns": history[0],
            "history_target_error_ns": abs(history[0] - target_history_ns),
            "states": aligned, "state_names": names,
            "alignment_error_ns": alignment_error_ns,
            "camera_info": self.camera_info,
        }

