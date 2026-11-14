"""NumPy-only zhuji RGB preprocessing for XFlow ``image_space=rgb320``."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


MODES = ("virtual_cam0", "stretch_undistort")


def camera_info_dict(width, height, distortion_model, K, D, frame_id=""):
    K = np.asarray(K, dtype=float).reshape(3, 3)
    D = np.asarray(D, dtype=float).reshape(-1)
    if (int(width) <= 1 or int(height) <= 1 or distortion_model != "plumb_bob"
            or D.shape != (5,) or not np.isfinite(K).all()
            or not np.isfinite(D).all() or K[2, 2] != 1):
        raise ValueError("unsupported or invalid camera_info")
    return {"width": int(width), "height": int(height),
            "distortion_model": distortion_model, "K": K, "D": D,
            "frame_id": str(frame_id)}


def camera_info_from_json(path):
    value = json.loads(Path(path).read_text())
    return camera_info_dict(value["width"], value["height"],
                            value["distortion_model"], value["k"], value["d"],
                            value.get("frame_id", value.get("header", {}).get("frame_id", "")))


def validate_camera_info(live, archived, max_relative_intrinsic=0.005,
                         distortion_atol=1e-9):
    for key in ("width", "height", "distortion_model"):
        if live[key] != archived[key]:
            raise ValueError("camera_info %s differs from archive" % key)
    if live.get("frame_id") and archived.get("frame_id") and live["frame_id"] != archived["frame_id"]:
        raise ValueError("camera_info frame_id differs from archive")
    if not np.allclose(live["D"], archived["D"], atol=distortion_atol, rtol=0):
        raise ValueError("camera distortion differs from archive")
    indices = ((0, 0), (1, 1), (0, 2), (1, 2))
    relative = {}
    for row, column in indices:
        current = float(live["K"][row, column])
        reference = float(archived["K"][row, column])
        delta = abs(current - reference) / max(abs(reference), 1e-12)
        relative["K%d%d" % (row, column)] = delta
        if delta > max_relative_intrinsic:
            raise ValueError("camera intrinsic drift exceeds %.3f%%" %
                             (100 * max_relative_intrinsic))
    return {"max_relative_intrinsic": max(relative.values()),
            "relative_intrinsics": relative,
            "distortion_max_abs": float(np.max(np.abs(live["D"] - archived["D"])))}


def _distort(x, y, K, D):
    k1, k2, p1, p2, k3 = D
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]


def undistort_map(source, output_K, output_size):
    """Equivalent to cv2.initUndistortRectifyMap for identity rectification."""
    width, height = map(int, output_size)
    u, v = np.meshgrid(np.arange(width, dtype=float),
                       np.arange(height, dtype=float))
    x = (u - output_K[0, 2]) / output_K[0, 0]
    y = (v - output_K[1, 2]) / output_K[1, 1]
    map_x, map_y = _distort(x, y, source["K"], source["D"])
    return map_x.astype(np.float32), map_y.astype(np.float32)


def remap_bilinear(rgb, map_x, map_y):
    rgb = np.asarray(rgb)
    if (rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3
            or map_x.shape != map_y.shape):
        raise ValueError("expected uint8 HxWx3 RGB and equal 2-D maps")
    height, width = rgb.shape[:2]
    x0 = np.floor(map_x).astype(np.int64)
    y0 = np.floor(map_y).astype(np.int64)
    dx = map_x - x0
    dy = map_y - y0
    output = np.zeros(map_x.shape + (3,), dtype=np.float64)
    for ox, oy, weight in (
            (0, 0, (1 - dx) * (1 - dy)),
            (1, 0, dx * (1 - dy)),
            (0, 1, (1 - dx) * dy),
            (1, 1, dx * dy)):
        x = x0 + ox; y = y0 + oy
        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        safe_x = np.clip(x, 0, width - 1)
        safe_y = np.clip(y, 0, height - 1)
        output += rgb[safe_y, safe_x] * (weight * valid)[..., None]
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def _area_weights(source_length, target_length):
    scale = source_length / float(target_length)
    weights = np.zeros((target_length, source_length), dtype=np.float32)
    for target in range(target_length):
        left, right = target * scale, (target + 1) * scale
        first, last = int(np.floor(left)), int(np.ceil(right))
        for source in range(first, last):
            overlap = max(0.0, min(right, source + 1.0) - max(left, source))
            if 0 <= source < source_length:
                weights[target, source] = overlap / scale
    return weights


def resize_area(rgb, target_width, target_height):
    """Separable exact pixel-area resize (the INTER_AREA operation used here)."""
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("expected uint8 HxWx3 RGB")
    source_height, source_width = rgb.shape[:2]
    wy = _area_weights(source_height, int(target_height))
    wx = _area_weights(source_width, int(target_width))
    temp = (wy @ rgb.reshape(source_height, -1).astype(np.float32)).reshape(
        int(target_height), source_width, 3)
    output = np.empty((int(target_height), int(target_width), 3), dtype=np.float32)
    for channel in range(3):
        output[:, :, channel] = temp[:, :, channel] @ wx.T
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


class ImagePreprocessor:
    def __init__(self, live_camera_info, config, mode="virtual_cam0"):
        if mode not in MODES:
            raise ValueError("unknown image preprocessing mode")
        self.camera = live_camera_info
        self.config = config
        self.mode = mode
        if (self.camera["width"], self.camera["height"]) != (640, 480):
            raise ValueError("zhuji source image must be 640x480")
        virtual_K = np.asarray(config["cam0"]["virtual_K_320"], dtype=float).reshape(3, 3)
        self.virtual_map = undistort_map(self.camera, virtual_K, (320, 320))
        self.virtual_invalid = ((self.virtual_map[0] < 0)
                                | (self.virtual_map[0] > self.camera["width"] - 1)
                                | (self.virtual_map[1] < 0)
                                | (self.virtual_map[1] > self.camera["height"] - 1))
        stretch_K = np.asarray(config["zhuji_stretch"]["new_K_alpha0"],
                               dtype=float).reshape(3, 3)
        self.stretch_map = undistort_map(self.camera, stretch_K, (640, 480))

    @property
    def black_border_ratio(self):
        return float(np.mean(self.virtual_invalid))

    def process(self, rgb):
        rgb = np.asarray(rgb)
        expected = (self.camera["height"], self.camera["width"], 3)
        if rgb.dtype != np.uint8 or rgb.shape != expected:
            raise ValueError("source RGB must be uint8 %s" % (expected,))
        if self.mode == "virtual_cam0":
            return remap_bilinear(rgb, *self.virtual_map)
        rectified = remap_bilinear(rgb, *self.stretch_map)
        return resize_area(rectified, 320, 320)

    def report(self):
        return {
            "mode": self.mode, "output_shape": [320, 320, 3],
            "image_space": "rgb320",
            "virtual_cam0_black_border_ratio": self.black_border_ratio,
            "pixel_center_convention": self.config["pixel_center_convention"],
        }


def config_section(path):
    return json.loads(Path(path).read_text())["image_preprocess"]

