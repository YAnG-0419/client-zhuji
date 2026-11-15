"""Strict loader for self-contained replay_hand cross-embodiment NPZ episodes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "replay_hand_cross_v1"
SIDES = ("left", "right")


def hand_names(side: str) -> list[str]:
    return [f"{side}_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)]


def load_cross_episode(path: Path, hz: float = 50.0):
    path = Path(path).resolve()
    if not path.is_file() or path.suffix != ".npz":
        raise ValueError("replay_hand input must be an existing .npz file")
    if not np.isfinite(hz) or not 5 <= hz <= 100:
        raise ValueError("sample rate must be within 5..100 Hz")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "timestamp_ns", "t", "eef", "hands", "left_hand_joint_names",
            "right_hand_joint_names", "metadata_json",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"missing replay_hand arrays: {missing}")
        source_t = data["t"].astype(float)
        eef_source = data["eef"].astype(float)
        hands_source = data["hands"].astype(float)
        stamps = data["timestamp_ns"].astype(np.int64)
        metadata = json.loads(str(data["metadata_json"]))
        for side in SIDES:
            if data[f"{side}_hand_joint_names"].tolist() != hand_names(side):
                raise ValueError(f"unexpected {side} hand joint order")
    count = len(source_t)
    if metadata.get("schema") != SCHEMA:
        raise ValueError("unsupported replay_hand schema")
    if metadata.get('frame') != 'tron2_base_link':
        raise ValueError('Replay EEF must be explicitly mapped to tron2_base_link')
    if source_t.shape != (count,) or stamps.shape != (count,):
        raise ValueError("invalid replay time axis")
    if eef_source.shape != (count, 2, 7) or hands_source.shape != (count, 2, 20):
        raise ValueError("invalid EEF or hand array shape")
    if count < 2 or not all(np.isfinite(value).all() for value in (source_t, eef_source, hands_source)):
        raise ValueError("replay arrays must contain at least two finite frames")
    if abs(source_t[0]) > 1e-9 or np.any(np.diff(source_t) <= 0) or np.max(np.diff(source_t)) > 0.1:
        raise ValueError("replay time must start at zero, increase, and have gaps <=100 ms")
    if np.any(np.diff(stamps) <= 0):
        raise ValueError("source timestamps must increase")
    quaternion_norm = np.linalg.norm(eef_source[:, :, 3:], axis=-1)
    if np.max(np.abs(quaternion_norm - 1.0)) > 1e-3:
        raise ValueError("EEF quaternion is not unit length")

    duration = float(source_t[-1])
    query = np.arange(0.0, duration, 1.0 / hz)
    query = np.unique(np.r_[query, duration])
    from scipy.spatial.transform import Rotation, Slerp
    eef = np.empty((len(query), 2, 7))
    hands = np.empty((len(query), 2, 20))
    for side in range(2):
        for axis in range(3):
            eef[:, side, axis] = np.interp(query, source_t, eef_source[:, side, axis])
        eef[:, side, 3:] = Slerp(source_t, Rotation.from_quat(eef_source[:, side, 3:]))(query).as_quat()
        for joint in range(20):
            hands[:, side, joint] = np.interp(query, source_t, hands_source[:, side, joint])

    installed = Path(__file__).resolve().parents[1] / "record/config/model/robot.urdf"
    if not installed.is_file():
        raise ValueError(f"missing installed Tron2 replay URDF: {installed}")
    metadata = dict(metadata)
    metadata.update(
        episode=str(path),
        episode_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_duration_s=duration,
        resampling=f"Cartesian Slerp and hand linear interpolation at {hz:g} Hz",
        arm_source="Franka/Sharpa root mapped through assumed common camera frame to Tron2 SDK EEF",
        hand_source="Sharpa 21 keypoints retargeted directly to deployed Wuji Hand 2 q20",
        joint_state_use="No recorded Tron2 joints; replay_hand generates a multi-start first-frame and previous-solution continuous IK seed path",
        fk={
            "frame_id": "base_link",
            "quaternion_order": "xyzw",
            "implementation": "pengyang replay Model/FK",
            "urdf_sha256": hashlib.sha256(installed.read_bytes()).hexdigest(),
            "sdk_eef_offset_from_wrist_roll_m": [-0.0338, 0.0, -0.2125],
        },
    )
    return query, eef, hands, metadata
