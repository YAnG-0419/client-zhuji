#!/usr/bin/env python3
"""Build a self-contained replay_hand episode from episode395 offline products.

This only transforms and synchronizes files. It never imports a robot SDK or ROS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA = "replay_hand_cross_v1"
SIDES = ("left", "right")
R_HAND_EEF = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
MOUNT_CENTER_EEF_M = {
    "left": np.array([-0.009962971509409037, 0.0009822991955358573, 0.07211807440614258]),
    "right": np.array([-0.0029837989, 0.0008311730, 0.0689440394]),
}
MOVE_P_BOX = {
    "left": np.array([[0.250, 0.732], [-0.213, 0.900], [-0.673, 0.500]]),
    "right": np.array([[0.250, 0.732], [-0.900, 0.213], [-0.673, 0.500]]),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hand_names(side: str) -> list[str]:
    return [f"{side}_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)]


def continuous_quaternions(matrices: np.ndarray) -> np.ndarray:
    quaternions = Rotation.from_matrix(matrices).as_quat()
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0:
            quaternions[index] *= -1
    return quaternions


def select_arm_indices(timestamps: np.ndarray, begin: int, end: int, hz: float) -> np.ndarray:
    available = (timestamps - begin) / 1e9
    duration = (end - begin) / 1e9
    query = np.arange(0.0, duration, 1.0 / hz)
    indices = np.searchsorted(available, query)
    indices = np.clip(indices, 0, len(timestamps) - 1)
    indices = np.unique(np.r_[indices, np.searchsorted(timestamps, end, side="right") - 1])
    indices = indices[(timestamps[indices] >= begin) & (timestamps[indices] <= end)]
    if len(indices) < 2:
        raise ValueError("Insufficient common replay interval")
    return indices


def build(arm_path: Path, hand_path: Path, output_path: Path, report_path: Path, hz: float) -> dict:
    if not np.isfinite(hz) or not 5 <= hz <= 100:
        raise ValueError("sample rate must be within 5..100 Hz")
    with np.load(arm_path, allow_pickle=False) as arm, np.load(hand_path, allow_pickle=False) as hand:
        arm_t = arm["timestamp_ns"].astype(np.int64)
        axes = [arm_t]
        for side in SIDES:
            axes.append(hand[f"{side}_timestamp_ns"].astype(np.int64))
        begin = max(int(axis[0]) for axis in axes)
        end = min(int(axis[-1]) for axis in axes)
        arm_indices = select_arm_indices(arm_t, begin, end, hz)
        stamps = arm_t[arm_indices]
        time_s = (stamps - stamps[0]) / 1e9
        eef = np.empty((len(stamps), 2, 7))
        hands = np.empty((len(stamps), 2, 20))
        mount_targets = np.empty((len(stamps), 2, 3))
        diagnostics = {}

        for side_index, side in enumerate(SIDES):
            transforms = arm[f"{side}_sharpa_root_T_tron_base"][arm_indices]
            palm_rotations = hand[f"{side}_wuji2_source_palm_rotation"]
            variation = (Rotation.from_matrix(palm_rotations[0]).inv() * Rotation.from_matrix(palm_rotations)).magnitude()
            if float(variation.max()) > 1e-8:
                raise ValueError(f"{side} source palm frame is not static")
            root_to_hand = palm_rotations[0]
            hand_rotation = transforms[:, :3, :3] @ root_to_hand
            eef_rotation = hand_rotation @ R_HAND_EEF
            mount_targets[:, side_index] = transforms[:, :3, 3]
            eef[:, side_index, :3] = mount_targets[:, side_index] - np.einsum(
                "nij,j->ni", eef_rotation, MOUNT_CENTER_EEF_M[side]
            )
            eef[:, side_index, 3:] = continuous_quaternions(eef_rotation)

            hand_t = hand[f"{side}_timestamp_ns"].astype(np.float64)
            hand_q = hand[f"{side}_wuji2_deployed_joint_positions"]
            names = hand[f"wuji2_{side}_joint_names"].tolist()
            if names != hand_names(side):
                raise ValueError(f"Unexpected {side} Wuji2 joint order")
            for joint in range(20):
                hands[:, side_index, joint] = np.interp(stamps, hand_t, hand_q[:, joint])

            box = MOVE_P_BOX[side]
            outside = np.any((eef[:, side_index, :3] < box[:, 0]) | (eef[:, side_index, :3] > box[:, 1]), axis=1)
            diagnostics[side] = {
                "sdk_eef_xyz_min_m": eef[:, side_index, :3].min(axis=0).tolist(),
                "sdk_eef_xyz_max_m": eef[:, side_index, :3].max(axis=0).tolist(),
                "move_p_box_outside_frames": int(outside.sum()),
                "mount_center_eef_m": MOUNT_CENTER_EEF_M[side].tolist(),
                "R_hand_eef": R_HAND_EEF.tolist(),
                "source_palm_frame_variation_deg_max": float(np.rad2deg(variation.max())),
            }

    metadata = {
        "schema": SCHEMA,
        "frame": "tron2_base_link",
        "eef_layout": "[left,right] xyz+xyzw",
        "hand_layout": "[left,right] firmware q20 radians",
        "source_time_unit": "ns",
        "sample_hz_requested": hz,
        "source_files": {
            str(arm_path.resolve()): sha256(arm_path),
            str(hand_path.resolve()): sha256(hand_path),
        },
        "camera_alignment": "Franka cam0 to Tron2 top camera assumed identity; Tron2 head 41.5 degree extrinsic",
        "mount_center_policy": "operator-confirmed board center equals current Wuji hand endpoint center",
        "orientation_policy": (
            "Static Sharpa-root to hand_coordinates frame, then R_HAND_EEF. "
            "Left mapping comes from replay_axis_20261104; right currently uses the mirrored-installation same-matrix assumption."
        ),
        "free_root_adapter_applied": False,
        "right_orientation_independently_validated": False,
        "motion_commands_sent": 0,
        "execution_ready": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            timestamp_ns=stamps,
            t=time_s,
            eef=eef,
            hands=hands,
            mount_targets_tron_base_m=mount_targets,
            left_hand_joint_names=np.asarray(hand_names("left")),
            right_hand_joint_names=np.asarray(hand_names("right")),
            metadata_json=np.asarray(json.dumps(metadata)),
        )
    report = {
        "status": "OFFLINE_REPLAY_HAND_INPUT_ONLY",
        "execution_ready": False,
        "motion_commands_sent": 0,
        "output": str(output_path.resolve()),
        "frames": int(len(time_s)),
        "duration_s": float(time_s[-1]),
        "sample_period_s": {
            "median": float(np.median(np.diff(time_s))),
            "max": float(np.max(np.diff(time_s))),
        },
        "diagnostics": diagnostics,
        "metadata": metadata,
        "warnings": [
            "Right R_hand_eef is a symmetric-installation assumption and has not been independently measured.",
            "Camera-frame identity and 41.5 degree alignment remain operator assumptions.",
            "The archive has not yet passed Tron2 IK or collision certification.",
            "Wuji/adapter/cables/table/object geometry is not included in the Tron2 arm collision model.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--hand", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--sample-hz", type=float, default=50.0)
    args = parser.parse_args()
    report = args.report or args.output.with_suffix(".json")
    print(json.dumps(build(args.arm, args.hand, args.output, report, args.sample_hz), indent=2))


if __name__ == "__main__":
    main()
