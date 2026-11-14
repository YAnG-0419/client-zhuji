"""Read-only proprioceptive FK helpers for the XFlow client.

This module accepts already-sampled SDK EEF poses and Wuji joint states.  It
contains no ROS node, publisher, controller, or robot transport.
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence

import numpy as np

from hand_retarget_wuji2 import Hand
from online_core import DeploymentMapping, SIDES, rigid


def firmware_joint_names(side: str):
    if side not in SIDES:
        raise ValueError("side must be left or right")
    return tuple("%s_finger%d_joint%d" % (side, finger, joint)
                 for finger in range(1, 6) for joint in range(1, 5))


def ordered_q20(side: str, names: Sequence[str], positions: Sequence[float]) -> np.ndarray:
    """Reorder a named JointState into installed `/wuji` finger-major order."""
    expected = firmware_joint_names(side)
    if len(names) != len(positions) or len(set(names)) != len(names):
        raise ValueError("joint names/positions must be unique and equally sized")
    lookup = dict(zip(names, positions))
    missing = [name for name in expected if name not in lookup]
    if missing:
        raise ValueError("missing Wuji joints: " + ", ".join(missing))
    q = np.asarray([lookup[name] for name in expected], dtype=float)
    if q.shape != (20,) or not np.isfinite(q).all():
        raise ValueError("Wuji joint positions must be 20 finite radians")
    return q


def _checked_q(side: str, q: Sequence[float], mapping: DeploymentMapping) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (20,) or not np.isfinite(q).all():
        raise ValueError(side + " q20 must contain 20 finite positions")
    bounds = mapping.limits[side]
    if np.any(q < bounds[:, 0] - 1e-8) or np.any(q > bounds[:, 1] + 1e-8):
        raise ValueError(side + " q20 is outside installed limits")
    return q


def measured_keypoints(mapping: DeploymentMapping,
                       base_T_sdk_eef: Mapping[str, np.ndarray],
                       q20: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Return [left,right] MediaPipe-ordered Wuji FK points in camera optical."""
    camera_T_base = np.linalg.inv(mapping.base_from_source_camera)
    output = np.empty((2, 21, 3), dtype=float)
    for side_index, side in enumerate(SIDES):
        base_T_eef = rigid(base_T_sdk_eef[side], "base_T_%s_sdk_eef" % side)
        q = _checked_q(side, q20[side], mapping)
        base_T_hand = rigid(base_T_eef @ mapping.sdk_eef_from_hand[side],
                            "base_T_%s_hand" % side)
        camera_T_hand = rigid(camera_T_base @ base_T_hand,
                              "camera_T_%s_hand" % side)
        local = Hand("wuji", side).fk(q)
        output[side_index] = (local @ camera_T_hand[:3, :3].T
                              + camera_T_hand[:3, 3])
    if not np.isfinite(output).all():
        raise ValueError("proprioceptive keypoints are nonfinite")
    return output


def camera_hand_to_sdk_eef(mapping: DeploymentMapping, side: str,
                           camera_T_hand: np.ndarray) -> np.ndarray:
    """Invert the calibrated hand/tool chain for one accepted retarget pose."""
    if side not in SIDES:
        raise ValueError("side must be left or right")
    camera_T_hand = rigid(camera_T_hand, "camera_T_hand")
    return rigid(mapping.base_from_source_camera @ camera_T_hand
                 @ np.linalg.inv(mapping.sdk_eef_from_hand[side]),
                 "base_T_sdk_eef")

