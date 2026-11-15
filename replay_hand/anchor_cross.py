"""Deterministic reachable first-frame anchor for cross-embodiment replay."""

from __future__ import annotations

import numpy as np
from replay_thresholds import MODEL_CLEARANCE_M
from scipy.spatial.transform import Rotation


SIDES = ("left", "right")
MOUNT_CENTER_EEF_M = np.array([
    [-0.009962971509409037, 0.0009822991955358573, 0.07211807440614258],
    [-0.002983798934724363, 0.0008311730062506703, 0.06894403937416507],
])

# Arm joints are the stationary final feedback actually reached during the
# first supervised approach trial on 2026-11-09. The original Wanchi offline
# anchor was 0.0400 rad beyond what the firmware reached. Re-anchoring at the
# measured pose keeps the relative episode motion while avoiding an idealized
# frame zero that the live MoveJ did not settle at. The head remains Pengyang's
# saved approximately 41.5-degree offline collision pose; live replay replaces
# it with measured head state and recertifies the path.
ANCHOR_Q16 = np.array([
    -0.9139001965522766, 0.25149989128112793, -0.18239986896514893,
    -1.0413998365402222, 0.2708001136779785, 0.5033998489379883,
    -0.14089995622634888, -0.9640997052192688, -0.43919992446899414,
    0.14329981803894043, -1.0765999555587769, -0.35039979219436646,
    0.4684000015258789, 0.2800999879837036,
    0.7233999967575073, 0.0037000179290771484,
])


def reanchor_eef(eef: np.ndarray, anchor_poses: np.ndarray):
    """Preserve base-frame relative mount motion at a reachable first pose.

    Source absolute camera position/orientation is deliberately replaced only
    at frame zero. Every later frame applies the source world-frame translation
    and rotation delta relative to frame zero to the fixed anchor.
    """
    source = np.asarray(eef, dtype=float)
    anchor = np.asarray(anchor_poses, dtype=float)
    if source.ndim != 3 or source.shape[1:] != (2, 7) or len(source) < 2:
        raise ValueError("expected EEF [frames,2,7]")
    if anchor.shape != (2, 7):
        raise ValueError("expected two anchor EEF poses")
    source_rotation = [Rotation.from_quat(source[:, side, 3:]) for side in range(2)]
    anchor_rotation = Rotation.from_quat(anchor[:, 3:])
    source_mount = np.empty((len(source), 2, 3))
    target = np.empty_like(source)
    target_mount = np.empty((len(source), 2, 3))
    translation = []
    rotation = []
    for side in range(2):
        source_mount[:, side] = source[:, side, :3] + source_rotation[side].apply(
            MOUNT_CENTER_EEF_M[side]
        )
        anchor_mount = anchor[side, :3] + anchor_rotation[side].apply(MOUNT_CENTER_EEF_M[side])
        target_mount[:, side] = anchor_mount + source_mount[:, side] - source_mount[0, side]
        delta = source_rotation[side] * source_rotation[side][0].inv()
        target_rotation = delta * anchor_rotation[side]
        target[:, side, :3] = target_mount[:, side] - target_rotation.apply(
            MOUNT_CENTER_EEF_M[side]
        )
        target[:, side, 3:] = target_rotation.as_quat()
        translation.append((anchor_mount - source_mount[0, side]).tolist())
        rotation.append(float((anchor_rotation[side] * source_rotation[side][0].inv()).magnitude()))
    first_position_error = np.linalg.norm(target[0, :, :3] - anchor[:, :3], axis=1)
    first_rotation_error = (
        Rotation.from_quat(target[0, :, 3:]).inv() * anchor_rotation
    ).magnitude()
    if max(first_position_error) > 1e-10 or max(first_rotation_error) > 1e-10:
        raise AssertionError("anchored frame zero must exactly equal anchor FK")
    report = {
        "policy": "fixed reachable frame-zero anchor; preserve source relative mount translation and world-frame wrist rotation",
        "absolute_camera_pose_preserved": False,
        "relative_motion_preserved": True,
        "anchor_q16": ANCHOR_Q16.tolist(),
        "anchor_eef_xyz_xyzw": anchor.tolist(),
        "per_arm_mount_translation_m": dict(zip(SIDES, translation)),
        "per_arm_frame_zero_rotation_change_rad": dict(zip(SIDES, rotation)),
        "mount_center_eef_m": dict(zip(SIDES, MOUNT_CENTER_EEF_M.tolist())),
        "arm_anchor_provenance": {
            "path": "/home/guest/pengyang/replay_hand/runs/replay_hand_20261109_115909_19458/feedback.jsonl",
            "sha256": "6b723950be7d309067eb7bae05ba39aed6091ccea304e48424357a9939128faa",
            "sample": "last stationary feedback before supervised approach timeout",
            "source_result_sha256": "ed26588d45021d664d4dd51b411162c54bd3e4cfbaf89a952c9c80351dd7fd56",
            "motion_commands_in_source_run": 1,
            "source_run_outcome": "MoveJ reached this collision-free measured pose; ServoP never started",
        },
        "head_anchor_provenance": {
            "path": "/home/guest/pengyang/poses/home.json",
            "sha256": "bf6e7658813aff0196e95598ede4c76b1ba3eeeb0d0521217d3fb7a4bb314084",
        },
    }
    return target, report


def anchor_episode(model, eef: np.ndarray):
    if np.any(ANCHOR_Q16 < model.lower) or np.any(ANCHOR_Q16 > model.upper):
        raise ValueError("recorded cross-replay anchor exceeds installed executable limits")
    model.set_q(ANCHOR_Q16)
    anchor_poses = model.poses().copy()
    target, report = reanchor_eef(eef, anchor_poses)
    clearance, pair = model.clearance()
    margin = np.minimum(ANCHOR_Q16 - model.lower, model.upper - ANCHOR_Q16)
    report.update(
        first_frame_clearance_score_m=float(clearance),
        first_frame_closest_bodies=pair,
        first_frame_arm_joint_margin_rad=float(margin[:14].min()),
    )
    if clearance < MODEL_CLEARANCE_M or margin[:14].min() < 0.03:
        raise ValueError("cross-replay anchor lacks required clearance or arm joint margin")
    return target, ANCHOR_Q16.copy(), report
