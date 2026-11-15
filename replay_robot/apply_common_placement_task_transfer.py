"""Apply one common SE(3) placement to task-transfer replay inputs.

This is the field-name adapter for replay_hand/apply_common_placement.py.  The
task-transfer schema names poses by direction (tron_base_T_*), while the older
102/395 schema used *_T_tron_base for the same stored direction.  Time, hands,
per-side tool transforms and bimanual relative poses are preserved.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


POSE_FIELDS = (
    "tron_base_T_hand",
    "tron_base_T_board_center",
    "tron_base_T_sdk_eef",
)


def apply(source, output, rigid, provenance):
    g = np.asarray(rigid, dtype=float)
    if (g.shape != (4, 4) or not np.isfinite(g).all()
            or not np.allclose(g[3], [0, 0, 0, 1], atol=1e-12)
            or not np.allclose(g[:3, :3].T @ g[:3, :3], np.eye(3), atol=1e-10)
            or abs(np.linalg.det(g[:3, :3]) - 1) > 1e-10):
        raise ValueError("Task placement must be a proper rigid transform")
    if output.exists():
        raise FileExistsError("Refusing to overwrite " + str(output))
    with np.load(source, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    meta = json.loads(str(data["metadata_json"]))
    if (meta.get("placement_policy") != "preserve_common_frame_v1"
            or meta.get("frame") != "tron2_base_link" or "task_placement" in meta):
        raise ValueError("Expected an original, unplaced task-transfer input")
    if not all(name in data for name in POSE_FIELDS):
        raise ValueError("Missing canonical tron_base_T_* pose arrays")

    original_time = data["t"].copy()
    original_stamps = data["timestamp_ns"].copy()
    original_hands = data["hands"].copy()
    original_tool_transforms = {
        side + "_" + suffix: data[side + "_" + suffix].copy()
        for side in ("left", "right")
        for suffix in ("T_sdk_eef_board_center", "T_board_center_hand")
    }
    old_hand = data["tron_base_T_hand"].copy()
    old_relative = np.linalg.inv(old_hand[:, 0]) @ old_hand[:, 1]

    eef = data["eef"].copy()
    eef[:, :, :3] = eef[:, :, :3] @ g[:3, :3].T + g[:3, 3]
    for side in range(2):
        eef[:, side, 3:] = (
            Rotation.from_matrix(g[:3, :3]) * Rotation.from_quat(eef[:, side, 3:])
        ).as_quat()
    data["eef"] = eef
    for name in POSE_FIELDS:
        data[name] = g @ data[name]
    # Historical loader alias; despite its name it stores tron_base_T_hand.
    data["hand_root_T_tron_base"] = data["tron_base_T_hand"].copy()

    new_relative = (np.linalg.inv(data["tron_base_T_hand"][:, 0])
                    @ data["tron_base_T_hand"][:, 1])
    relative_error = float(np.max(np.abs(old_relative - new_relative)))
    if relative_error > 1e-9:
        raise AssertionError("Bimanual relative pose changed")
    np.testing.assert_array_equal(data["t"], original_time)
    np.testing.assert_array_equal(data["timestamp_ns"], original_stamps)
    np.testing.assert_array_equal(data["hands"], original_hands)
    for side in ("left", "right"):
        for suffix in ("T_sdk_eef_board_center", "T_board_center_hand"):
            name = side + "_" + suffix
            np.testing.assert_array_equal(data[name], original_tool_transforms[name])

    meta.update(
        execution_ready=False,
        motion_commands_sent=0,
        task_placement=dict(
            T_target_base_original_base=g.tolist(),
            source_frame="tron2_base_link",
            provenance=provenance,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            bimanual_relative_matrix_error_max=relative_error,
            semantics=("Offline common rigid task placement after measured camera mapping; "
                       "not a replacement calibration"),
            adapter="replay_robot/apply_common_placement_task_transfer.py",
        ),
    )
    data["metadata_json"] = np.asarray(json.dumps(meta))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    args = parser.parse_args()
    search = json.loads(args.search.read_text())
    chosen = search["accepted"][0] if "accepted" in search else search
    result = apply(args.source, args.output,
                   chosen["T_target_base_original_base"], str(args.search))
    print(json.dumps(result["task_placement"], indent=2))
