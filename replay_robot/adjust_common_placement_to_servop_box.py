"""Translate a common-placement candidate into the ServoP xyz box."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


BOX = np.array([[[.250, .732], [-.213, .900], [-.673, .500]],
                [[.250, .732], [-.900, .213], [-.673, .500]]])


def transformed_xyz(eef, rigid):
    return eef[:, :, :3] @ rigid[:3, :3].T + rigid[:3, 3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--optimizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin-mm", type=float, default=1.)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite " + str(args.output))
    report = json.loads(args.optimizer.read_text())
    rigid = np.asarray(report["T_target_base_original_base"], dtype=float)
    with np.load(args.source, allow_pickle=False) as archive:
        eef = archive["eef"].copy()
    xyz = transformed_xyz(eef, rigid)
    lower_needed = np.max(BOX[..., 0] - xyz.min(axis=0), axis=0)
    upper_allowed = np.min(BOX[..., 1] - xyz.max(axis=0), axis=0)
    margin = args.margin_mm / 1000.
    delta = np.maximum(lower_needed + margin, 0.)
    if np.any(delta > upper_allowed - margin):
        raise ValueError("No pure common translation can fit the ServoP box")
    adjusted = rigid.copy()
    adjusted[:3, 3] += delta
    adjusted_xyz = transformed_xyz(eef, adjusted)
    if not np.all((adjusted_xyz >= BOX[..., 0]) & (adjusted_xyz <= BOX[..., 1])):
        raise AssertionError("Adjusted placement still leaves the ServoP box")
    result = dict(report)
    result["T_target_base_original_base"] = adjusted.tolist()
    result["first_eef_midpoint_m"] = (
        np.asarray(report["first_eef_midpoint_m"], dtype=float) + delta).tolist()
    result["servop_box_adjustment"] = dict(
        common_translation_m=delta.tolist(),
        margin_mm=args.margin_mm,
        before_passing_frames=int(np.all(
            (xyz >= BOX[..., 0]) & (xyz <= BOX[..., 1]), axis=(1, 2)).sum()),
        after_passing_frames=len(xyz),
        source_frames=len(xyz),
        rotation_unchanged=True,
        bimanual_relative_pose_unchanged=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["servop_box_adjustment"], indent=2))
