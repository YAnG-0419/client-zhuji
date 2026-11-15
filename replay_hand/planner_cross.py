"""Generate a deterministic continuous Tron2 IK seed path for cross-embodiment EEF."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from replay_thresholds import MODEL_CLEARANCE_M


def saved_home(root: Path) -> np.ndarray:
    data = json.loads((root / "poses/home.json").read_text())
    groups = data["groups"]
    return np.asarray(
        groups["left_arm"]["positions_rad"]
        + groups["right_arm"]["positions_rad"]
        + groups["head"]["positions_rad"],
        dtype=float,
    )


def candidates(root: Path, model, preferred=None) -> list[tuple[str, np.ndarray]]:
    home = saved_home(root)
    ready = np.r_[[0.0, 0.0, 0.0, -1.6, 0.0, 0.0, 0.0] * 2, home[14:]]
    middle = (model.lower + model.upper) / 2.0
    middle[14:] = home[14:]
    result = [("saved_home", home), ("test_ik_ready", ready), ("joint_midpoint", middle)]
    if preferred is not None:
        name, value = preferred
        result.insert(0, (str(name), np.asarray(value, dtype=float).copy()))
    return result


def solve_seed_path(base, model, root: Path, t: np.ndarray, eef: np.ndarray, out: Path,
                    preferred=None):
    from episode import pose_error

    attempts = []
    accepted = []
    for name, initial in candidates(root, model, preferred):
        detail = {"name": name}
        try:
            if np.any(initial < model.lower) or np.any(initial > model.upper):
                raise ValueError("candidate outside executable joint limits")
            q = base.solve_eef(model, initial.copy(), eef[0])
            q[14:] = initial[14:]
            model.set_q(q)
            position, rotation = pose_error(model.poses(), eef[0])
            clearance, pair = model.clearance()
            nearest = model.poses().copy()
            detail.update(
                position_error_m=float(max(position)),
                rotation_error_rad=float(max(rotation)),
                per_arm_position_error_m=position.tolist(),
                per_arm_rotation_error_rad=rotation.tolist(),
                target_eef_xyz_xyzw=eef[0].tolist(),
                nearest_eef_xyz_xyzw=nearest.tolist(),
                solved_q16=q.tolist(),
                clearance_score_m=float(clearance),
                closest_bodies=pair,
                max_joint_change_rad=float(np.max(np.abs(q[:14] - initial[:14]))),
            )
            if max(position) > base.IK_POSITION_TOL or max(rotation) > base.IK_ROTATION_TOL:
                raise ValueError("first-frame pose residual exceeds replay tolerances")
            if clearance < MODEL_CLEARANCE_M:
                raise ValueError("first-frame clearance below 5 mm")
            margin = float(np.min(np.minimum(q - model.lower, model.upper - q)))
            detail["minimum_joint_margin_rad"] = margin
            cost = float(max(position) * 100 + max(rotation) + 0.01 * np.linalg.norm(q[:14] - initial[:14]))
            accepted.append((cost, name, q.copy(), detail))
        except Exception as exc:
            detail["error"] = type(exc).__name__ + ": " + str(exc)
        attempts.append(detail)
    if not accepted:
        (out / "cross_seed_failure.json").write_text(json.dumps({"attempts": attempts}, indent=2) + "\n")
        raise ValueError("no first-frame cross IK candidate passed")
    accepted.sort(key=lambda item: item[0])
    _, selected_name, q, selected = accepted[0]
    path = [q.copy()]
    max_position = max_rotation = max_step = 0.0
    for index in range(1, len(t)):
        previous = path[-1]
        q = base.solve_eef(model, previous.copy(), eef[index])
        q[14:] = path[0][14:]
        model.set_q(q)
        position, rotation = pose_error(model.poses(), eef[index])
        max_position = max(max_position, float(max(position)))
        max_rotation = max(max_rotation, float(max(rotation)))
        step = float(np.max(np.abs(q[:14] - previous[:14])))
        max_step = max(max_step, step)
        if max(position) > base.IK_POSITION_TOL or max(rotation) > base.IK_ROTATION_TOL:
            raise ValueError(
                "cross seed IK at %.3fs exceeds tolerance: %.5fm / %.5frad"
                % (t[index], max(position), max(rotation))
            )
        if np.any(q < model.lower + 1e-5) or np.any(q > model.upper - 1e-5):
            raise ValueError("cross seed IK reaches a joint limit at %.3fs" % t[index])
        if step > 0.20:
            raise ValueError("cross seed IK discontinuity %.4frad at %.3fs" % (step, t[index]))
        path.append(q.copy())
        if index % 200 == 0:
            print("CROSS SEED IK %d/%d" % (index + 1, len(t)), flush=True)
    path = np.asarray(path)
    report = {
        "selected_first_seed": selected_name,
        "first_frame_attempts": attempts,
        "frames": int(len(path)),
        "max_position_error_m": max_position,
        "max_rotation_error_rad": max_rotation,
        "max_joint_step_rad": max_step,
        "head_policy": "fixed at saved home 41.5 degree head pose for offline planning; live replay replaces with measured head and recertifies",
    }
    (out / "cross_seed.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(out / "cross_seed.npz", t=t, q16=path)
    return path, report
