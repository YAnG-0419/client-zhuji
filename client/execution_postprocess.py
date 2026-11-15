"""Pure execution-layer anchoring, smoothing, and diagnostic speed gates.

This module only transforms arrays. It deliberately contains no ROS, socket,
driver, publisher, or hardware-command code.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


ANCHOR_MODES = ("decay", "constant", "none")
SPEED_POLICIES = ("reject", "clamp_report")


def validate_execution_config(config):
    try:
        execution = config["execution"]
        anchor = execution["anchor"]
        observation = config["observation"]
        timing = config["timing"]
    except (KeyError, TypeError) as error:
        raise ValueError("execution configuration is incomplete: " + str(error))
    anchor_weights(2, str(anchor["mode"]), int(anchor["decay_steps"]))
    _valid_window(32, int(execution["smoothing_window"]))
    if type(anchor["apply_to_hand_q"]) is not bool:
        raise ValueError("apply_to_hand_q must be bool")
    if execution["speed_policy"] not in SPEED_POLICIES:
        raise ValueError("unknown speed policy")
    if (float(execution["hand_joint_speed_limit_rad_s"]) <= 0
            or float(execution["arm_joint_speed_limit_rad_s"]) <= 0):
        raise ValueError("speed limits must be positive")
    if float(observation["buffer_seconds"]) < 4.0:
        raise ValueError("observation buffer_seconds must be at least 4")
    if timing["latency_policy"] != "sync_hold":
        raise ValueError("deployment default latency policy must be sync_hold")
    if timing["history_clock"] not in ("execution", "wall"):
        raise ValueError("unknown history clock")
    return True


def _transforms(value, name="transforms"):
    value = np.asarray(value, dtype=float)
    if (value.ndim != 3 or value.shape[1:] != (4, 4)
            or not np.isfinite(value).all()):
        raise ValueError(name + " must be Nx4x4 finite transforms")
    if not np.allclose(value[:, 3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError(name + " has invalid homogeneous rows")
    return value


def _stats(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values):
        return {"mean": 0.0, "std": 0.0, "max": 0.0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values)),
            "max": float(np.max(values))}


def select_root_mode(configured, contract):
    if configured != "auto":
        if configured not in ("free_root", "kabsch_fixed_root"):
            raise ValueError("unknown root mode: " + str(configured))
        return configured, "explicit CLI/config override"
    if contract == "humanhand":
        return "free_root", "humanhand output uses non-rigid finger-base landmarks"
    if contract == "realrobot":
        return "kabsch_fixed_root", "realrobot output preserves installed-hand geometry"
    raise ValueError("auto root mode does not know contract: " + str(contract))


def anchor_weights(length, mode="decay", decay_steps=12):
    if mode not in ANCHOR_MODES:
        raise ValueError("anchor mode must be decay, constant, or none")
    if type(decay_steps) is not int or decay_steps < 1:
        raise ValueError("decay_steps must be a positive integer")
    if mode == "constant":
        return np.ones(length, dtype=float)
    if mode == "none":
        return np.zeros(length, dtype=float)
    x = np.minimum(np.arange(length, dtype=float) / decay_steps, 1.0)
    smoothstep = x * x * (3.0 - 2.0 * x)
    return 1.0 - smoothstep


def _valid_window(length, window):
    if type(window) is not int or window < 1 or window % 2 == 0:
        raise ValueError("smoothing window must be a positive odd integer")
    if window == 1 or length < 3:
        return 1
    return min(window, length if length % 2 else length - 1)


def _smooth_values(value, window):
    value = np.asarray(value, dtype=float)
    window = _valid_window(len(value), window)
    if window == 1:
        return value.copy()
    return savgol_filter(value, window_length=window,
                         polyorder=min(2, window - 1), axis=0, mode="interp")


def smooth_root_trajectory(roots, window=5):
    roots = _transforms(roots)
    window = _valid_window(len(roots), window)
    if window == 1:
        return roots.copy()
    output = roots.copy()
    output[:, :3, 3] = _smooth_values(roots[:, :3, 3], window)
    r0 = Rotation.from_matrix(roots[0, :3, :3])
    rotations = Rotation.from_matrix(roots[:, :3, :3])
    relative_vectors = (r0.inv() * rotations).as_rotvec()
    smoothed_vectors = _smooth_values(relative_vectors, window)
    output[:, :3, :3] = (r0 * Rotation.from_rotvec(smoothed_vectors)).as_matrix()
    return output


def smooth_q_trajectory(q, window=5):
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or not np.isfinite(q).all():
        raise ValueError("q must be one finite NxJ trajectory")
    return _smooth_values(q, window)


def apply_local_anchor(roots_fit, q_fit, measured_root, measured_q, limits, *,
                       mode="decay", decay_steps=12, apply_to_hand_q=True):
    """Right-multiply a partial local SE(3) offset into fitted hand roots."""
    roots_fit = _transforms(roots_fit, "roots_fit")
    measured_root = _transforms(np.asarray(measured_root)[None], "measured_root")[0]
    q_fit = np.asarray(q_fit, dtype=float)
    measured_q = np.asarray(measured_q, dtype=float)
    limits = np.asarray(limits, dtype=float)
    if (q_fit.ndim != 2 or measured_q.shape != (q_fit.shape[1],)
            or limits.shape != (q_fit.shape[1], 2)
            or len(q_fit) != len(roots_fit)
            or not np.isfinite(q_fit).all()
            or not np.isfinite(measured_q).all()
            or not np.isfinite(limits).all()):
        raise ValueError("invalid anchor q/limit shapes")
    weights = anchor_weights(len(roots_fit), mode, decay_steps)
    local_offset = np.linalg.inv(roots_fit[0]) @ measured_root
    offset_rotation = Rotation.from_matrix(local_offset[:3, :3]).as_rotvec()
    anchored_roots = roots_fit.copy()
    for index, weight in enumerate(weights):
        partial = np.eye(4)
        partial[:3, :3] = Rotation.from_rotvec(weight * offset_rotation).as_matrix()
        partial[:3, 3] = weight * local_offset[:3, 3]
        anchored_roots[index] = roots_fit[index] @ partial
    if apply_to_hand_q:
        q_offset = measured_q - q_fit[0]
        anchored_q = q_fit + weights[:, None] * q_offset[None]
    else:
        q_offset = np.zeros_like(measured_q)
        anchored_q = q_fit.copy()
    anchored_q = np.clip(anchored_q, limits[:, 0], limits[:, 1])
    report = {
        "mode": mode, "decay_steps": decay_steps,
        "apply_to_hand_q": bool(apply_to_hand_q),
        "weights": weights.tolist(),
        "local_offset_translation_m": local_offset[:3, 3].tolist(),
        "local_offset_translation_norm_m": float(np.linalg.norm(local_offset[:3, 3])),
        "local_offset_rotation_rotvec_rad": offset_rotation.tolist(),
        "local_offset_rotation_rad": float(np.linalg.norm(offset_rotation)),
        "q_offset_abs_max_rad": float(np.max(np.abs(q_offset))),
        "q_limit_clip_count": int(np.count_nonzero(
            np.abs(anchored_q - (q_fit + weights[:, None] * q_offset[None])) > 1e-12)),
    }
    return anchored_roots, anchored_q, report


def pose_delta(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    translation = np.linalg.norm(a[..., :3, 3] - b[..., :3, 3], axis=-1)
    relative = np.swapaxes(a[..., :3, :3], -1, -2) @ b[..., :3, :3]
    rotation = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()
    return translation, rotation.reshape(translation.shape)


def check_stationary_since_sample(sample_eef, dispatch_eef, sample_q20,
                                  dispatch_q20, *, max_translation_m=0.002,
                                  max_rotation_rad=math.radians(1.0),
                                  max_joint_rad=0.05):
    sample_eef = _transforms(sample_eef, "sample_eef")
    dispatch_eef = _transforms(dispatch_eef, "dispatch_eef")
    sample_q20 = np.asarray(sample_q20, dtype=float)
    dispatch_q20 = np.asarray(dispatch_q20, dtype=float)
    if sample_eef.shape != dispatch_eef.shape or sample_q20.shape != dispatch_q20.shape:
        raise ValueError("stationary check shapes differ")
    translation, rotation = pose_delta(sample_eef, dispatch_eef)
    joint = np.abs(dispatch_q20 - sample_q20)
    reasons = []
    if float(np.max(translation)) > max_translation_m:
        reasons.append("EEF translation changed")
    if float(np.max(rotation)) > max_rotation_rad:
        reasons.append("EEF rotation changed")
    if float(np.max(joint)) > max_joint_rad:
        reasons.append("hand joint changed")
    return {
        "stationary": not reasons, "rejection_reasons": reasons,
        "eef_translation_max_m": float(np.max(translation)),
        "eef_rotation_max_rad": float(np.max(rotation)),
        "hand_joint_max_rad": float(np.max(joint)),
        "thresholds": {"eef_translation_m": max_translation_m,
                       "eef_rotation_rad": max_rotation_rad,
                       "hand_joint_rad": max_joint_rad},
    }


def trajectory_jitter(roots, q):
    roots = _transforms(roots)
    q = np.asarray(q, dtype=float)
    if len(roots) < 3:
        zeros = _stats([])
        return {"root_translation_second_difference_m": zeros,
                "root_rotation_second_difference_rad": zeros,
                "q_second_difference_rad": zeros}
    translation_dd = np.linalg.norm(np.diff(roots[:, :3, 3], n=2, axis=0), axis=1)
    r0 = Rotation.from_matrix(roots[0, :3, :3])
    rv = (r0.inv() * Rotation.from_matrix(roots[:, :3, :3])).as_rotvec()
    rotation_dd = np.linalg.norm(np.diff(rv, n=2, axis=0), axis=1)
    q_dd = np.max(np.abs(np.diff(q, n=2, axis=0)), axis=1)
    return {"root_translation_second_difference_m": _stats(translation_dd),
            "root_rotation_second_difference_rad": _stats(rotation_dd),
            "q_second_difference_rad": _stats(q_dd)}


def q_speed_report(q, measured_q, times_ns, exec_start_ns, limit_rad_s=1.0):
    q = np.asarray(q, dtype=float)
    measured_q = np.asarray(measured_q, dtype=float)
    times = np.asarray(times_ns, dtype=np.int64)
    augmented_q = np.concatenate([measured_q[None], q], axis=0)
    augmented_t = np.concatenate([[int(exec_start_ns)], times])
    dt = np.diff(augmented_t) / 1e9
    if np.any(dt <= 0):
        raise ValueError("q speed timestamps must follow exec_start")
    speed = np.abs(np.diff(augmented_q, axis=0)) / dt.reshape((-1,) + (1,) * (q.ndim - 1))
    peak = float(np.max(speed))
    flat = int(np.argmax(speed))
    location = np.unravel_index(flat, speed.shape)
    return {"peak_rad_s": peak, "within_limit": bool(peak <= limit_rad_s + 1e-12),
            "limit_rad_s": limit_rad_s, "peak_step": int(location[0]),
            "peak_component": [int(value) for value in location[1:]],
            "includes_measured_to_step0": True, "per_step_peak_rad_s":
            np.max(speed.reshape(len(speed), -1), axis=1).tolist()}


def eef_speed_report(eef, measured_eef, times_ns, exec_start_ns):
    eef = _transforms(eef)
    measured_eef = _transforms(np.asarray(measured_eef)[None], "measured_eef")[0]
    augmented = np.concatenate([measured_eef[None], eef], axis=0)
    times = np.concatenate([[int(exec_start_ns)], np.asarray(times_ns, np.int64)])
    dt = np.diff(times) / 1e9
    translation, rotation = pose_delta(augmented[:-1], augmented[1:])
    return {"linear_peak_m_s": float(np.max(translation / dt)),
            "angular_peak_rad_s": float(np.max(rotation / dt)),
            "includes_measured_to_step0": True}


def clamp_q_report(q, measured_q, times_ns, exec_start_ns, limit_rad_s=1.0):
    """Return a diagnostic-only sequential q projection and clip statistics."""
    q = np.asarray(q, dtype=float)
    output = q.copy()
    previous = np.asarray(measured_q, dtype=float).copy()
    previous_time = int(exec_start_ns)
    clipped_steps = 0
    max_clip = 0.0
    clipped_values = 0
    for index, timestamp in enumerate(np.asarray(times_ns, np.int64)):
        bound = limit_rad_s * ((int(timestamp) - previous_time) / 1e9)
        delta = output[index] - previous
        projected = previous + np.clip(delta, -bound, bound)
        amount = np.abs(projected - output[index])
        if np.any(amount > 1e-12):
            clipped_steps += 1
            clipped_values += int(np.count_nonzero(amount > 1e-12))
            max_clip = max(max_clip, float(np.max(amount)))
        output[index] = projected
        previous = projected
        previous_time = int(timestamp)
    return output, {"diagnostic_only": True, "execution_ready": False,
                    "clipped_steps": clipped_steps,
                    "clipped_values": clipped_values,
                    "max_clip_rad": max_clip,
                    "limit_rad_s": limit_rad_s}


def keypoint_offset_report(predicted_first, current, measured_roots):
    predicted_first = np.asarray(predicted_first, dtype=float)
    current = np.asarray(current, dtype=float)
    roots = np.asarray(measured_roots, dtype=float)
    if predicted_first.shape != (2, 21, 3) or current.shape != predicted_first.shape:
        raise ValueError("keypoint arrays must be 2x21x3")
    sides = []
    for side in range(2):
        inverse = np.linalg.inv(roots[side])
        p = (predicted_first[side] @ inverse[:3, :3].T + inverse[:3, 3])
        c = (current[side] @ inverse[:3, :3].T + inverse[:3, 3])
        delta = p - c
        sides.append({"per_keypoint_local_delta_m": delta.tolist(),
                      "norm_mean_m": float(np.mean(np.linalg.norm(delta, axis=1))),
                      "norm_std_m": float(np.std(np.linalg.norm(delta, axis=1))),
                      "norm_max_m": float(np.max(np.linalg.norm(delta, axis=1)))})
    return {"left": sides[0], "right": sides[1]}


def wrist_extrapolation_report(predicted, current_wrist, step_s=1.0 / 30.0):
    predicted = np.asarray(predicted, dtype=float)
    current_wrist = np.asarray(current_wrist, dtype=float)
    if predicted.shape[0] < 4 or predicted.shape[1:] != (2, 21, 3):
        raise ValueError("predicted keypoints need at least four 2x21 frames")
    t = np.arange(1, 5, dtype=float) * step_s
    output = {}
    for side, name in enumerate(("left", "right")):
        coefficients = np.polyfit(t, predicted[:4, side, 0], 1)
        intercept = coefficients[1]
        residual = intercept - current_wrist[side]
        output[name] = {"extrapolated_t0_m": intercept.tolist(),
                        "residual_m": residual.tolist(),
                        "residual_norm_m": float(np.linalg.norm(residual))}
    return output
