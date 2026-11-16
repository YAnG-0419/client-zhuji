"""Timeline shaping and chunk splicing for the zhuji XFlow client.

Pure NumPy: no ROS, socket, or robot imports.  Two independent pieces, each
switchable from the config so runs can be compared:

``shape_timeline``
    The planner's per-step durations only cap *velocity*, so a chunk leaves
    rest at full speed and stops dead at the end.  This keeps the planned
    waypoints and changes only *when* each one is reached: an ease-in/ease-out
    speed profile over ``ramp_s``, then one uniform slow-down so the
    finite-difference velocities and accelerations of every stream stay inside
    their caps (velocity scales with 1/k, acceleration with 1/k^2).

``build_splice``
    Prefetch: the next observation is taken while the current chunk is still
    running, so by the time its plan is ready the robot has already covered
    part of it.  Points already covered (in model time) are dropped, the rest
    is shaped, and a decaying cubic-Hermite offset makes the new segment leave
    the running schedule at the same position and velocity.

Adapted from the structure of YAnG-0419/client-franka-sharpa (prefetch.py,
chunk_timing.py), re-derived for this client's SDK-EEF + q20 streams.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.spatial.transform import Rotation


SIDES = ("left", "right")


def _smoothstep(x):
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def speed_profile(u, duration, ramp_s, floor, *, ease_in=True, ease_out=True):
    """Speed ratio in [floor, 1]: smoothstep up over ramp_s and back down."""
    u = np.asarray(u, dtype=float)
    ramp = min(float(ramp_s), 0.5 * float(duration))
    if ramp <= 0.0:
        return np.ones_like(u)
    up = np.clip(u / ramp, 0.0, 1.0) if ease_in else np.ones_like(u)
    down = np.clip((duration - u) / ramp, 0.0, 1.0) if ease_out else np.ones_like(u)
    return floor + (1.0 - floor) * _smoothstep(np.minimum(up, down))


def _acceleration(dt, velocity, rest_start, rest_end):
    """Peak |acceleration| from per-segment velocities, counting rest edges."""
    rows = [velocity]
    mid = [0.5 * (dt[1:] + dt[:-1])]
    if rest_start:
        rows.insert(0, np.zeros((1, velocity.shape[1])))
        mid.insert(0, [0.5 * dt[0]])
    if rest_end:
        rows.append(np.zeros((1, velocity.shape[1])))
        mid.append([0.5 * dt[-1]])
    full = np.vstack(rows)
    spacing = np.concatenate([np.atleast_1d(np.asarray(part, dtype=float)) for part in mid])
    if not len(spacing):
        return 0.0
    return float(np.max(np.abs(np.diff(full, axis=0) / spacing[:, None])))


def joint_rates(times, values, *, rest_start, rest_end):
    """Peak |velocity| and |acceleration| of a [N,...] joint stream."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float).reshape(len(times), -1)
    dt = np.diff(times)
    velocity = np.diff(values, axis=0) / dt[:, None]
    return float(np.max(np.abs(velocity))), _acceleration(dt, velocity, rest_start, rest_end)


def pose_rates(times, eef, *, rest_start, rest_end):
    """Peak linear/angular velocity and acceleration of a [N,2,7] EEF stream."""
    times = np.asarray(times, dtype=float)
    eef = np.asarray(eef, dtype=float)
    dt = np.diff(times)
    linear = np.linalg.norm(np.diff(eef[:, :, :3], axis=0), axis=2) / dt[:, None]
    angular = np.empty_like(linear)
    for side in range(eef.shape[1]):
        rotations = Rotation.from_quat(eef[:, side, 3:])
        angular[:, side] = (rotations[:-1].inv() * rotations[1:]).magnitude() / dt
    return tuple((float(np.max(speed)), _acceleration(dt, speed, rest_start, rest_end))
                 for speed in (linear, angular))


@dataclass(frozen=True)
class Shaped:
    times: np.ndarray          # [N] seconds after the chunk start
    slowdown: float
    nominal_duration_s: float
    duration_s: float
    peaks: dict
    eased: bool


def shape_timeline(base_durations, *, arm_q, hand_q, eef, caps,
                   ramp_s=0.25, speed_floor=0.2, ease_in=True, ease_out=True,
                   max_slowdown=4.0):
    """Warp per-step durations so the chunk respects velocity AND acceleration.

    ``base_durations`` are the planner's already velocity-capped step lengths.
    ``arm_q``/``hand_q``/``eef`` are the commanded streams including the start
    point, so each has ``len(base_durations) + 1`` rows.
    """
    base = np.asarray(base_durations, dtype=float)
    if base.ndim != 1 or len(base) < 1 or np.any(base <= 0) or not np.isfinite(base).all():
        raise ValueError("base_durations must be positive and finite")
    nominal = np.r_[0.0, np.cumsum(base)]
    duration = float(nominal[-1])
    rest_start, rest_end = bool(ease_in), bool(ease_out)
    if ease_in or ease_out:
        mid = 0.5 * (nominal[1:] + nominal[:-1])
        warped = np.r_[0.0, np.cumsum(np.diff(nominal) / speed_profile(
            mid, duration, ramp_s, speed_floor, ease_in=ease_in, ease_out=ease_out))]
    else:
        warped = nominal.copy()

    arm_q = np.asarray(arm_q, dtype=float)
    hand_q = np.asarray(hand_q, dtype=float)
    eef = np.asarray(eef, dtype=float)
    if not (len(arm_q) == len(hand_q) == len(eef) == len(warped)):
        raise ValueError("streams must include the start point")

    arm_v, arm_a = joint_rates(warped, arm_q, rest_start=rest_start, rest_end=rest_end)
    hand_v, hand_a = joint_rates(warped, hand_q, rest_start=rest_start, rest_end=rest_end)
    (lin_v, lin_a), (ang_v, ang_a) = pose_rates(warped, eef, rest_start=rest_start, rest_end=rest_end)
    ratios = [
        arm_v / caps["arm_cap"], hand_v / caps["hand_cap"],
        lin_v / caps["linear_cap"], ang_v / caps["angular_cap"],
        np.sqrt(arm_a / caps["arm_accel_cap"]), np.sqrt(hand_a / caps["hand_accel_cap"]),
        np.sqrt(lin_a / caps["linear_accel_cap"]), np.sqrt(ang_a / caps["angular_accel_cap"]),
    ]
    slowdown = float(max(1.0, max(ratios)))
    if slowdown > max_slowdown:
        raise ValueError("chunk needs a %.1fx slow-down (limit %.1fx); refusing"
                         % (slowdown, max_slowdown))
    times = warped * slowdown
    arm_v, arm_a = joint_rates(times, arm_q, rest_start=rest_start, rest_end=rest_end)
    hand_v, hand_a = joint_rates(times, hand_q, rest_start=rest_start, rest_end=rest_end)
    (lin_v, lin_a), (ang_v, ang_a) = pose_rates(times, eef, rest_start=rest_start, rest_end=rest_end)
    peaks = {"arm_rad_s": arm_v, "arm_rad_s2": arm_a, "hand_rad_s": hand_v,
             "hand_rad_s2": hand_a, "linear_m_s": lin_v, "linear_m_s2": lin_a,
             "angular_rad_s": ang_v, "angular_rad_s2": ang_a}
    return Shaped(times=times[1:], slowdown=slowdown, nominal_duration_s=duration,
                  duration_s=float(times[-1]), peaks=peaks,
                  eased=bool(ease_in or ease_out))


def interpolate_eef(times, eef, query):
    """Linear position + shortest-arc slerp orientation at ``query`` seconds."""
    times = np.asarray(times, dtype=float)
    eef = np.asarray(eef, dtype=float)
    query = np.atleast_1d(np.asarray(query, dtype=float))
    out = np.empty((len(query), eef.shape[1], 7))
    clipped = np.clip(query, times[0], times[-1])
    index = np.clip(np.searchsorted(times, clipped, side="right") - 1, 0, len(times) - 2)
    weight = (clipped - times[index]) / np.maximum(times[index + 1] - times[index], 1e-12)
    for side in range(eef.shape[1]):
        out[:, side, :3] = (eef[index, side, :3]
                            + weight[:, None] * (eef[index + 1, side, :3] - eef[index, side, :3]))
        for k, (i, w) in enumerate(zip(index, weight)):
            q0, q1 = eef[i, side, 3:], eef[i + 1, side, 3:]
            if np.dot(q0, q1) < 0:
                q1 = -q1
            value = q0 + w * (q1 - q0)
            out[k, side, 3:] = value / np.linalg.norm(value)
    return out


def interpolate_rows(times, values, query):
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    query = np.atleast_1d(np.asarray(query, dtype=float))
    flat = values.reshape(len(times), -1)
    out = np.column_stack([np.interp(query, times, flat[:, j]) for j in range(flat.shape[1])])
    return out.reshape((len(query),) + values.shape[1:])


@dataclass(frozen=True)
class Splice:
    times: np.ndarray        # [N] seconds after dispatch
    model_t: np.ndarray      # [N] model seconds since the new plan's sample
    eef: np.ndarray          # [N,2,7]
    q20: np.ndarray          # [N,2,20]
    progress_model_s: float
    skipped_points: int
    slowdown: float
    peaks: dict
    blend_s: float


def build_splice(*, progress_model_s, plan_model_t, plan_eef, plan_q20, plan_q16,
                 current_eef, current_q20, current_eef_velocity, caps, moving=True,
                 ramp_s=0.25, speed_floor=0.2, blend_s=0.2, max_slowdown=4.0):
    """Continue a running chunk with a plan sampled while it was executing.

    ``progress_model_s`` is how far, in the new plan's model time, the robot has
    already travelled since that plan's observation.  ``moving`` says whether the
    robot is still executing at the splice point: if it is, the blend matches the
    running velocity; if not, the segment eases in from rest instead.  Returns
    ``None`` when nothing useful is left or the caps cannot be met.
    """
    model = np.asarray(plan_model_t, dtype=float)
    eef = np.asarray(plan_eef, dtype=float)
    q20 = np.asarray(plan_q20, dtype=float)
    q16 = np.asarray(plan_q16, dtype=float)
    if len(model) < 2:
        return None
    min_gap = 0.25 * float(np.median(np.diff(model)))
    keep = model > progress_model_s + min_gap
    if np.count_nonzero(keep) < 2:
        return None
    # Segment on model time, starting from where the plan says the robot is now.
    seg_model = np.r_[progress_model_s, model[keep]]
    seg_eef = np.vstack([interpolate_eef(model, eef, progress_model_s), eef[keep]])
    seg_q20 = np.vstack([interpolate_rows(model, q20, progress_model_s), q20[keep]])
    seg_q16 = np.vstack([interpolate_rows(model, q16, progress_model_s), q16[keep]])

    # Decaying cubic-Hermite correction: start at the running schedule's pose
    # and velocity, then follow the new plan.
    base = np.diff(seg_model)
    try:
        shaped = shape_timeline(base, arm_q=seg_q16[:, :14], hand_q=seg_q20, eef=seg_eef,
                                caps=caps, ramp_s=ramp_s, speed_floor=speed_floor,
                                ease_in=not moving, ease_out=True,
                                max_slowdown=max_slowdown)
    except ValueError:
        return None   # caller falls back to letting the running chunk finish
    times = np.r_[0.0, shaped.times]
    window = min(blend_s * shaped.slowdown, float(times[-1]))
    s = np.clip(times / window, 0.0, 1.0) if window > 0 else np.zeros_like(times)
    h00 = (1.0 - 3.0 * s ** 2 + 2.0 * s ** 3)[:, None]
    h10 = ((s - 2.0 * s ** 2 + s ** 3) * window)[:, None]

    position_offset = np.asarray(current_eef, float)[:, :3] - seg_eef[0, :, :3]
    if moving:
        segment_velocity = (seg_eef[1, :, :3] - seg_eef[0, :, :3]) / max(times[1], 1e-9)
        velocity_offset = np.asarray(current_eef_velocity, float) - segment_velocity
    else:
        # Starting from rest the ease-in already ramps the speed up; forcing the
        # plan's own velocity to zero on top of that would spike the acceleration.
        velocity_offset = np.zeros((seg_eef.shape[1], 3))
    out_eef = seg_eef.copy()
    out_eef[:, :, :3] += h00[:, :, None] * position_offset + h10[:, :, None] * velocity_offset
    for side in range(seg_eef.shape[1]):
        start = Rotation.from_quat(np.asarray(current_eef, float)[side, 3:])
        delta = (Rotation.from_quat(seg_eef[0, side, 3:]).inv() * start).as_rotvec()
        corrections = Rotation.from_rotvec(h00[:, 0, None] * delta)
        out_eef[:, side, 3:] = (Rotation.from_quat(seg_eef[:, side, 3:])
                                * corrections).as_quat()
    out_q20 = seg_q20 + h00[:, :, None] * (np.asarray(current_q20, float) - seg_q20[0])

    rates = dict(rest_start=not moving, rest_end=True)
    arm_v, arm_a = joint_rates(times, seg_q16[:, :14], **rates)
    hand_v, hand_a = joint_rates(times, out_q20, **rates)
    (lin_v, lin_a), (ang_v, ang_a) = pose_rates(times, out_eef, **rates)
    peaks = {"arm_rad_s": arm_v, "arm_rad_s2": arm_a, "hand_rad_s": hand_v,
             "hand_rad_s2": hand_a, "linear_m_s": lin_v, "linear_m_s2": lin_a,
             "angular_rad_s": ang_v, "angular_rad_s2": ang_a}
    over = [name for name, value, cap in (
        ("hand_rad_s", hand_v, caps["hand_cap"]), ("linear_m_s", lin_v, caps["linear_cap"]),
        ("angular_rad_s", ang_v, caps["angular_cap"]),
        ("hand_rad_s2", hand_a, caps["hand_accel_cap"]),
        ("linear_m_s2", lin_a, caps["linear_accel_cap"]),
        ("angular_rad_s2", ang_a, caps["angular_accel_cap"])) if value > cap * 1.05]
    if over:
        return None
    return Splice(times=times[1:], model_t=np.interp(times[1:], times, seg_model),
                  eef=out_eef[1:], q20=out_q20[1:],
                  progress_model_s=float(progress_model_s),
                  skipped_points=int(np.count_nonzero(~keep)), slowdown=shaped.slowdown,
                  peaks=peaks, blend_s=float(window))
