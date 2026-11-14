"""Streaming MediaPipe-21 to Wuji Hand 2 retargeting; pure computation only."""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Optional

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


HERE = Path(__file__).resolve().parent
VENDOR = HERE / "vendor/keypoints21_to_wujihand2"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
os.environ.setdefault("MUJOCO_GL", "disable")

from retarget import Hand, PARENTS, TIPS, rigid_align  # noqa: E402
from convert import normalized_goals  # noqa: E402


# These are the five MediaPipe palm-base correspondences used by the original
# importer.  URDF inspection shows that they are *not* rigid in Wuji: every one
# is downstream of its finger's first flex joint.  Only point 0 is rigid, which
# is insufficient to estimate a 6-DoF pose.  The fixed-root mode therefore
# aligns the current-q FK positions (and refines q/root alternately), rather
# than incorrectly treating neutral-FK positions as rigid landmarks.
PALM_ALIGNMENT_POINTS = np.array([0, 1, 5, 9, 13, 17], dtype=int)
# The original free-root initializer uses the five finger-base points.  Keep
# that behavior separate from the q-conditioned fixed-root alignment above.
FREE_ROOT_INITIAL_POINTS = np.array([1, 5, 9, 13, 17], dtype=int)
RIGID_POINTS = FREE_ROOT_INITIAL_POINTS  # Backward-compatible public name.
MODES = ("free_root", "kabsch_fixed_root")


def _rigid(value, name="pose"):
    value = np.asarray(value, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(name + " must be one finite 4x4 transform")
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-10, rtol=0):
        raise ValueError(name + " has invalid homogeneous row")
    rotation = value[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0)
            or np.linalg.det(rotation) < 0.999999):
        raise ValueError(name + " has invalid rotation")
    return value


def _pose(rotation, translation):
    value = np.eye(4)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


def _pose_delta(a, b):
    translation = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rotation = float(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude())
    return translation, rotation


class Wuji2StreamingRetargeter:
    """Stateful per-hand solver. No ROS, sockets, or robot transports."""

    def __init__(self, side, mode="free_root", thresholds=None, max_nfev=300):
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        if mode not in MODES:
            raise ValueError("root mode must be free_root or kabsch_fixed_root")
        self.side = side
        self.mode = mode
        self.hand = Hand("wuji", side)
        self.neutral_points = self.hand.fk(self.hand.neutral)
        self.weights = np.full(21, 0.05, dtype=float)
        self.weights[0] = 0.1
        self.weights[TIPS] = 1.0
        self.sqrt_weights = np.sqrt(self.weights)[:, None]
        self.max_nfev = int(max_nfev)
        if not 10 <= self.max_nfev <= 2000:
            raise ValueError("max_nfev must be within [10, 2000]")
        defaults = dict(max_tip_rmse_m=0.020, max_all_rmse_m=0.015,
                        max_root_keypoint_error_m=0.015, max_joint_step_rad=0.35,
                        max_root_translation_step_m=0.08,
                        max_root_rotation_step_rad=np.deg2rad(45.0))
        defaults.update(thresholds or {})
        if not all(np.isfinite(value) and value > 0 for value in defaults.values()):
            raise ValueError("all retarget thresholds must be positive finite values")
        self.thresholds = defaults
        self.q = None
        self.root = None
        self.last_timestamp_ns = None

    @property
    def joint_names(self):
        return tuple(self.hand.names)

    def begin_chunk(self, seed_q20, seed_cam_T_root=None):
        q = np.asarray(seed_q20, dtype=float)
        if q.shape != (20,) or not np.isfinite(q).all():
            raise ValueError("seed_q20 must contain 20 finite positions")
        if np.any(q < self.hand.lo - 1e-8) or np.any(q > self.hand.hi + 1e-8):
            raise ValueError("seed_q20 is outside the Wuji model limits")
        self.q = np.clip(q, self.hand.lo + 1e-7, self.hand.hi - 1e-7)
        self.root = None if seed_cam_T_root is None else _rigid(seed_cam_T_root, "seed_cam_T_root").copy()
        self.last_timestamp_ns = None

    def _reject(self, reason, elapsed_ms=0.0, **diagnostic):
        result = dict(mode=self.mode, side=self.side, reason=reason,
                      elapsed_ms=float(elapsed_ms))
        result.update(diagnostic)
        return dict(accepted=False, cam_T_root=None, q20=None, diagnostic=result)

    def _free_root(self, goal, temporal):
        if self.root is None:
            rotation, translation = rigid_align(
                self.hand.fk(self.q)[FREE_ROOT_INITIAL_POINTS],
                goal[FREE_ROOT_INITIAL_POINTS])
            root = _pose(rotation, translation)
        else:
            root = self.root
        initial = np.r_[self.q, Rotation.from_matrix(root[:3, :3]).as_rotvec(), root[:3, 3]]
        previous_q = self.q.copy()
        bounds = (np.r_[self.hand.lo, np.full(6, -np.inf)],
                  np.r_[self.hand.hi, np.full(6, np.inf)])

        def fun(value):
            rotation = Rotation.from_rotvec(value[20:23]).as_matrix()
            predicted = self.hand.fk(value[:20]) @ rotation.T + value[23:26]
            return np.r_[((predicted - goal) * self.sqrt_weights).ravel(),
                         temporal * (value[:20] - previous_q)]

        def jac(value):
            rotation = Rotation.from_rotvec(value[20:23]).as_matrix()
            local, joint_jacobian = self.hand.fk(value[:20], True)
            output = np.zeros((83, 26), dtype=float)
            output[:63, :20] = (
                np.einsum("ij,njk->nik", rotation, joint_jacobian)
                * self.sqrt_weights[:, :, None]
            ).reshape(63, 20)
            for axis in range(3):
                delta = np.eye(3)[axis] * 1e-6
                derivative = (Rotation.from_rotvec(value[20:23] + delta).as_matrix()
                              - Rotation.from_rotvec(value[20:23] - delta).as_matrix()) / 2e-6
                output[:63, 20 + axis] = (local @ derivative.T * self.sqrt_weights).ravel()
            output[:63, 23:] = (np.tile(np.eye(3), (21, 1))
                                * np.repeat(self.sqrt_weights, 3, axis=0))
            output[63:, :20] = temporal * np.eye(20)
            return output

        answer = least_squares(fun, initial, jac=jac, bounds=bounds,
                               max_nfev=self.max_nfev, x_scale="jac",
                               ftol=1e-9, xtol=1e-9, gtol=1e-8)
        q = answer.x[:20]
        root = _pose(Rotation.from_rotvec(answer.x[20:23]).as_matrix(), answer.x[23:26])
        return q, root, answer

    def _fixed_root(self, goal, temporal):
        # A single Kabsch against neutral FK is biased whenever the MCP flex
        # joints are non-neutral.  Refine the root from current-q FK, then keep
        # that root fixed while solving q.  Four small alternating iterations
        # are deterministic and retain the intended fixed-root finger solve.
        previous_q = self.q.copy()
        q = self.q.copy()
        answer = None
        root = None
        for _ in range(4):
            local = self.hand.fk(q)
            rotation, translation = rigid_align(
                local[PALM_ALIGNMENT_POINTS], goal[PALM_ALIGNMENT_POINTS])
            root = _pose(rotation, translation)

            def fun(candidate):
                predicted = self.hand.fk(candidate) @ rotation.T + translation
                return np.r_[((predicted - goal) * self.sqrt_weights).ravel(),
                             temporal * (candidate - previous_q)]

            def jac(candidate):
                _, joint_jacobian = self.hand.fk(candidate, True)
                output = np.zeros((83, 20), dtype=float)
                output[:63] = (np.einsum("ij,njk->nik", rotation, joint_jacobian)
                               * self.sqrt_weights[:, :, None]).reshape(63, 20)
                output[63:] = temporal * np.eye(20)
                return output

            answer = least_squares(fun, q, jac=jac,
                                   bounds=(self.hand.lo, self.hand.hi),
                                   max_nfev=self.max_nfev, x_scale="jac",
                                   ftol=1e-9, xtol=1e-9, gtol=1e-8)
            q = answer.x
        # Keep the returned pose consistent with the final q estimate.
        rotation, translation = rigid_align(
            self.hand.fk(q)[PALM_ALIGNMENT_POINTS],
            goal[PALM_ALIGNMENT_POINTS])
        return q, _pose(rotation, translation), answer

    def retarget_frame(self, keypoints, timestamp_ns, normalize=True):
        began = time.perf_counter()
        if self.q is None:
            return self._reject("begin_chunk must be called first")
        points = np.asarray(keypoints, dtype=float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            return self._reject("expected 21 finite camera-frame keypoints")
        timestamp_ns = int(timestamp_ns)
        if self.last_timestamp_ns is not None and timestamp_ns <= self.last_timestamp_ns:
            return self._reject("timestamp is not strictly increasing within the chunk")
        palm = np.cross(points[5] - points[0], points[9] - points[0])
        if np.linalg.norm(palm) < 1e-8:
            return self._reject("degenerate palm keypoints")
        try:
            goal = normalized_goals(points[None], self.neutral_points)[0] if normalize else points.copy()
            temporal = 1e-4 if self.last_timestamp_ns is None else 0.005
            if self.mode == "free_root":
                q, root, answer = self._free_root(goal, temporal)
            else:
                q, root, answer = self._fixed_root(goal, temporal)
            predicted = self.hand.fk(q) @ root[:3, :3].T + root[:3, 3]
            error = np.linalg.norm(predicted - goal, axis=1)
            joint_step = float(np.max(np.abs(q - self.q)))
            if self.root is None:
                root_translation_step = root_rotation_step = 0.0
            else:
                root_translation_step, root_rotation_step = _pose_delta(root, self.root)
            diagnostic = dict(
                mode=self.mode, side=self.side,
                tip_rmse_m=float(np.sqrt(np.mean(error[TIPS] ** 2))),
                all_rmse_m=float(np.sqrt(np.mean(error ** 2))),
                root_keypoint_error_m=float(error[0]),
                root_to_keypoint0_distance_m=float(
                    np.linalg.norm(predicted[0] - root[:3, 3])),
                joint_step_rad=joint_step,
                root_translation_step_m=root_translation_step,
                root_rotation_step_rad=root_rotation_step,
                nfev=int(answer.nfev), status=int(answer.status),
                optimality=float(answer.optimality),
                elapsed_ms=1000.0 * (time.perf_counter() - began),
                normalized_bone_lengths=bool(normalize),
            )
            checks = (
                (diagnostic["tip_rmse_m"], self.thresholds["max_tip_rmse_m"], "tip RMSE"),
                (diagnostic["all_rmse_m"], self.thresholds["max_all_rmse_m"], "all-point RMSE"),
                (diagnostic["root_keypoint_error_m"], self.thresholds["max_root_keypoint_error_m"], "root keypoint error"),
                (joint_step, self.thresholds["max_joint_step_rad"], "joint step"),
                (root_translation_step, self.thresholds["max_root_translation_step_m"], "root translation step"),
                (root_rotation_step, self.thresholds["max_root_rotation_step_rad"], "root rotation step"),
            )
            failed = ["%s %.6g > %.6g" % (name, value, limit)
                      for value, limit, name in checks if value > limit]
            if failed:
                diagnostic["reason"] = "; ".join(failed)
                return dict(accepted=False, cam_T_root=None, q20=None,
                            diagnostic=diagnostic)
            self.q = np.clip(q, self.hand.lo, self.hand.hi)
            self.root = root.copy()
            self.last_timestamp_ns = timestamp_ns
            diagnostic["reason"] = None
            return dict(accepted=True, cam_T_root=root.copy(), q20=self.q.copy(),
                        diagnostic=diagnostic)
        except Exception as error:
            return self._reject(type(error).__name__ + ": " + str(error),
                                elapsed_ms=1000.0 * (time.perf_counter() - began))


def pose_error(actual, expected):
    actual, expected = _rigid(actual), _rigid(expected)
    return _pose_delta(actual, expected)

