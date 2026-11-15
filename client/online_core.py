"""Pure online-deployment planning core; this module never opens robot devices.

Transform convention is ``T_a_b``: a matrix maps coordinates from frame b to
frame a.  Model chunks contain Wuji hand-base poses in the source optical
camera frame.  They are converted to TRON2 SDK EEF targets with the same board
correspondence used by the latest replay conversion.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


SIDES = ("left", "right")


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rigid(value: np.ndarray, name: str = "transform") -> np.ndarray:
    """Return a checked rigid transform (or stack of transforms)."""
    value = np.asarray(value, dtype=float)
    if value.shape[-2:] != (4, 4) or not np.isfinite(value).all():
        raise ValueError(name + " must contain finite 4x4 transforms")
    expected = np.broadcast_to([0.0, 0.0, 0.0, 1.0], value[..., 3, :].shape)
    np.testing.assert_allclose(value[..., 3, :], expected, atol=1e-10, rtol=0)
    rotation = value[..., :3, :3]
    identity = np.broadcast_to(np.eye(3), rotation.shape)
    np.testing.assert_allclose(np.swapaxes(rotation, -1, -2) @ rotation,
                               identity, atol=1e-8, rtol=0)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0)
    return value


def _tool_frames(calibration: dict, axes: np.ndarray, center: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eef_from_board_center = rigid(calibration["T_eef_board"], "T_eef_board").copy()
    eef_from_board_center[:3, 3] += (
        eef_from_board_center[:3, :3] @ np.asarray(center, dtype=float)
    )
    board_center_from_hand = np.eye(4)
    board_center_from_hand[:3, :3] = np.asarray(axes, dtype=float)
    return (rigid(eef_from_board_center, "T_sdk_eef_board_center"),
            rigid(board_center_from_hand, "T_board_center_hand"))


@dataclass(frozen=True)
class DeploymentMapping:
    base_from_source_camera: np.ndarray
    sdk_eef_from_hand: Dict[str, np.ndarray]
    coordinate_origin_distance_m: Dict[str, float]
    operator_adapter_length_m: float
    limits: Dict[str, np.ndarray]
    provenance: Dict[str, str]
    report: dict

    @classmethod
    def load(cls, root: Path, config_path: Path,
             camera_mapping: Optional[str] = None) -> "DeploymentMapping":
        root = Path(root).resolve()
        config_path = Path(config_path).resolve()
        config = _json(config_path)
        if config.get("schema") != "pengyang_online_eef_client_v1":
            raise ValueError("unsupported deployment config schema")
        paths = {key: (root / value).resolve() for key, value in config["paths"].items()}
        mapping_config = config["camera_mapping"]
        mode = camera_mapping or mapping_config["default"]
        if mode not in ("live_zhuji_camera", "board_transfer"):
            raise ValueError("camera_mapping must be live_zhuji_camera or board_transfer")
        source = _json(paths["source_observation"])
        target = _json(paths["target_observation"])
        camera = _json(paths["target_camera_calibration"])
        for key in ("board_fingerprint", "board_config", "board_axes", "convention"):
            if source[key] != target[key]:
                raise ValueError("source/target observation mismatch: " + key)
        base_from_target_camera = rigid(camera["T_base_camera"], "T_base_camera")
        if mode == "live_zhuji_camera":
            # The online keypoints already live in the zhuji optical frame.
            # Applying the calibration-board source transfer here would be a
            # second, incorrect camera transform.
            target_camera_from_source_camera = np.eye(4)
        else:
            source_camera_from_board = rigid(source["camera_T_board"], "source camera_T_board")
            target_camera_from_board = rigid(target["camera_T_board"], "target camera_T_board")
            target_camera_from_source_camera = rigid(
                target_camera_from_board @ np.linalg.inv(source_camera_from_board),
                "target_camera_T_source_camera",
            )
        base_from_source_camera = rigid(
            base_from_target_camera @ target_camera_from_source_camera,
            "base_T_source_camera",
        )

        alignment = _json(paths["alignment_config"])
        limits_json = _json(paths["joint_limits"])
        adapter_length = float(config["adapter_annotation"]["approximate_length_m"])
        if not np.isfinite(adapter_length) or not 0.001 <= adapter_length <= 0.25:
            raise ValueError("adapter annotation must be within [0.001, 0.25] m")
        if config["adapter_annotation"].get("apply_as_transform") is not False:
            raise ValueError("physical adapter annotation must not override calibrated transforms")
        transforms: Dict[str, np.ndarray] = {}
        original: Dict[str, float] = {}
        tool_report = {}
        for side in SIDES:
            calibration_path = paths[side + "_tool_calibration"]
            eef_from_board, board_from_hand = _tool_frames(
                _json(calibration_path),
                alignment["sides"][side]["R_board_center_hand"],
                alignment["board_center_in_board_m"],
            )
            # The approximate 5 cm adapter is a physical gap annotation, not the
            # distance between these coordinate origins.  Replay uses this full
            # calibrated transform unchanged and must not add or rescale 5 cm.
            eef_from_hand = rigid(eef_from_board @ board_from_hand, "T_sdk_eef_hand")
            measured = float(np.linalg.norm(eef_from_hand[:3, 3]))
            transforms[side] = eef_from_hand
            original[side] = measured
            tool_report[side] = {
                "calibrated_coordinate_origin_distance_m": measured,
                "T_sdk_eef_hand": eef_from_hand.tolist(),
            }
        limits = {side: np.asarray(limits_json[side], dtype=float) for side in SIDES}
        if any(limits[side].shape != (20, 2) for side in SIDES):
            raise ValueError("Wuji limits must be 20x2 per hand")
        provenance = {str(config_path): _sha256(config_path)}
        provenance.update({str(path): _sha256(path) for path in paths.values()})
        report = {
            "camera_mapping_mode": mode,
            "source_camera_id": source["camera_id"],
            "target_camera_id": target["camera_id"],
            "board_fingerprint": source["board_fingerprint"],
            "target_camera_T_source_camera": target_camera_from_source_camera.tolist(),
            "T_tron_base_source_camera": base_from_source_camera.tolist(),
            "adapter_annotation": config["adapter_annotation"],
            "transform_policy": config["transform_policy"],
            "tools": tool_report,
            "physical_validation_complete": False,
        }
        return cls(base_from_source_camera, transforms, original, adapter_length,
                   limits, provenance, report)


@dataclass(frozen=True)
class TimedChunk:
    """Accepted future targets. EEF quaternions are xyzw."""
    times_ns: np.ndarray
    eef: np.ndarray
    hands: np.ndarray
    metadata: dict

    def __post_init__(self) -> None:
        times = np.asarray(self.times_ns, dtype=np.int64)
        eef = np.asarray(self.eef, dtype=float)
        hands = np.asarray(self.hands, dtype=float)
        if (times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0)
                or eef.shape != (len(times), 2, 7)
                or hands.shape != (len(times), 2, 20)
                or not np.isfinite(eef).all() or not np.isfinite(hands).all()):
            raise ValueError("invalid timed EEF/Wuji chunk")
        norms = np.linalg.norm(eef[:, :, 3:], axis=2)
        if not np.allclose(norms, 1.0, atol=1e-6, rtol=0):
            raise ValueError("EEF quaternion is not unit length")
        object.__setattr__(self, "times_ns", times.copy())
        object.__setattr__(self, "eef", eef.copy())
        object.__setattr__(self, "hands", hands.copy())

    def sample(self, query_ns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        query = np.atleast_1d(np.asarray(query_ns, dtype=np.int64))
        if np.any(query < self.times_ns[0]) or np.any(query > self.times_ns[-1]):
            raise ValueError("chunk sampling would extrapolate or hold stale targets")
        index = np.clip(np.searchsorted(self.times_ns, query, side="right") - 1,
                        0, len(self.times_ns) - 2)
        weight = ((query - self.times_ns[index]) /
                  (self.times_ns[index + 1] - self.times_ns[index])).astype(float)
        eef = self.eef[index] * (1.0 - weight[:, None, None]) + self.eef[index + 1] * weight[:, None, None]
        hands = (self.hands[index] * (1.0 - weight[:, None, None])
                 + self.hands[index + 1] * weight[:, None, None])
        for side in range(2):
            rotations = Slerp(
                (self.times_ns - self.times_ns[0]) / 1e9,
                Rotation.from_quat(self.eef[:, side, 3:]),
            )((query - self.times_ns[0]) / 1e9)
            eef[:, side, 3:] = rotations.as_quat()
        return eef, hands


def _continuous_quaternions(matrices: np.ndarray) -> np.ndarray:
    quaternions = Rotation.from_matrix(matrices).as_quat()
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index], quaternions[index - 1]) < 0:
            quaternions[index] *= -1
    return quaternions


def convert_camera_npz(path: Path, mapping: DeploymentMapping,
                       origin_ns: Optional[int] = None) -> TimedChunk:
    """Convert one model/recorded camera-frame chunk into TRON2 SDK EEF targets."""
    path = Path(path).resolve()
    with np.load(path, allow_pickle=False) as data:
        required_scalars = {
            "frame": "undistorted_color_optical",
            "pose_target": "Wuji l_base_link / r_base_link; not FR3 flange",
            "quaternion_order": "wxyz",
            "position_unit": "metre",
            "joint_unit": "radian",
        }
        for key, expected in required_scalars.items():
            if str(data[key]) != expected:
                raise ValueError("unsupported chunk " + key)
        if data["side_order"].tolist() != ["left", "right"]:
            raise ValueError("chunk side order must be left, right")
        relative_s = np.asarray(data["time_s"], dtype=float)
        stamps = np.asarray(data["timestamp_ns"], dtype=np.int64)
        if (relative_s.ndim != 1 or len(relative_s) < 2 or relative_s[0] != 0
                or not np.isfinite(relative_s).all() or np.any(np.diff(relative_s) <= 0)
                or stamps.shape != relative_s.shape or np.any(np.diff(stamps) <= 0)):
            raise ValueError("invalid chunk clock")
        np.testing.assert_allclose(relative_s, (stamps - stamps[0]) / 1e9,
                                   atol=1e-9, rtol=0)
        if origin_ns is not None:
            stamps = int(origin_ns) + np.rint(relative_s * 1e9).astype(np.int64)
        eef = np.empty((len(stamps), 2, 7), dtype=float)
        hands = np.empty((len(stamps), 2, 20), dtype=float)
        root_poses = []
        for side_index, side in enumerate(SIDES):
            camera_from_hand = rigid(data[side + "_root_pose"], side + " root pose")
            q = np.asarray(data[side + "_joint_positions_rad"], dtype=float)
            if q.shape != (len(stamps), 20) or not np.isfinite(q).all():
                raise ValueError(side + " hand qpos shape/nonfinite")
            bounds = mapping.limits[side]
            if np.any(q < bounds[:, 0] - 1e-8) or np.any(q > bounds[:, 1] + 1e-8):
                raise ValueError(side + " hand target outside installed limits")
            base_from_hand = rigid(mapping.base_from_source_camera @ camera_from_hand,
                                   "base_T_" + side + "_hand")
            base_from_eef = rigid(base_from_hand @ np.linalg.inv(mapping.sdk_eef_from_hand[side]),
                                  "base_T_" + side + "_sdk_eef")
            closure = base_from_eef @ mapping.sdk_eef_from_hand[side]
            np.testing.assert_allclose(closure, base_from_hand, atol=1e-9, rtol=0)
            eef[:, side_index, :3] = base_from_eef[:, :3, 3]
            eef[:, side_index, 3:] = _continuous_quaternions(base_from_eef[:, :3, :3])
            hands[:, side_index] = q
            root_poses.append(base_from_hand)
        roots = np.stack(root_poses, axis=1)
        source_left = rigid(data["left_root_pose"], "source left root")
        source_right = rigid(data["right_root_pose"], "source right root")
        source_relative = np.linalg.inv(source_left) @ source_right
        target_relative = np.linalg.inv(roots[:, 0]) @ roots[:, 1]
        relative_error = float(np.max(np.abs(source_relative - target_relative)))
        if relative_error > 1e-9:
            raise ValueError("bimanual relative pose was not preserved")
        metadata = {
            "schema": "pengyang_online_eef_chunk_v1",
            "source": str(path),
            "source_sha256": _sha256(path),
            "frame": "tron2_base_link",
            "eef_frame": "TRON2 SDK measured/commanded EEF",
            "hand_frame": "Wuji l_base_link/r_base_link",
            "quaternion_order": "xyzw",
            "operator_adapter_length_m": mapping.operator_adapter_length_m,
            "adapter_length_semantics": "approximate physical flange/gap; annotation only",
            "calibrated_coordinate_origin_distance_m": mapping.coordinate_origin_distance_m,
            "transform_policy": "use calibrated SDK EEF to Wuji hand-origin transform unchanged; add no 5 cm offset",
            "bimanual_relative_matrix_error_max": relative_error,
            "mapping": mapping.report,
            "provenance": mapping.provenance,
            "execution_ready": False,
            "hardware_commands_sent": 0,
        }
        return TimedChunk(stamps, eef, hands, metadata)


def resample_chunk(chunk: TimedChunk, period_ns: int) -> TimedChunk:
    if not isinstance(period_ns, int) or period_ns <= 0:
        raise ValueError("positive integer resample period required")
    times = np.arange(chunk.times_ns[0], chunk.times_ns[-1], period_ns, dtype=np.int64)
    times = np.unique(np.r_[times, chunk.times_ns[-1]])
    eef, hands = chunk.sample(times)
    metadata = dict(chunk.metadata, resample_period_ns=period_ns,
                    original_frames=len(chunk.times_ns), resampled_frames=len(times))
    return TimedChunk(times, eef, hands, metadata)


def _pair_slerp(q0: np.ndarray, q1: np.ndarray, weight: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    denominator = np.sin(theta)
    near = dot > 0.9995
    safe = np.where(near, 1.0, denominator)
    w = weight[..., None]
    result = np.where(
        near,
        q0 * (1.0 - w) + q1 * w,
        (np.sin((1.0 - w) * theta) * q0 + np.sin(w * theta) * q1) / safe,
    )
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def replace_future(old: TimedChunk, new: TimedChunk, switch_ns: int,
                   blend_ns: int, period_ns: int) -> TimedChunk:
    """Blend only future, same-clock targets; never execute a stale prediction."""
    if blend_ns <= 0 or period_ns <= 0 or switch_ns < old.times_ns[0]:
        raise ValueError("invalid handoff timing")
    end_blend = switch_ns + blend_ns
    if end_blend > old.times_ns[-1] or end_blend >= new.times_ns[-1]:
        raise ValueError("both chunks must cover the complete blend and useful new tail")
    if switch_ns < new.times_ns[0]:
        raise ValueError("new chunk does not cover the handoff")
    times = np.arange(switch_ns, new.times_ns[-1], period_ns, dtype=np.int64)
    times = np.unique(np.r_[times, new.times_ns[-1]])
    eef, hands = new.sample(times)
    mask = times <= end_blend
    old_eef, old_hands = old.sample(times[mask])
    u = (times[mask] - switch_ns) / float(blend_ns)
    weight = u ** 3 * (10.0 + u * (-15.0 + 6.0 * u))
    eef[mask, :, :3] = (old_eef[:, :, :3] * (1.0 - weight[:, None, None])
                              + eef[mask, :, :3] * weight[:, None, None])
    eef[mask, :, 3:] = _pair_slerp(old_eef[:, :, 3:], eef[mask, :, 3:], weight[:, None])
    hands[mask] = (old_hands * (1.0 - weight[:, None, None])
                   + hands[mask] * weight[:, None, None])
    metadata = dict(new.metadata, handoff={"switch_ns": int(switch_ns),
                                           "blend_ns": int(blend_ns),
                                           "period_ns": int(period_ns),
                                           "weight": "quintic smoothstep"})
    return TimedChunk(times, eef, hands, metadata)


class AsyncPlanner:
    """One outstanding planning job. The worker never publishes robot targets."""

    def __init__(self, plan: Callable[[Any, int], Any]):
        self._plan = plan
        self._job: Optional[dict] = None
        self._thread: Optional[threading.Thread] = None

    def submit(self, payload: Any, deadline_ns: int) -> None:
        if self._job is not None:
            raise RuntimeError("only one outstanding online planning request is allowed")
        job = {"done": threading.Event(), "submitted_ns": time.monotonic_ns(),
               "deadline_ns": int(deadline_ns)}
        self._job = job

        def run() -> None:
            try:
                result = self._plan(payload, int(deadline_ns))
                ready = time.monotonic_ns()
                if ready >= deadline_ns:
                    raise TimeoutError("online planning missed its handoff deadline")
                job.update(result=result, ready_ns=ready,
                           planning_s=(ready - job["submitted_ns"]) / 1e9)
            except BaseException as error:  # delivered to the single control owner
                job["error"] = error
            finally:
                job["done"].set()

        self._thread = threading.Thread(target=run, name="pengyang-online-plan", daemon=True)
        self._thread.start()

    def poll(self) -> Optional[dict]:
        if self._job is None or not self._job["done"].is_set():
            return None
        job, self._job = self._job, None
        if "error" in job:
            raise job["error"]
        return job

    def close(self, timeout_s: float = 5.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                raise RuntimeError("online planning worker did not stop")

