"""Online XFlow chunk planner for zhuji (TRON2 + Wuji Hand 2).

Pure planning: observation -> XFlow inference -> Wuji2 retarget -> anchored,
smoothed SDK EEF + q20 targets -> speed-capped timeline -> IK/collision
certification.  It never imports the robot transport and never publishes ROS
messages; ``live_client.py`` owns the only hardware sender.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from execution_postprocess import (apply_local_anchor,
                                   smooth_q_trajectory, smooth_root_trajectory,
                                   validate_execution_config)
from hand_retarget_wuji2 import Wuji2StreamingRetargeter
from image_preprocess import ImagePreprocessor
from online_core import DeploymentMapping
from proprio import ProprioKinematics, camera_hand_to_sdk_eef
from timing_policy import ModelTiming
from xflow_rpc import XFlowClient, XFlowRemoteError


SIDES = ("left", "right")
EXECUTION_FPS = 30.0
EEF_BOX = np.asarray([[[.250, .732], [-.213, .900], [-.673, .5]],
                      [[.250, .732], [-.900, .213], [-.673, .5]]], dtype=float)


IK_LIMIT_MARGIN = 0.02  # rad, see ChunkPlanner.__init__


class PlanRejected(RuntimeError):
    """The chunk is unsafe or unusable; the executor keeps holding."""


def pose7_matrix(value):
    value = np.asarray(value, dtype=float)
    output = np.eye(4)
    output[:3, :3] = Rotation.from_quat(value[3:]).as_matrix()
    output[:3, 3] = value[:3]
    return output


def matrix_pose7(value):
    return np.r_[value[:3, 3], Rotation.from_matrix(value[:3, :3]).as_quat()]


def smoothstep_peak_rate(steps):
    """Maximum per-step decrement of 1 - smoothstep(k/steps)."""
    return 1.5 / float(steps)


def budget_decay_steps(q_offset, translation_offset_m, rotation_offset_rad, *,
                       minimum=12, maximum=32, dt=1.0 / EXECUTION_FPS,
                       q_budget=0.5, translation_budget=0.05, rotation_budget=0.3):
    """Decay length so the anchor term alone stays inside its speed budget."""
    needed = [minimum]
    for offset, budget in ((float(np.max(np.abs(q_offset))) if np.size(q_offset) else 0.0, q_budget),
                           (float(translation_offset_m), translation_budget),
                           (float(rotation_offset_rad), rotation_budget)):
        if budget <= 0:
            raise ValueError("anchor speed budgets must be positive")
        needed.append(int(math.ceil(1.5 * offset / (budget * dt))))
    steps = max(needed)
    return int(min(steps, maximum)), bool(steps > maximum)


def synchronized_step_times(arm_q, hand_q, eef, start_arm_q, start_hand_q, start_eef, *,
                            arm_cap, hand_cap, linear_cap, angular_cap,
                            base_dt=1.0 / EXECUTION_FPS):
    """One clock for arms and hands: each step is >= 1/30 s, longer if a cap binds.

    Mirrors replay_hand.full_sync.synchronized_timeline (which produced the
    validated replays), extended with SDK EEF linear/angular caps.  Returns
    step times relative to execution start and the per-step binding reason.
    """
    arm = np.vstack([np.asarray(start_arm_q, float)[None], np.asarray(arm_q, float)])
    hand = np.concatenate([np.asarray(start_hand_q, float)[None], np.asarray(hand_q, float)])
    poses = np.concatenate([np.asarray(start_eef, float)[None], np.asarray(eef, float)])
    count = len(arm) - 1
    durations = np.empty(count)
    reasons = []
    for index in range(count):
        candidates = {"frame": float(np.broadcast_to(base_dt, (count,))[index]),
                      "arm": float(np.max(np.abs(arm[index + 1] - arm[index]))) / arm_cap,
                      "hand": float(np.max(np.abs(hand[index + 1] - hand[index]))) / hand_cap}
        linear = angular = 0.0
        for side in range(2):
            linear = max(linear, float(np.linalg.norm(poses[index + 1, side, :3]
                                                      - poses[index, side, :3])))
            relative = (Rotation.from_quat(poses[index, side, 3:]).inv()
                        * Rotation.from_quat(poses[index + 1, side, 3:]))
            angular = max(angular, float(relative.magnitude()))
        candidates["eef_linear"] = linear / linear_cap
        candidates["eef_angular"] = angular / angular_cap
        reason = max(candidates, key=candidates.get)
        durations[index] = candidates[reason]
        reasons.append(reason)
    return np.cumsum(durations), reasons


class ChunkPlanner:
    def __init__(self, root, config_path, *, model, task, relay_host="127.0.0.1",
                 relay_port=28765, image_mode="virtual_cam0", virtual_preset="cam0_newk",
                 arm_cap=0.8, hand_cap=0.9, linear_cap=0.25, angular_cap=1.0,
                 num_inference_steps=10, steps=32, speed_ratio=None):
        self.root = Path(root).resolve()
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text())
        validate_execution_config(self.config)
        self.mapping = DeploymentMapping.load(self.root, self.config_path, "live_zhuji_camera")
        self.kinematics = ProprioKinematics(self.mapping)
        self.camera_T_base = np.linalg.inv(self.mapping.base_from_source_camera)
        self.task = str(task)
        self.model_name = None if model in (None, "", "auto") else str(model)
        self.image_mode, self.virtual_preset = image_mode, virtual_preset
        self.caps = dict(arm_cap=float(arm_cap), hand_cap=float(hand_cap),
                         linear_cap=float(linear_cap), angular_cap=float(angular_cap))
        if not all(np.isfinite(v) and v > 0 for v in self.caps.values()):
            raise ValueError("speed caps must be positive")
        if not 2 <= int(steps) <= 32:
            raise ValueError("steps must be within [2,32]")
        self.steps = int(steps)
        self.num_inference_steps = int(num_inference_steps)
        self.client = XFlowClient(relay_host, relay_port, timeout_s=30.0, model=self.model_name)
        info = self.client.info()
        if self.task not in info["tasks"]:
            raise ValueError("task %s is not served; tasks: %s" % (self.task, sorted(info["tasks"])))
        self.model_name = self.client.model     # resolved default_model when None
        model_info = dict(info["models"][self.model_name])
        model_info.setdefault("fps", 30.0)
        # speed_ratio = training-motion speed / robot speed (server 2026-11-15).
        # Default comes from the server (hh 4, rb 1); history offset and step
        # times both follow it.  Older servers have no speed_ratio: hh -> 4.
        default_ratio = model_info.get("speed_ratio")
        if default_ratio is None:
            default_ratio = 4.0 if model_info.get("contract") == "humanhand" else 1.0
        self.speed_ratio = float(default_ratio if speed_ratio is None else speed_ratio)
        if not 0.25 <= self.speed_ratio <= 16.0:
            raise ValueError("speed_ratio must be within [0.25, 16]")
        self.request_speed_ratio = None if speed_ratio is None else self.speed_ratio
        training_history = model_info.get("training_history_offset_frames")
        if training_history is None:
            training_history = 4 if model_info.get("contract") == "humanhand" else 16
        self.history_frames = int(round(float(training_history) * self.speed_ratio))
        self.timing = ModelTiming.from_info(model_info)
        self.deployment_id = str(model_info["deployment_id"])
        self.client.reset()
        # zhuji has Wuji hands, which match neither training hand (human for hh,
        # Sharpa for rb).  Both contracts therefore use the bone-length-normalized
        # free-root fit; the Sharpa-geometry Kabsch shortcut does not apply here.
        self.root_mode = "free_root"
        self.root_reason = "Wuji hand differs from the %s training hand" % self.timing.contract
        hand = self.config["hand_retarget"]
        self.retargeters = {side: Wuji2StreamingRetargeter(
            side, self.root_mode, thresholds=hand["thresholds"], max_nfev=hand["max_nfev"],
            weights=hand["weights"],
            # A chunk already starts from the measured q20/root, so frame 0 uses
            # the streaming prior too (round-3 fix for the frame0->1 jump).
            temporal_first=hand["temporal_stream"], temporal_stream=hand["temporal_stream"],
            fixed_outer_iterations=hand["fixed_outer_iterations"]) for side in SIDES}
        for path in (self.root / "replay_hand", self.root / "replay", self.root / "tools",
                     self.root / "Pinocchio", self.root):
            if str(path) not in sys.path:
                sys.path.append(str(path))
        from pin_model import PinArms
        from transport import replay_backend
        from hand_body_clearance import HandBodyClearance
        self.pin = PinArms()
        # Plan arm joints at least IK_LIMIT_MARGIN inside the SDK command envelope.
        # ServoP solves its own IK for the streamed EEF and was measured
        # overshooting a saturated joint by ~7 mrad (right wrist roll 0.7576 vs
        # 0.7505); a measured start outside the envelope then fails certification.
        self.pin.lower[:14] = self.pin.lower[:14] + IK_LIMIT_MARGIN
        self.pin.upper[:14] = self.pin.upper[:14] - IK_LIMIT_MARGIN
        _, Model, _ = replay_backend()
        self.collision = Model()
        self.hand_checker = HandBodyClearance(
            self.root / "replay_robot/models/tron2_wuji_board_v1_20261110/robot.urdf")
        self.processor = None
        self.processor_camera = None

    def describe(self):
        return {"model": self.model_name, "deployment_id": self.deployment_id,
                "contract": self.timing.contract, "task": self.task,
                "root_mode": self.root_mode, "root_mode_reason": self.root_reason,
                "image_mode": self.image_mode, "virtual_preset": self.virtual_preset,
                "history_execution_frames": self.history_frames, "caps": self.caps,
                "speed_ratio": self.speed_ratio,
                "num_inference_steps": self.num_inference_steps, "steps": self.steps}

    def _preprocessor(self, camera):
        key = (tuple(np.asarray(camera["K"]).ravel()), tuple(np.asarray(camera["D"]).ravel()))
        if self.processor is None or self.processor_camera != key:
            image_config = self.config["image_preprocess"]
            self.processor = ImagePreprocessor(camera, image_config, self.image_mode,
                                               self.virtual_preset)
            self.processor_camera = key
        return self.processor

    def _infer(self, snapshot, points):
        processor = self._preprocessor(snapshot["camera_info"])
        head = processor.process(snapshot["head_rgb"])
        history = (None if snapshot.get("history_rgb") is None
                   else processor.process(snapshot["history_rgb"]))
        trace = {
            "schema": "xflow_inference_trace_v1",
            "request_created_time_ns": time.time_ns(),
            "sample_ns": int(snapshot["sample_ns"]),
            "task": self.task, "model": self.model_name,
            "deployment_id_expected": self.deployment_id,
            "contract": self.timing.contract,
            "image_space": "rgb320",
            "num_inference_steps": self.num_inference_steps,
            "history_sent": history is not None,
            "head_rgb": head,
            "history_rgb": history,
            "request_keypoints": points.astype(np.float32),
        }
        self.last_trace = trace
        if self.client.session_id is None:
            self.client.reset()
        try:
            response = self.client.infer(task=self.task, head_rgb=head, history_rgb=history,
                                         keypoints=points.astype(np.float32),
                                         sample_ns=int(snapshot["sample_ns"]), image_space="rgb320",
                                         num_inference_steps=self.num_inference_steps,
                                     speed_ratio=self.request_speed_ratio)
        except XFlowRemoteError as error:
            if error.code == "STALE_SESSION":
                self.client.session_id = None
            raise PlanRejected("server error " + str(error))
        trace.update({
            "response_received_time_ns": time.time_ns(),
            "response_request_id": response["request_id"],
            "response_session_id": response["session_id"],
            "response_model": response["model"],
            "response_deployment_id": response["deployment_id"],
            "response_contract": response["contract"],
            "response_task": response["task"],
            "response_round_trip_ms": float(response["round_trip_ms"]),
            "response_inference_seconds": float(response["inference_seconds"]),
            "response_timings_ms": response["timings_ms"],
            "response_keypoints": np.asarray(response["keypoints"], np.float32),
            "response_keypoint_time_s": np.asarray(response["keypoint_time_s"], np.float32),
            "response_proprio_keypoints": np.asarray(response["proprio_keypoints"], np.float32),
        })
        return response, trace

    def plan(self, snapshot):
        began = time.perf_counter()
        self.last_trace = {"schema": "xflow_inference_trace_v1",
                           "sample_ns": int(snapshot["sample_ns"]),
                           "task": self.task, "model": self.model_name,
                           "inference_started": False}
        report = {"sample_ns": int(snapshot["sample_ns"]), "history_sent": snapshot.get("history_rgb") is not None}
        measured_eef = {side: pose7_matrix(snapshot["eef_measured"][i]) for i, side in enumerate(SIDES)}
        measured_q20 = {side: np.asarray(snapshot["q20_measured"][i], float) for i, side in enumerate(SIDES)}
        command_eef = {side: pose7_matrix(snapshot["eef_command"][i]) for i, side in enumerate(SIDES)}
        command_q20 = np.asarray(snapshot["q20_command"], float)
        q16 = np.asarray(snapshot["q16"], float)
        points = self.kinematics.measured_keypoints(measured_eef, measured_q20)
        stage = {"proprio_ms": 1000 * (time.perf_counter() - began)}

        response, inference_trace = self._infer(snapshot, points)
        predicted = np.asarray(response["keypoints"], float)[:self.steps]
        # Robot-time execution instants from the server (speed_ratio applied);
        # default hh/rb: k/30 s.  Speed caps may only lengthen these steps.
        step_times = np.asarray(response["keypoint_time_s"], float)[:self.steps]
        base_durations = np.diff(np.r_[0.0, step_times])
        if np.any(base_durations <= 0) or not np.isfinite(base_durations).all():
            raise PlanRejected("server keypoint_time_s is not increasing")
        report["rpc"] = {"round_trip_ms": response["round_trip_ms"],
                         "timings_ms": response["timings_ms"],
                         "deployment_id": response["deployment_id"]}
        stage["infer_ms"] = 1000 * (time.perf_counter() - began) - stage["proprio_ms"]

        tick = time.perf_counter()
        measured_root = {side: self.camera_T_base @ measured_eef[side] @ self.mapping.sdk_eef_from_hand[side]
                         for side in SIDES}
        command_root = {side: self.camera_T_base @ command_eef[side] @ self.mapping.sdk_eef_from_hand[side]
                        for side in SIDES}
        roots = np.empty((self.steps, 2, 4, 4)); q20 = np.empty((self.steps, 2, 20))
        for side_index, side in enumerate(SIDES):
            solver = self.retargeters[side]
            solver.begin_chunk(measured_q20[side], measured_root[side])
            for frame in range(self.steps):
                answer = solver.retarget_frame(predicted[frame, side_index],
                                               1 + frame, normalize=True)
                if not answer["accepted"]:
                    raise PlanRejected("%s retarget frame %d rejected: %s"
                                       % (side, frame, answer["diagnostic"].get("reason")))
                roots[frame, side_index] = answer["cam_T_root"]
                q20[frame, side_index] = answer["q20"]
        stage["retarget_ms"] = 1000 * (time.perf_counter() - tick)

        execution = self.config["execution"]
        window = int(execution["smoothing_window"])
        anchors = {}
        eef = np.empty((self.steps, 2, 7))
        for side_index, side in enumerate(SIDES):
            smooth_root = smooth_root_trajectory(roots[:, side_index], window)
            smooth_q = smooth_q_trajectory(q20[:, side_index], window)
            offset = np.linalg.inv(smooth_root[0]) @ command_root[side]
            steps, saturated = budget_decay_steps(
                command_q20[side_index] - smooth_q[0], np.linalg.norm(offset[:3, 3]),
                Rotation.from_matrix(offset[:3, :3]).magnitude())
            roots[:, side_index], q20[:, side_index], anchor = apply_local_anchor(
                smooth_root, smooth_q, command_root[side], command_q20[side_index],
                self.mapping.limits[side], mode="decay", decay_steps=steps, apply_to_hand_q=True)
            anchor.pop("weights", None)
            anchor["budget_saturated"] = saturated
            anchors[side] = anchor
            for frame in range(self.steps):
                eef[frame, side_index] = matrix_pose7(
                    camera_hand_to_sdk_eef(self.mapping, side, roots[frame, side_index]))
        report["anchor"] = anchors

        for frame in range(self.steps):
            if np.any(eef[frame, :, :3] < EEF_BOX[:, :, 0]) or np.any(eef[frame, :, :3] > EEF_BOX[:, :, 1]):
                raise PlanRejected("target %d outside the ServoP EEF workspace box" % frame)

        tick = time.perf_counter()
        q_path = np.empty((self.steps, 16))
        q = q16.copy()
        for frame in range(self.steps):
            q, _ = self.pin.solve(q, eef[frame])
            position, rotation = self.pin.fk(q)
            p_error = np.linalg.norm(position - eef[frame, :, :3], axis=1)
            r_error = Rotation.from_matrix(Rotation.from_quat(eef[frame, :, 3:]).as_matrix()
                                           @ np.swapaxes(rotation, -1, -2)).magnitude()
            if np.max(p_error) > 0.020 or np.max(r_error) > math.radians(2.0):
                raise PlanRejected("IK unreachable at step %d: %.1f mm / %.2f deg"
                                   % (frame, 1000 * np.max(p_error), math.degrees(np.max(r_error))))
            q_path[frame] = q
        q_path[:, 14:] = q16[14:]
        # Stream the FK of the in-envelope joint path, as replay does, not the raw
        # target: where a joint saturates (e.g. wrist roll at +/-0.7505) the raw
        # target is unreachable and the controller's own IK pushes the joint past
        # the SDK envelope.  FK(q_path) is reachable inside the envelope.
        raw_eef = eef.copy()
        for frame in range(self.steps):
            position, rotation = self.pin.fk(q_path[frame])
            for side in range(2):
                eef[frame, side, :3] = position[side]
                eef[frame, side, 3:] = Rotation.from_matrix(rotation[side]).as_quat()
        fk_p = np.linalg.norm(eef[:, :, :3] - raw_eef[:, :, :3], axis=-1)
        fk_r = np.array([[(Rotation.from_quat(raw_eef[f, s, 3:]).inv()
                           * Rotation.from_quat(eef[f, s, 3:])).magnitude() for s in range(2)]
                         for f in range(self.steps)])
        report["ik_projection"] = {"max_mm": float(1000 * fk_p.max()),
                                   "max_deg": float(np.degrees(fk_r.max())),
                                   "joints_at_limit": [int(j) for j in range(14)
                                                       if np.any(q_path[:, j] <= self.pin.lower[j] + 1e-3)
                                                       or np.any(q_path[:, j] >= self.pin.upper[j] - 1e-3)],
                                   "limit_margin_rad": IK_LIMIT_MARGIN}
        stage["ik_ms"] = 1000 * (time.perf_counter() - tick)

        start_eef = np.stack([np.asarray(snapshot["eef_command"][i], float) for i in range(2)])
        times, reasons = synchronized_step_times(
            q_path[:, :14], q20, eef, q16[:14], command_q20, start_eef,
            base_dt=base_durations, **self.caps)
        report["timing"] = {"duration_s": float(times[-1]),
                            "nominal_duration_s": float(step_times[-1]),
                            "speed_ratio": self.speed_ratio,
                            "stretch": float(times[-1] / step_times[-1]),
                            "binding": {name: reasons.count(name) for name in sorted(set(reasons))}}

        tick = time.perf_counter()
        try:
            path = np.vstack([q16[None], q_path])
            self.collision.check_path(path)
            hands = np.concatenate([command_q20[None], q20])
            report["hand_body"] = self.hand_checker.check_path(path, hands)["min_swept_clearance_m"]
        except ValueError as error:
            raise PlanRejected("collision certification failed: " + str(error))
        stage["collision_ms"] = 1000 * (time.perf_counter() - tick)
        stage["total_ms"] = 1000 * (time.perf_counter() - began)
        report["stage_ms"] = stage
        return {"times_s": times, "eef": eef, "q20": q20, "q16": q_path,
                "predicted_keypoints": predicted.astype(np.float32),
                "current_keypoints": points.astype(np.float32), "report": report,
                "inference_trace": inference_trace}


def worker_main(connection, kwargs):
    """Child-process loop: one planner, one request at a time, never sends commands."""
    try:
        planner = ChunkPlanner(**kwargs)
        connection.send({"ready": True, "describe": planner.describe()})
    except BaseException as error:
        connection.send({"ready": False, "error": type(error).__name__ + ": " + str(error)})
        return
    while True:
        try:
            request = connection.recv()
        except EOFError:
            return
        if request is None:
            return
        try:
            connection.send({"ok": True, "plan": planner.plan(request)})
        except PlanRejected as error:
            connection.send({"ok": False, "reason": str(error),
                             "inference_trace": planner.last_trace})
        except BaseException as error:
            connection.send({"ok": False, "reason": type(error).__name__ + ": " + str(error),
                             "unexpected": True, "inference_trace": planner.last_trace})
