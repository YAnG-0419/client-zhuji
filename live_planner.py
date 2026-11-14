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

from execution_postprocess import (apply_local_anchor, select_root_mode,
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
        candidates = {"frame": base_dt,
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
                 num_inference_steps=10, steps=32):
        self.root = Path(root).resolve()
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text())
        validate_execution_config(self.config)
        self.mapping = DeploymentMapping.load(self.root, self.config_path, "live_zhuji_camera")
        self.kinematics = ProprioKinematics(self.mapping)
        self.camera_T_base = np.linalg.inv(self.mapping.base_from_source_camera)
        self.task, self.model_name = str(task), str(model)
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
        model_info = dict(info["models"][self.model_name])
        model_info.setdefault("fps", 30.0)
        self.timing = ModelTiming.from_info(model_info)
        self.deployment_id = str(model_info["deployment_id"])
        self.client.reset()
        # humanhand: 4 native frames = 16 execution frames; realrobot: native 16.
        self.history_frames = 16
        self.root_mode, self.root_reason = select_root_mode("auto", self.timing.contract)
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
        if self.client.session_id is None:
            self.client.reset()
        try:
            return self.client.infer(task=self.task, head_rgb=head, history_rgb=history,
                                     keypoints=points.astype(np.float32),
                                     sample_ns=int(snapshot["sample_ns"]), image_space="rgb320",
                                     num_inference_steps=self.num_inference_steps)
        except XFlowRemoteError as error:
            if error.code == "STALE_SESSION":
                self.client.session_id = None
            raise PlanRejected("server error " + str(error))

    def plan(self, snapshot):
        began = time.perf_counter()
        report = {"sample_ns": int(snapshot["sample_ns"]), "history_sent": snapshot.get("history_rgb") is not None}
        measured_eef = {side: pose7_matrix(snapshot["eef_measured"][i]) for i, side in enumerate(SIDES)}
        measured_q20 = {side: np.asarray(snapshot["q20_measured"][i], float) for i, side in enumerate(SIDES)}
        command_eef = {side: pose7_matrix(snapshot["eef_command"][i]) for i, side in enumerate(SIDES)}
        command_q20 = np.asarray(snapshot["q20_command"], float)
        q16 = np.asarray(snapshot["q16"], float)
        points = self.kinematics.measured_keypoints(measured_eef, measured_q20)
        stage = {"proprio_ms": 1000 * (time.perf_counter() - began)}

        response = self._infer(snapshot, points)
        predicted = np.asarray(response["keypoints"], float)[:self.steps]
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
                                               1 + frame, normalize=(self.timing.contract == "humanhand"))
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
        stage["ik_ms"] = 1000 * (time.perf_counter() - tick)

        start_eef = np.stack([np.asarray(snapshot["eef_command"][i], float) for i in range(2)])
        times, reasons = synchronized_step_times(
            q_path[:, :14], q20, eef, q16[:14], command_q20, start_eef, **self.caps)
        report["timing"] = {"duration_s": float(times[-1]),
                            "nominal_duration_s": self.steps / EXECUTION_FPS,
                            "stretch": float(times[-1] * EXECUTION_FPS / self.steps),
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
                "current_keypoints": points.astype(np.float32), "report": report}


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
            connection.send({"ok": False, "reason": str(error)})
        except BaseException as error:
            connection.send({"ok": False, "reason": type(error).__name__ + ": " + str(error),
                             "unexpected": True})
