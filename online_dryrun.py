#!/usr/bin/env python3
"""End-to-end XFlow kp21 dry-run.  Generates files only; no execute option."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from hand_retarget_wuji2 import Wuji2StreamingRetargeter
from execution_postprocess import (
    ANCHOR_MODES, SPEED_POLICIES, apply_local_anchor, clamp_q_report,
    eef_speed_report, keypoint_offset_report, pose_delta as execution_pose_delta,
    q_speed_report, select_root_mode, smooth_q_trajectory,
    smooth_root_trajectory, trajectory_jitter, validate_execution_config,
    wrist_extrapolation_report)
from image_preprocess import (ImagePreprocessor, camera_info_dict,
                              config_section)
from online_core import DeploymentMapping, TimedChunk, resample_chunk
from proprio import (ProprioKinematics, camera_hand_to_sdk_eef,
                     firmware_joint_names)
from timing_policy import (EXECUTION_CLOCK, FRAME_PER_STEP_30HZ, NATIVE_TIME,
                           REALTIME_CUTOFF, SYNC_HOLD, WALL_CLOCK, ModelTiming,
                           build_schedule)
from xflow_rpc import XFlowClient


SIDES = ("left", "right")
EEF_BOX = np.asarray([[[.250, .732], [-.213, .900], [-.673, .5]],
                      [[.250, .732], [-.900, .213], [-.673, .5]]], dtype=float)
ARM_STATE_NAMES = (
    "abad_L_Joint", "hip_L_Joint", "yaw_L_Joint", "knee_L_Joint",
    "wrist_yaw_L_Joint", "wrist_pitch_L_Joint", "wrist_roll_L_Joint",
    "abad_R_Joint", "hip_R_Joint", "yaw_R_Joint", "knee_R_Joint",
    "wrist_yaw_R_Joint", "wrist_pitch_R_Joint", "wrist_roll_R_Joint",
    "head_pitch_Joint", "head_yaw_Joint")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n")


def pose7_matrix(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (7,) or not np.isfinite(value).all():
        raise ValueError("EEF pose must be xyz+xyzw")
    output = np.eye(4)
    output[:3, :3] = Rotation.from_quat(value[3:]).as_matrix()
    output[:3, 3] = value[:3]
    return output


def matrix_pose7(value):
    return np.r_[value[:3, 3], Rotation.from_matrix(value[:3, :3]).as_quat()]


def pose_error(actual, expected):
    translation = np.linalg.norm(actual[..., :3, 3] - expected[..., :3, 3], axis=-1)
    relative = actual[..., :3, :3] @ np.swapaxes(expected[..., :3, :3], -1, -2)
    rotation = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(translation.shape)
    return translation, rotation


def stats(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    return {"rmse": float(np.sqrt(np.mean(values * values))),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values))}


def load_observation(path):
    data = np.load(path, allow_pickle=False)
    required = ["sample_ns", "arm_joint_names", "arm_joint_positions_rad"]
    for side in SIDES:
        required += [side + "_sdk_eef_xyzw", side + "_wuji_q20_rad"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("observation is missing: " + ", ".join(missing))
    if tuple(data["arm_joint_names"].tolist()) != ARM_STATE_NAMES:
        raise ValueError("recorded arm state alias/order differs from Pinocchio q16 contract")
    for side in SIDES:
        if tuple(data[side + "_wuji_joint_names"].tolist()) != firmware_joint_names(side):
            raise ValueError(side + " Wuji runtime order differs")
    return data


def has_recorded_future(data):
    return all("future_" + side + "_sdk_eef_xyzw" in data
               and "future_" + side + "_wuji_q20_rad" in data for side in SIDES)


def base_eef_dict(data, prefix=""):
    return {side: pose7_matrix(data[prefix + side + "_sdk_eef_xyzw"])
            for side in SIDES}


def q_dict(data, prefix=""):
    return {side: np.asarray(data[prefix + side + "_wuji_q20_rad"], float)
            for side in SIDES}


def fake_prediction(data, mapping, steps):
    kinematics = ProprioKinematics(mapping)
    output = []
    roots = []
    for index in range(steps):
        if index == 0:
            eef = base_eef_dict(data); q20 = q_dict(data)
        else:
            eef = {side: pose7_matrix(
                data["future_" + side + "_sdk_eef_xyzw"][index - 1])
                   for side in SIDES}
            q20 = {side: data["future_" + side + "_wuji_q20_rad"][index - 1]
                   for side in SIDES}
        output.append(kinematics.measured_keypoints(eef, q20))
        camera_T_base = np.linalg.inv(mapping.base_from_source_camera)
        roots.append([camera_T_base @ eef[side] @ mapping.sdk_eef_from_hand[side]
                      for side in SIDES])
    return np.asarray(output), np.asarray(roots)


def recorded_future_roots(data, mapping, steps):
    camera_T_base = np.linalg.inv(mapping.base_from_source_camera)
    return np.asarray([
        [camera_T_base
         @ pose7_matrix(data["future_" + side + "_sdk_eef_xyzw"][index])
         @ mapping.sdk_eef_from_hand[side]
         for side in SIDES]
        for index in range(steps)])


def model_timing_fake():
    return ModelTiming.from_info({
        "contract": "humanhand", "fps": 30.0, "history_offset_frames": 4,
        "keypoint_time_s": (np.arange(1, 33) / 120.0).tolist()})


def _cache_paths(directory, label):
    safe = "".join(character if character.isalnum() or character in "-_" else "_"
                   for character in label)
    return Path(directory) / (safe + "_response.npz"), Path(directory) / (safe + "_request.json")


def save_response_cache(directory, label, response, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    npz, request = _cache_paths(directory, label)
    np.savez_compressed(npz,
                        keypoints=np.asarray(response["keypoints"], np.float32),
                        keypoint_time_s=np.asarray(response["keypoint_time_s"], np.float32))
    cache_metadata = dict(metadata)
    cache_metadata.update({
        "schema": "xflow_response_cache_v1", "response_npz": npz.name,
        "contract": response["contract"], "model": response["model"],
        "deployment_id": response["deployment_id"],
        "task": response["task"], "sample_ns": int(response["sample_ns"]),
        "timings_ms": response["timings_ms"],
        "inference_seconds": float(response["inference_seconds"]),
        "round_trip_ms": float(response["round_trip_ms"]),
    })
    save_json(request, cache_metadata)
    return npz, request


def load_response_cache(directory, label):
    npz, request = _cache_paths(directory, label)
    legacy_npz = Path(directory) / "dryrun_targets.npz"
    legacy_report = Path(directory) / "report.json"
    if not request.is_file() and legacy_npz.is_file() and legacy_report.is_file():
        report = json.loads(legacy_report.read_text())
        with np.load(legacy_npz, allow_pickle=False) as arrays:
            predicted = np.asarray(arrays["predicted_keypoints"], np.float32)
        metadata = dict(report["rpc"])
        metadata.update({"schema": "r1_dryrun_equivalent_response_cache",
                         "contract": "humanhand", "task": "flower",
                         "sample_ns": int(np.load(report["source"], allow_pickle=False)["sample_ns"])
                         if Path(report["source"]).is_file() else 1,
                         "history_offset_frames": 4,
                         "source_report": str(legacy_report)})
        times = np.arange(1, 33, dtype=np.float32) / 120.0
        timing = ModelTiming.from_info({
            "contract": "humanhand", "fps": 30.0, "history_offset_frames": 4,
            "keypoint_time_s": times.tolist()})
        return predicted, timing, metadata
    metadata = json.loads(request.read_text())
    with np.load(npz, allow_pickle=False) as arrays:
        predicted = np.asarray(arrays["keypoints"], np.float32)
        times = np.asarray(arrays["keypoint_time_s"], np.float32)
    if predicted.shape != (32, 2, 21, 3) or times.shape != (32,):
        raise ValueError("cached response has unexpected shape")
    timing = ModelTiming.from_info({
        "contract": metadata["contract"], "fps": 30.0,
        "history_offset_frames": int(metadata.get("history_offset_frames", 4)),
        "keypoint_time_s": times.tolist()})
    return predicted, timing, metadata


def offline_ik(root, chunk, seed_q16, exec_start_ns):
    sys.path.insert(0, str(Path(root) / "Pinocchio"))
    from pin_model import PinArms
    model = PinArms()
    q = np.asarray(seed_q16, dtype=float).copy()
    solved = []
    iterations = []
    failures = []
    for index, target in enumerate(chunk.eef):
        try:
            q, count = model.solve(q, target)
            position, rotations = model.fk(q)
            p_error = np.linalg.norm(position - target[:, :3], axis=1)
            target_rotation = Rotation.from_quat(target[:, 3:]).as_matrix()
            r_error = Rotation.from_matrix(
                target_rotation @ np.swapaxes(rotations, -1, -2)).magnitude()
            if np.max(p_error) > 0.020 or np.max(r_error) > np.deg2rad(2):
                raise ValueError("IK residual exceeds 20 mm / 2 deg")
            solved.append(q.copy()); iterations.append(count)
        except BaseException as error:
            failures.append({"frame": index,
                             "error": type(error).__name__ + ": " + str(error)})
            break
    q_path = np.asarray(solved)
    anchor_report = None
    if len(q_path) >= 1:
        # The EEF trajectory is already anchored to the measured pose, but a
        # redundant IK solve may choose a different wrist configuration at
        # frame zero. Apply the same smooth local-offset idea in joint space:
        # exact measured q16 at t0, then decay over the same 0.4 s. This is part
        # of IK branch continuity, not the diagnostic clamp_report policy.
        decay_frames = max(1, int(round(12 * 50.0 / 30.0)))
        x = np.minimum(np.arange(len(q_path), dtype=float) / decay_frames, 1.0)
        weights = 1.0 - x * x * (3.0 - 2.0 * x)
        q_offset = np.asarray(seed_q16, float) - q_path[0]
        q_path = q_path + weights[:, None] * q_offset[None]
        position_errors = []
        rotation_errors = []
        residual_failures = []
        for index, (candidate, target) in enumerate(zip(q_path, chunk.eef)):
            position, rotations = model.fk(candidate)
            p_error = np.linalg.norm(position - target[:, :3], axis=1)
            target_rotation = Rotation.from_quat(target[:, 3:]).as_matrix()
            r_error = Rotation.from_matrix(
                target_rotation @ np.swapaxes(rotations, -1, -2)).magnitude()
            position_errors.extend(p_error.tolist())
            rotation_errors.extend(r_error.tolist())
            if np.max(p_error) > 0.020 or np.max(r_error) > np.deg2rad(2):
                residual_failures.append({"frame": index,
                                          "position_max_m": float(np.max(p_error)),
                                          "rotation_max_rad": float(np.max(r_error))})
        failures.extend(residual_failures)
        anchor_report = {"mode": "decay", "decay_seconds": 12 / 30.0,
                         "decay_frames_50hz": decay_frames,
                         "q16_offset_abs_max_rad": float(np.max(np.abs(q_offset))),
                         "weights": weights.tolist(),
                         "post_anchor_residual_failures": residual_failures}
        speed_report = q_speed_report(q_path[:, :14],
                                      np.asarray(seed_q16, float)[:14],
                                      chunk.times_ns[:len(q_path)], exec_start_ns)
        peak_speed = speed_report["peak_rad_s"]
    else:
        position_errors = []; rotation_errors = []
        speed_report = {"peak_rad_s": float("inf"), "within_limit": False,
                        "includes_measured_to_step0": True}
        peak_speed = float("inf")
    return q_path, {
        "frames_attempted": len(solved),
        "frames_solved": len(solved), "failures": failures,
        "all_reachable": len(solved) == len(chunk.times_ns) and not failures,
        "position_error_m": stats(position_errors) if position_errors else None,
        "rotation_error_rad": stats(rotation_errors) if rotation_errors else None,
        "iterations_per_arm_max": (int(np.max(iterations)) if iterations else None),
        "arm_joint_speed_peak_rad_s": peak_speed,
        "arm_joint_speed_within_1rad_s": bool(peak_speed <= 1.0),
        "arm_joint_speed": speed_report,
        "ik_branch_anchor": anchor_report,
    }


def run(args):
    began = time.perf_counter()
    root = args.root.resolve()
    config = json.loads(args.config.read_text())
    validate_execution_config(config)
    mapping = DeploymentMapping.load(root, args.config, "live_zhuji_camera")
    data = load_observation(args.source)
    steps = int(args.steps)
    if not 2 <= steps <= 32:
        raise ValueError("steps must be within [2,32]")
    current_eef = base_eef_dict(data)
    current_q = q_dict(data)
    camera_T_base = np.linalg.inv(mapping.base_from_source_camera)
    current_roots = {side: camera_T_base @ current_eef[side]
                     @ mapping.sdk_eef_from_hand[side] for side in SIDES}
    kinematics = ProprioKinematics(mapping)
    current_points = kinematics.measured_keypoints(current_eef, current_q)
    stage = {"load_and_proprio_ms": (time.perf_counter() - began) * 1000}

    response_label = getattr(args, "response_label", args.source.stem)
    if args.server == "fake-gt":
        if not has_recorded_future(data):
            raise ValueError("fake-gt requires recorded future EEF and q20")
        predicted, expected_roots = fake_prediction(data, mapping, steps)
        reference_name = "fake_gt"
        timing = model_timing_fake()
        rpc_report = {"server": "fake-gt", "infer_requests": 0}
        normalize = False
    elif args.server == "replay":
        predicted_all, timing, cached = load_response_cache(
            getattr(args, "replay_responses"), response_label)
        predicted = predicted_all[:steps]
        expected_roots = (recorded_future_roots(data, mapping, steps)
                          if has_recorded_future(data) else None)
        reference_name = "recorded_future_reference"
        normalize = True
        rpc_report = {"server": "replay-cache", "infer_requests": 0,
                      "cache": cached,
                      "round_trip_ms": float(cached.get("round_trip_ms", 0.0))}
    else:
        image_config = config_section(args.config)
        camera = camera_info_dict(int(data["camera_width"]), int(data["camera_height"]),
                                  str(data["camera_distortion_model"]),
                                  data["camera_K"], data["camera_D"],
                                  str(data["camera_frame_id"]))
        processor = ImagePreprocessor(camera, image_config, args.image_mode,
                                      getattr(args, "virtual_preset", "cam0_newk"))
        head = processor.process(data["head_rgb"])
        history = (processor.process(data["history_rgb"])
                   if "history_rgb" in data else None)
        service = config["xflow"]
        # Follow whichever service the relay points at: model/deployment_id are
        # optional pins, so a new checkpoint needs no config edit.
        with XFlowClient(service["relay_host"], service["relay_port"],
                         model=service.get("model"),
                         deployment_id=service.get("deployment_id"), infer_budget=1) as client:
            info = client.info(); client.reset()
            response = client.infer(task=args.task, head_rgb=head, history_rgb=history,
                                    keypoints=current_points.astype(np.float32),
                                    sample_ns=int(data["sample_ns"]), image_space="rgb320",
                                    num_inference_steps=getattr(args, "num_inference_steps", 10))
            predicted = response["keypoints"][:steps]
            model_info = dict(info["models"][client.model])
            model_info.setdefault("fps", 30.0)
            timing = ModelTiming.from_info(model_info)
            rpc_report = {"server": "remote", "infer_requests": 1,
                          "round_trip_ms": response["round_trip_ms"],
                          "timings_ms": response["timings_ms"],
                          "inference_seconds": response["inference_seconds"],
                          "model": response["model"],
                          "deployment_id": response["deployment_id"]}
            if getattr(args, "save_responses", None):
                save_response_cache(args.save_responses, response_label, response, {
                    "source": str(args.source), "image_mode": args.image_mode,
                    "virtual_preset": getattr(args, "virtual_preset", "cam0_newk"),
                    "history_clock": getattr(args, "history_clock", EXECUTION_CLOCK),
                    "history_offset_frames": 16,
                    "request_task": args.task,
                    "num_inference_steps": getattr(args, "num_inference_steps", 10),
                })
        expected_roots = (recorded_future_roots(data, mapping, steps)
                          if has_recorded_future(data) else None)
        reference_name = "recorded_future_reference"
        normalize = True
    stage["prediction_ms"] = (time.perf_counter() - began) * 1000 - stage["load_and_proprio_ms"]

    latency_policy = getattr(args, "latency_policy", config["timing"]["latency_policy"])
    history_clock = getattr(args, "history_clock", config["timing"]["history_clock"])
    round_trip_ms = float(rpc_report.get("round_trip_ms", 0.0))
    dispatch_ns = int(data["sample_ns"]) + max(1, int(round(round_trip_ms * 1e6)))
    schedule = build_schedule(timing, sample_ns=int(data["sample_ns"]),
                              dispatch_ns=(dispatch_ns if args.time_scale == FRAME_PER_STEP_30HZ else None),
                              reserve_s=config["timing"]["planning_reserve_s"],
                              mode=args.time_scale, latency_policy=latency_policy,
                              history_clock=history_clock, steps=steps)
    schedule.report["stationary_assumption"] = "unverified_offline"
    requested_root_mode = getattr(args, "root_mode", config["hand_retarget"]["root_mode"])
    if requested_root_mode == "both":
        modes = ("free_root", "kabsch_fixed_root")
        root_reasons = {mode: "explicit comparison" for mode in modes}
    else:
        selected, reason = select_root_mode(requested_root_mode, timing.contract)
        modes = (selected,); root_reasons = {selected: reason}
    outputs = {}
    reports = {}
    retarget_started = time.perf_counter()
    hand_config = config["hand_retarget"]
    for mode in modes:
        solvers = {}
        for side in SIDES:
            solvers[side] = Wuji2StreamingRetargeter(
                side, mode, thresholds=hand_config["thresholds"],
                max_nfev=hand_config["max_nfev"], weights=hand_config["weights"],
                temporal_first=hand_config["temporal_first"],
                temporal_stream=hand_config["temporal_stream"],
                fixed_outer_iterations=hand_config["fixed_outer_iterations"])
            solvers[side].begin_chunk(current_q[side], current_roots[side])
        roots = np.empty((steps, 2, 4, 4)); q20 = np.empty((steps, 2, 20))
        diagnostics = []
        rejected = []
        for frame in range(steps):
            frame_diagnostics = []
            for side_index, side in enumerate(SIDES):
                answer = solvers[side].retarget_frame(
                    predicted[frame, side_index], schedule.deadlines_ns[frame],
                    normalize=normalize)
                frame_diagnostics.append(answer["diagnostic"])
                if not answer["accepted"]:
                    rejected.append({"frame": frame, "side": side,
                                     "diagnostic": answer["diagnostic"]})
                    continue
                roots[frame, side_index] = answer["cam_T_root"]
                q20[frame, side_index] = answer["q20"]
            diagnostics.append(frame_diagnostics)
        if rejected:
            reports[mode] = {"rejected_frames": rejected, "completed": False}
            continue
        fit_roots = roots.copy(); fit_q20 = q20.copy()
        smooth_roots = np.empty_like(roots); smooth_q20 = np.empty_like(q20)
        anchor_reports = {}
        execution = config["execution"]
        smoothing_window = int(getattr(args, "smoothing_window",
                                       execution["smoothing_window"]))
        requested_smoothing_window = smoothing_window
        if args.server == "fake-gt":
            # Fake GT contains no model jitter. Keep this geometry-preservation
            # regression exact; smoothing behavior has dedicated noisy tests.
            smoothing_window = 1
        anchor_mode = getattr(args, "anchor_mode", execution["anchor"]["mode"])
        decay_steps = int(getattr(args, "decay_steps",
                                  execution["anchor"]["decay_steps"]))
        for side_index, side in enumerate(SIDES):
            smoothed_root = smooth_root_trajectory(roots[:, side_index], smoothing_window)
            smoothed_q = smooth_q_trajectory(q20[:, side_index], smoothing_window)
            roots[:, side_index], q20[:, side_index], anchor_reports[side] = apply_local_anchor(
                smoothed_root, smoothed_q, current_roots[side], current_q[side],
                mapping.limits[side], mode=anchor_mode, decay_steps=decay_steps,
                apply_to_hand_q=execution["anchor"]["apply_to_hand_q"])
            smooth_roots[:, side_index] = smoothed_root
            smooth_q20[:, side_index] = smoothed_q
        eef = np.empty((steps, 2, 7))
        eef_matrices = np.empty((steps, 2, 4, 4))
        for frame in range(steps):
            for side_index, side in enumerate(SIDES):
                base_T_eef = camera_hand_to_sdk_eef(mapping, side,
                                                    roots[frame, side_index])
                eef_matrices[frame, side_index] = base_T_eef
                eef[frame, side_index] = matrix_pose7(base_T_eef)
        chunk = TimedChunk(np.asarray(schedule.deadlines_ns, np.int64), eef, q20,
                           {"mode": mode, "time_policy": schedule.report,
                            "execution_ready": False, "hardware_commands_sent": 0})
        accepted = resample_chunk(chunk, int(round(config["timing"]["planning_period_s"] * 1e9)))
        outside = ((accepted.eef[:, :, :3] < EEF_BOX[None, :, :, 0])
                   | (accepted.eef[:, :, :3] > EEF_BOX[None, :, :, 1]))
        measured_q = np.stack([current_q[side] for side in SIDES])
        measured_eef = np.stack([current_eef[side] for side in SIDES])
        hand_speed = q_speed_report(q20, measured_q, chunk.times_ns,
                                    schedule.exec_start_ns,
                                    execution["hand_joint_speed_limit_rad_s"])
        eef_speeds = {side: eef_speed_report(eef_matrices[:, side_index],
                                              current_eef[side], chunk.times_ns,
                                              schedule.exec_start_ns)
                      for side_index, side in enumerate(SIDES)}
        speed_policy = getattr(args, "speed_policy", execution["speed_policy"])
        clamp_report = None
        if speed_policy == "clamp_report":
            _, hand_clamp = clamp_q_report(q20, measured_q, chunk.times_ns,
                                           schedule.exec_start_ns,
                                           execution["hand_joint_speed_limit_rad_s"])
            clamp_report = {"hand_q20": hand_clamp,
                            "diagnostic_only": True, "execution_ready": False}
        elif speed_policy not in SPEED_POLICIES:
            raise ValueError("unknown speed policy: " + str(speed_policy))
        mode_report = {
            "completed": True, "rejected_frames": [],
            "actual_root_mode": mode, "root_mode_reason": root_reasons[mode],
            "anchor": anchor_reports, "smoothing_window": smoothing_window,
            "requested_smoothing_window": requested_smoothing_window,
            "smoothing_bypassed_reason": ("fake_gt_has_no_model_noise"
                                          if args.server == "fake-gt" else None),
            "speed_policy": speed_policy, "clamp_report": clamp_report,
            "frames": steps, "resampled_50hz_frames": len(accepted.times_ns),
            "eef_box_violations": int(np.any(outside, axis=2).sum()),
            "quaternion_norm_error_max": float(np.max(np.abs(
                np.linalg.norm(accepted.eef[:, :, 3:], axis=2) - 1))),
            "hand_speed_peak_rad_s": hand_speed["peak_rad_s"],
            "hand_speed_within_1rad_s": hand_speed["within_limit"],
            "hand_joint_speed": hand_speed,
            "sdk_eef_speed": eef_speeds,
            "hand_within_installed_limits": bool(all(
                np.all(q20[:, index] >= mapping.limits[side][:, 0] - 1e-8)
                and np.all(q20[:, index] <= mapping.limits[side][:, 1] + 1e-8)
                for index, side in enumerate(SIDES))),
            "retarget_elapsed_ms": stats([[item[side]["elapsed_ms"]
                                            for side in range(2)]
                                           for item in diagnostics]),
            "tip_rmse_m": stats([[item[side]["tip_rmse_m"]
                                   for side in range(2)] for item in diagnostics]),
            "jitter": {
                side: {"fit": trajectory_jitter(fit_roots[:, index], fit_q20[:, index]),
                       "smoothed": trajectory_jitter(smooth_roots[:, index],
                                                     smooth_q20[:, index]),
                       "anchored": trajectory_jitter(roots[:, index], q20[:, index])}
                for index, side in enumerate(SIDES)},
        }
        first_root_t_before, first_root_r_before = execution_pose_delta(
            fit_roots[0], np.stack([current_roots[side] for side in SIDES]))
        first_root_t_after, first_root_r_after = execution_pose_delta(
            roots[0], np.stack([current_roots[side] for side in SIDES]))
        first_eef_t, first_eef_r = execution_pose_delta(eef_matrices[0], measured_eef)
        mode_report["first_step_from_measured"] = {
            "root_before_anchor_translation_m": first_root_t_before.tolist(),
            "root_before_anchor_rotation_rad": first_root_r_before.tolist(),
            "root_after_anchor_translation_m": first_root_t_after.tolist(),
            "root_after_anchor_rotation_rad": first_root_r_after.tolist(),
            "sdk_eef_translation_m": first_eef_t.tolist(),
            "sdk_eef_rotation_rad": first_eef_r.tolist(),
            "q20_abs_max_rad": float(np.max(np.abs(q20[0] - measured_q))),
        }
        if expected_roots is not None:
            root_t, root_r = pose_error(roots, expected_roots[:steps])
            if reference_name == "fake_gt":
                expected_eef = np.stack([
                    np.stack([current_eef[side] if i == 0 else pose7_matrix(
                        data["future_" + side + "_sdk_eef_xyzw"][i - 1])
                              for side in SIDES]) for i in range(steps)])
            else:
                expected_eef = np.stack([
                    np.stack([pose7_matrix(data["future_" + side + "_sdk_eef_xyzw"][i])
                              for side in SIDES]) for i in range(steps)])
            actual_eef = np.empty_like(expected_eef)
            for i in range(steps):
                for j in range(2):
                    actual_eef[i, j] = pose7_matrix(eef[i, j])
            eef_t, eef_r = pose_error(actual_eef, expected_eef)
            if reference_name == "fake_gt":
                expected_q = np.stack([
                    np.vstack([current_q[side],
                               data["future_" + side + "_wuji_q20_rad"][:steps - 1]])
                    for side in SIDES], axis=1)
            else:
                expected_q = np.stack([data["future_" + side + "_wuji_q20_rad"][:steps]
                                       for side in SIDES], axis=1)
            source_relative = np.linalg.inv(expected_roots[:steps, 0]) @ expected_roots[:steps, 1]
            actual_relative = np.linalg.inv(roots[:, 0]) @ roots[:, 1]
            relative_t, relative_r = pose_error(actual_relative, source_relative)
            mode_report[reference_name] = {
                "root_translation_m": stats(root_t), "root_rotation_rad": stats(root_r),
                "sdk_eef_translation_m": stats(eef_t), "sdk_eef_rotation_rad": stats(eef_r),
                "q20_abs_error_rad": stats(np.abs(q20 - expected_q)),
                "bimanual_relative_translation_m": stats(relative_t),
                "bimanual_relative_rotation_rad": stats(relative_r),
            }
        ik_started = time.perf_counter()
        q16, ik_report = offline_ik(root, accepted, data["arm_joint_positions_rad"],
                                    schedule.exec_start_ns)
        if speed_policy == "clamp_report" and len(q16):
            _, arm_clamp = clamp_q_report(
                q16[:, :14], np.asarray(data["arm_joint_positions_rad"], float)[:14],
                accepted.times_ns[:len(q16)], schedule.exec_start_ns,
                execution["arm_joint_speed_limit_rad_s"])
            clamp_report["arm_q16"] = arm_clamp
        mode_report["offline_pinocchio_ik"] = ik_report
        mode_report["offline_pinocchio_ik_ms"] = (time.perf_counter() - ik_started) * 1000
        mode_report["all_offline_safety_checks_passed"] = bool(
            mode_report["eef_box_violations"] == 0
            and mode_report["hand_within_installed_limits"]
            and mode_report["hand_speed_within_1rad_s"]
            and ik_report["all_reachable"]
            and ik_report["arm_joint_speed_within_1rad_s"]
            and speed_policy == "reject")
        reports[mode] = mode_report
        outputs[mode] = {"raw": chunk, "accepted": accepted, "roots": roots,
                         "q20": q20, "q16": q16,
                         "fit_roots": fit_roots, "fit_q20": fit_q20,
                         "smooth_roots": smooth_roots, "smooth_q20": smooth_q20}
    stage["retarget_and_ik_ms"] = (time.perf_counter() - retarget_started) * 1000
    return {"mapping": mapping, "data": data, "schedule": schedule,
            "rpc": rpc_report, "reports": reports, "outputs": outputs,
            "stage": stage, "current_keypoints": current_points,
            "predicted_keypoints": predicted,
            "prediction_offset": keypoint_offset_report(
                predicted[0], current_points,
                np.stack([current_roots[side] for side in SIDES])),
            "wrist_extrapolation": wrist_extrapolation_report(
                predicted, current_points[:, 0])}


def _fraction_sources(source, fraction_text):
    if not fraction_text:
        return [(source, source.stem)]
    if not source.is_dir():
        raise ValueError("--snapshot-fractions requires --source to be a snapshot directory")
    output = []
    for text in fraction_text.split(","):
        fraction = float(text)
        if not 0.0 < fraction < 1.0:
            raise ValueError("snapshot fractions must be within (0,1)")
        label = "frac%03d" % int(round(fraction * 100))
        candidates = (source / (label + ".npz"),
                      source / label / "observation.npz",
                      source / ("observation_" + label + ".npz"))
        match = next((candidate for candidate in candidates if candidate.is_file()), None)
        if match is None:
            raise FileNotFoundError("snapshot for %s; tried %s" %
                                    (label, ", ".join(str(value) for value in candidates)))
        output.append((match, label))
    return output


def _write_result(output, result):
    arrays = {"current_keypoints": result["current_keypoints"],
              "predicted_keypoints": result["predicted_keypoints"],
              "execution_deadlines_ns": np.asarray(result["schedule"].deadlines_ns)}
    for mode, values in result["outputs"].items():
        arrays[mode + "_roots_camera"] = values["roots"]
        arrays[mode + "_q20"] = values["q20"]
        arrays[mode + "_fit_roots_camera"] = values["fit_roots"]
        arrays[mode + "_fit_q20"] = values["fit_q20"]
        arrays[mode + "_smooth_roots_camera"] = values["smooth_roots"]
        arrays[mode + "_smooth_q20"] = values["smooth_q20"]
        arrays[mode + "_eef_raw"] = values["raw"].eef
        arrays[mode + "_eef_50hz"] = values["accepted"].eef
        arrays[mode + "_times_50hz_ns"] = values["accepted"].times_ns
        arrays[mode + "_ik_q16"] = values["q16"]
    target = output / "dryrun_targets.npz"
    np.savez_compressed(target, **arrays)
    return target


def main():
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("deployment_config.json"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--server", choices=("fake-gt", "remote", "replay"), default="fake-gt")
    parser.add_argument("--root-mode", choices=("auto", "free_root", "kabsch_fixed_root", "both"), default="auto")
    parser.add_argument("--image-mode", choices=("virtual_cam0", "stretch_undistort"), default="virtual_cam0")
    parser.add_argument("--virtual-preset", choices=("cam0_newk", "hh_435", "hh_335"), default="cam0_newk")
    parser.add_argument("--time-scale", choices=(FRAME_PER_STEP_30HZ, NATIVE_TIME),
                        default=FRAME_PER_STEP_30HZ,
                        help="default is 32 distinct 30Hz frames; no 0.25 speed-scale mode")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--task", default="flower")
    parser.add_argument("--latency-policy", choices=(SYNC_HOLD, REALTIME_CUTOFF),
                        default=SYNC_HOLD)
    parser.add_argument("--history-clock", choices=(EXECUTION_CLOCK, WALL_CLOCK),
                        default=EXECUTION_CLOCK)
    parser.add_argument("--anchor-mode", choices=ANCHOR_MODES, default="decay")
    parser.add_argument("--decay-steps", type=int, default=12)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--speed-policy", choices=SPEED_POLICIES, default="reject")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--snapshot-fractions",
                        help="comma-separated fractions; source must contain fracNNN snapshots")
    parser.add_argument("--save-responses", type=Path)
    parser.add_argument("--replay-responses", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.source = args.source.resolve(); args.config = args.config.resolve()
    if args.server == "replay" and args.replay_responses is None:
        parser.error("--server replay requires --replay-responses")
    output = args.output or (args.root / "client/runs" /
                             ("r2-online-dryrun-" + time.strftime("%Y%m%d-%H%M%S")))
    output = output.resolve(); output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "xflow_wuji2_online_dryrun_r2", "completed": False,
              "execution_ready": False, "hardware_commands_sent": 0,
              "source": str(args.source), "server": args.server,
              "root_mode": args.root_mode, "image_mode": args.image_mode,
              "time_scale": args.time_scale, "latency_policy": args.latency_policy,
              "history_clock": args.history_clock, "anchor_mode": args.anchor_mode,
              "speed_policy": args.speed_policy, "steps": args.steps,
              "infer_requests": 0, "snapshots": {}}
    try:
        sources = _fraction_sources(args.source, args.snapshot_fractions)
        all_completed = True
        for source, label in sources:
            args.source = source
            args.response_label = label + "-" + args.image_mode + ("" if args.virtual_preset == "cam0_newk" else "-" + args.virtual_preset)
            destination = output if len(sources) == 1 else output / label
            destination.mkdir(parents=True, exist_ok=(destination == output))
            result = run(args)
            targets = _write_result(destination, result)
            snapshot = {"completed": bool(result["outputs"]), "rpc": result["rpc"],
                        "schedule": result["schedule"].report,
                        "modes": result["reports"],
                        "stage_timings_ms": result["stage"],
                        "mapping": result["mapping"].report,
                        "prediction_offset": result["prediction_offset"],
                        "wrist_extrapolation": result["wrist_extrapolation"],
                        "targets_npz": str(targets),
                        "hardware_commands_sent": 0,
                        "execution_ready": False}
            save_json(destination / "report.json", snapshot)
            report["snapshots"][label] = snapshot
            report["infer_requests"] += int(result["rpc"].get("infer_requests", 0))
            all_completed = all_completed and snapshot["completed"]
        report["completed"] = all_completed
    except BaseException as error:
        report["error"] = type(error).__name__ + ": " + str(error)
    save_json(output / "summary.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

