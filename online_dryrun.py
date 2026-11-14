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
from image_preprocess import (ImagePreprocessor, camera_info_dict,
                              config_section)
from online_core import DeploymentMapping, TimedChunk, resample_chunk
from proprio import (ProprioKinematics, camera_hand_to_sdk_eef,
                     firmware_joint_names)
from timing_policy import (FRAME_PER_STEP_30HZ, NATIVE_TIME, ModelTiming,
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
        required += [side + "_sdk_eef_xyzw", side + "_wuji_q20_rad",
                     "future_" + side + "_sdk_eef_xyzw",
                     "future_" + side + "_wuji_q20_rad"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("observation is missing: " + ", ".join(missing))
    if tuple(data["arm_joint_names"].tolist()) != ARM_STATE_NAMES:
        raise ValueError("recorded arm state alias/order differs from Pinocchio q16 contract")
    for side in SIDES:
        if tuple(data[side + "_wuji_joint_names"].tolist()) != firmware_joint_names(side):
            raise ValueError(side + " Wuji runtime order differs")
    return data


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
        eef = {side: pose7_matrix(data["future_" + side + "_sdk_eef_xyzw"][index])
               for side in SIDES}
        q20 = {side: data["future_" + side + "_wuji_q20_rad"][index]
               for side in SIDES}
        output.append(kinematics.measured_keypoints(eef, q20))
        camera_T_base = np.linalg.inv(mapping.base_from_source_camera)
        roots.append([camera_T_base @ eef[side] @ mapping.sdk_eef_from_hand[side]
                      for side in SIDES])
    return np.asarray(output), np.asarray(roots)


def model_timing_fake():
    return ModelTiming.from_info({
        "contract": "humanhand", "fps": 30.0, "history_offset_frames": 4,
        "keypoint_time_s": (np.arange(1, 33) / 120.0).tolist()})


def offline_ik(root, chunk, seed_q16):
    sys.path.insert(0, str(Path(root) / "Pinocchio"))
    from pin_model import PinArms
    model = PinArms()
    q = np.asarray(seed_q16, dtype=float).copy()
    solved = []
    iterations = []
    position_errors = []
    rotation_errors = []
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
            position_errors.extend(p_error.tolist()); rotation_errors.extend(r_error.tolist())
        except BaseException as error:
            failures.append({"frame": index,
                             "error": type(error).__name__ + ": " + str(error)})
            break
    q_path = np.asarray(solved)
    if len(q_path) >= 2:
        dt = np.diff(chunk.times_ns[:len(q_path)]) / 1e9
        speed = np.abs(np.diff(q_path[:, :14], axis=0)) / dt[:, None]
        peak_speed = float(np.max(speed))
    else:
        peak_speed = float("inf")
    return q_path, {
        "frames_attempted": len(solved) + len(failures),
        "frames_solved": len(solved), "failures": failures,
        "all_reachable": len(solved) == len(chunk.times_ns),
        "position_error_m": stats(position_errors) if position_errors else None,
        "rotation_error_rad": stats(rotation_errors) if rotation_errors else None,
        "iterations_per_arm_max": (int(np.max(iterations)) if iterations else None),
        "arm_joint_speed_peak_rad_s": peak_speed,
        "arm_joint_speed_within_1rad_s": bool(peak_speed <= 1.0),
    }


def run(args):
    began = time.perf_counter()
    root = args.root.resolve()
    config = json.loads(args.config.read_text())
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

    if args.server == "fake-gt":
        predicted, expected_roots = fake_prediction(data, mapping, steps)
        timing = model_timing_fake()
        rpc_report = {"server": "fake-gt", "infer_requests": 0}
        normalize = False
    else:
        image_config = config_section(args.config)
        camera = camera_info_dict(int(data["camera_width"]), int(data["camera_height"]),
                                  str(data["camera_distortion_model"]),
                                  data["camera_K"], data["camera_D"],
                                  str(data["camera_frame_id"]))
        processor = ImagePreprocessor(camera, image_config, args.image_mode)
        head = processor.process(data["head_rgb"])
        history = processor.process(data["history_rgb"])
        service = config["xflow"]
        with XFlowClient(service["relay_host"], service["relay_port"], model=service["model"],
                         deployment_id=service["deployment_id"], infer_budget=1) as client:
            info = client.info(); client.reset()
            response = client.infer(task=args.task, head_rgb=head, history_rgb=history,
                                    keypoints=current_points.astype(np.float32),
                                    sample_ns=int(data["sample_ns"]), image_space="rgb320")
            predicted = response["keypoints"][:steps]
            model_info = dict(info["models"][service["model"]])
            model_info.setdefault("fps", 30.0)
            timing = ModelTiming.from_info(model_info)
            rpc_report = {"server": "remote", "infer_requests": 1,
                          "round_trip_ms": response["round_trip_ms"],
                          "timings_ms": response["timings_ms"],
                          "inference_seconds": response["inference_seconds"],
                          "model": response["model"],
                          "deployment_id": response["deployment_id"]}
        expected_roots = None
        normalize = True
    stage["prediction_ms"] = (time.perf_counter() - began) * 1000 - stage["load_and_proprio_ms"]

    exec_start_ns = int(data["sample_ns"]) + int(round(config["timing"]["planning_reserve_s"] * 1e9))
    schedule = build_schedule(timing, sample_ns=int(data["sample_ns"]),
                              exec_start_ns=(exec_start_ns if args.time_scale == FRAME_PER_STEP_30HZ else None),
                              mode=args.time_scale, steps=steps)
    modes = ("free_root", "kabsch_fixed_root") if args.root_mode == "both" else (args.root_mode,)
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
        eef = np.empty((steps, 2, 7)); diagnostics = []
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
                base_T_eef = camera_hand_to_sdk_eef(
                    mapping, side, answer["cam_T_root"])
                eef[frame, side_index] = matrix_pose7(base_T_eef)
            diagnostics.append(frame_diagnostics)
        if rejected:
            reports[mode] = {"rejected_frames": rejected, "completed": False}
            continue
        chunk = TimedChunk(np.asarray(schedule.deadlines_ns, np.int64), eef, q20,
                           {"mode": mode, "time_policy": schedule.report,
                            "execution_ready": False, "hardware_commands_sent": 0})
        accepted = resample_chunk(chunk, int(round(config["timing"]["planning_period_s"] * 1e9)))
        outside = ((accepted.eef[:, :, :3] < EEF_BOX[None, :, :, 0])
                   | (accepted.eef[:, :, :3] > EEF_BOX[None, :, :, 1]))
        hand_dt = np.diff(chunk.times_ns) / 1e9
        hand_speed = np.abs(np.diff(q20, axis=0)) / hand_dt[:, None, None]
        mode_report = {
            "completed": True, "rejected_frames": [],
            "frames": steps, "resampled_50hz_frames": len(accepted.times_ns),
            "eef_box_violations": int(np.any(outside, axis=2).sum()),
            "quaternion_norm_error_max": float(np.max(np.abs(
                np.linalg.norm(accepted.eef[:, :, 3:], axis=2) - 1))),
            "hand_speed_peak_rad_s": float(np.max(hand_speed)),
            "hand_within_installed_limits": bool(all(
                np.all(q20[:, index] >= mapping.limits[side][:, 0] - 1e-8)
                and np.all(q20[:, index] <= mapping.limits[side][:, 1] + 1e-8)
                for index, side in enumerate(SIDES))),
            "retarget_elapsed_ms": stats([[item[side]["elapsed_ms"]
                                            for side in range(2)]
                                           for item in diagnostics]),
        }
        if expected_roots is not None:
            root_t, root_r = pose_error(roots, expected_roots[:steps])
            expected_eef = np.stack([
                np.stack([pose7_matrix(data["future_" + side + "_sdk_eef_xyzw"][i])
                          for side in SIDES]) for i in range(steps)])
            actual_eef = np.empty_like(expected_eef)
            for i in range(steps):
                for j in range(2):
                    actual_eef[i, j] = pose7_matrix(eef[i, j])
            eef_t, eef_r = pose_error(actual_eef, expected_eef)
            expected_q = np.stack([data["future_" + side + "_wuji_q20_rad"][:steps]
                                   for side in SIDES], axis=1)
            source_relative = np.linalg.inv(expected_roots[:steps, 0]) @ expected_roots[:steps, 1]
            actual_relative = np.linalg.inv(roots[:, 0]) @ roots[:, 1]
            relative_t, relative_r = pose_error(actual_relative, source_relative)
            mode_report["fake_gt"] = {
                "root_translation_m": stats(root_t), "root_rotation_rad": stats(root_r),
                "sdk_eef_translation_m": stats(eef_t), "sdk_eef_rotation_rad": stats(eef_r),
                "q20_abs_error_rad": stats(np.abs(q20 - expected_q)),
                "bimanual_relative_translation_m": stats(relative_t),
                "bimanual_relative_rotation_rad": stats(relative_r),
            }
        ik_started = time.perf_counter()
        q16, ik_report = offline_ik(root, accepted, data["arm_joint_positions_rad"])
        mode_report["offline_pinocchio_ik"] = ik_report
        mode_report["offline_pinocchio_ik_ms"] = (time.perf_counter() - ik_started) * 1000
        reports[mode] = mode_report
        outputs[mode] = {"raw": chunk, "accepted": accepted, "roots": roots,
                         "q20": q20, "q16": q16}
    stage["retarget_and_ik_ms"] = (time.perf_counter() - retarget_started) * 1000
    return {"mapping": mapping, "data": data, "schedule": schedule,
            "rpc": rpc_report, "reports": reports, "outputs": outputs,
            "stage": stage, "current_keypoints": current_points,
            "predicted_keypoints": predicted}


def main():
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).with_name("deployment_config.json"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--server", choices=("fake-gt", "remote"), default="fake-gt")
    parser.add_argument("--root-mode", choices=("free_root", "kabsch_fixed_root", "both"), default="both")
    parser.add_argument("--image-mode", choices=("virtual_cam0", "stretch_undistort"), default="virtual_cam0")
    parser.add_argument("--time-scale", choices=(FRAME_PER_STEP_30HZ, NATIVE_TIME),
                        default=FRAME_PER_STEP_30HZ,
                        help="default is 32 distinct 30Hz frames; no 0.25 speed-scale mode")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--task", default="flower")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.source = args.source.resolve(); args.config = args.config.resolve()
    output = args.output or (args.root / "client/runs" /
                             ("online_dryrun_" + time.strftime("%Y%m%d_%H%M%S")))
    output = output.resolve(); output.mkdir(parents=True, exist_ok=False)
    report = {"schema": "xflow_wuji2_online_dryrun_v1", "completed": False,
              "execution_ready": False, "hardware_commands_sent": 0,
              "source": str(args.source), "server": args.server,
              "root_mode": args.root_mode, "image_mode": args.image_mode,
              "time_scale": args.time_scale, "steps": args.steps}
    try:
        result = run(args)
        arrays = {"current_keypoints": result["current_keypoints"],
                  "predicted_keypoints": result["predicted_keypoints"],
                  "execution_deadlines_ns": np.asarray(result["schedule"].deadlines_ns)}
        for mode, values in result["outputs"].items():
            arrays[mode + "_roots_camera"] = values["roots"]
            arrays[mode + "_q20"] = values["q20"]
            arrays[mode + "_eef_raw"] = values["raw"].eef
            arrays[mode + "_eef_50hz"] = values["accepted"].eef
            arrays[mode + "_times_50hz_ns"] = values["accepted"].times_ns
            arrays[mode + "_ik_q16"] = values["q16"]
        np.savez_compressed(output / "dryrun_targets.npz", **arrays)
        report.update(completed=bool(result["outputs"]), rpc=result["rpc"],
                      schedule=result["schedule"].report,
                      modes=result["reports"], stage_timings_ms=result["stage"],
                      mapping=result["mapping"].report,
                      targets_npz=str(output / "dryrun_targets.npz"))
    except BaseException as error:
        report["error"] = type(error).__name__ + ": " + str(error)
    save_json(output / "report.json", report)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

