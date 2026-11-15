"""Persistent inference/command/feedback evidence for XFlow live deployment.

The recorder is deliberately transport agnostic: it observes arrays supplied by
``live_client`` and never imports ROS, the robot SDK, or a publisher.  Each
``plan_id`` links the exact model request/response to one optional execution
record so model/planning errors can be separated from tracking errors later.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation


SCHEMA = "xflow_deployment_audit_v1"
SIDES = ("left", "right")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("not JSON serializable: " + type(value).__name__)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                     allow_nan=False, default=_json_value) + "\n")


def _stats(value):
    value = np.asarray(value, dtype=float).reshape(-1)
    if not len(value):
        return None
    return {"mean": float(np.mean(value)), "p50": float(np.percentile(value, 50)),
            "p95": float(np.percentile(value, 95)), "max": float(np.max(value))}


def _pose_errors(actual, target):
    actual = np.asarray(actual, dtype=float)
    target = np.asarray(target, dtype=float)
    position = np.linalg.norm(actual[..., :3] - target[..., :3], axis=-1)
    flat_actual = actual[..., 3:].reshape(-1, 4)
    flat_target = target[..., 3:].reshape(-1, 4)
    rotation = (Rotation.from_quat(flat_actual).inv()
                * Rotation.from_quat(flat_target)).magnitude().reshape(position.shape)
    return position, rotation


def _metadata_without_arrays(trace):
    if not trace:
        return {}
    return {key: value for key, value in trace.items()
            if not isinstance(value, np.ndarray) and key != "history_rgb"}


def save_plan_audit(output_dir, plan_id, snapshot, answer):
    """Save exact preprocessed inference I/O and planner output/rejection."""
    output_dir = Path(output_dir)
    plans = output_dir / "audit_plans"
    plans.mkdir(parents=True, exist_ok=True)
    stem = "plan%04d" % int(plan_id)
    trace = (answer.get("plan", {}).get("inference_trace") if answer.get("ok")
             else answer.get("inference_trace")) or {}
    plan = answer.get("plan") if answer.get("ok") else None
    arrays = {
        "sample_q16_rad": np.asarray(snapshot["q16"], np.float64),
        "sample_eef_measured_xyzw": np.asarray(snapshot["eef_measured"], np.float64),
        "sample_q20_measured_rad": np.asarray(snapshot["q20_measured"], np.float64),
        "sample_eef_command_xyzw": np.asarray(snapshot["eef_command"], np.float64),
        "sample_q20_command_rad": np.asarray(snapshot["q20_command"], np.float64),
    }
    array_keys = ("head_rgb", "history_rgb", "request_keypoints",
                  "response_keypoints", "response_keypoint_time_s",
                  "response_proprio_keypoints")
    for key in array_keys:
        if trace.get(key) is not None:
            arrays["model_" + key] = np.asarray(trace[key])
    if plan is not None:
        arrays.update({
            "plan_times_s": np.asarray(plan["times_s"], np.float64),
            "plan_eef_xyzw": np.asarray(plan["eef"], np.float64),
            "plan_q20_rad": np.asarray(plan["q20"], np.float64),
            "plan_q16_rad": np.asarray(plan["q16"], np.float64),
        })
    npz_path = plans / (stem + ".npz")
    # Uncompressed write is intentional: audit persistence runs beside a
    # real-time sender and must not spend CPU compressing camera noise.
    np.savez(str(npz_path), **arrays)
    metadata = {
        "schema": SCHEMA, "kind": "inference_and_plan", "plan_id": int(plan_id),
        "created_time_ns": time.time_ns(), "sample_ns": int(snapshot["sample_ns"]),
        "status": "planned" if answer.get("ok") else "rejected",
        "reason": None if answer.get("ok") else answer.get("reason"),
        "unexpected": bool(answer.get("unexpected", False)),
        "history_sent": trace.get("history_rgb") is not None,
        "inference": _metadata_without_arrays(trace),
        "planner_report": None if plan is None else plan.get("report"),
        "npz": str(npz_path),
        "array_keys": sorted(arrays),
    }
    json_path = plans / (stem + ".json")
    save_json(json_path, metadata)
    return {"plan_id": int(plan_id), "npz": str(npz_path), "json": str(json_path),
            "status": metadata["status"]}


class ExecutionAuditRecorder:
    """Collect one chunk's high-rate sends and 50 Hz measured feedback."""

    def __init__(self, output_dir, plan_id, plan, start_eef, start_q20, start_q16,
                 *, execute, arm_rate_hz, plan_artifact=None,
                 eef_position_limit_m=0.04, eef_rotation_limit_rad=0.20):
        self.output_dir = Path(output_dir)
        self.executions = self.output_dir / "audit_executions"
        self.executions.mkdir(parents=True, exist_ok=True)
        self.plan_id = int(plan_id)
        self.plan = plan
        self.plan_artifact = plan_artifact
        self.execute = bool(execute)
        self.arm_rate_hz = float(arm_rate_hz)
        self.eef_position_limit_m = float(eef_position_limit_m)
        self.eef_rotation_limit_rad = float(eef_rotation_limit_rad)
        self.start_monotonic_ns = time.monotonic_ns()
        self.start_time_ns = time.time_ns()
        self.start_eef = np.asarray(start_eef, np.float64).copy()
        self.start_q20 = np.asarray(start_q20, np.float64).copy()
        self.start_q16 = np.asarray(start_q16, np.float64).copy()
        self.times = np.r_[0.0, np.asarray(plan["times_s"], float)]
        self.q16 = np.concatenate([self.start_q16[None], np.asarray(plan["q16"], float)])
        self.sent = []
        self.feedback = []
        self.finalized = False

    def target_q16(self, elapsed_s):
        elapsed_s = float(np.clip(elapsed_s, 0.0, self.times[-1]))
        index = int(np.clip(np.searchsorted(self.times, elapsed_s, side="right") - 1,
                            0, len(self.times) - 2))
        denominator = max(self.times[index + 1] - self.times[index], 1e-9)
        weight = (elapsed_s - self.times[index]) / denominator
        return self.q16[index] + weight * (self.q16[index + 1] - self.q16[index])

    def record_send(self, elapsed_s, target_eef):
        self.sent.append((time.monotonic_ns(), time.time_ns(), float(elapsed_s),
                          np.asarray(target_eef, np.float64).copy()))

    def record_feedback(self, elapsed_s, target_eef, target_q20, measured_eef,
                        measured_q20, measured_q16, measured_dq16, *,
                        arm_guard_failure=False):
        target_eef = np.asarray(target_eef, np.float64)
        target_q20 = np.asarray(target_q20, np.float64)
        measured_eef = np.asarray(measured_eef, np.float64)
        measured_q20 = np.asarray(measured_q20, np.float64)
        measured_q16 = np.asarray(measured_q16, np.float64)
        measured_dq16 = np.asarray(measured_dq16, np.float64)
        target_q16 = self.target_q16(elapsed_s)
        eef_position, eef_rotation = _pose_errors(measured_eef, target_eef)
        self.feedback.append((time.monotonic_ns(), time.time_ns(), float(elapsed_s),
                              target_eef.copy(), target_q20.copy(), target_q16.copy(),
                              measured_eef.copy(), measured_q20.copy(), measured_q16.copy(),
                              measured_dq16.copy(), eef_position, eef_rotation,
                              np.abs(measured_q20 - target_q20),
                              np.abs(measured_q16 - target_q16),
                              bool(arm_guard_failure)))

    def finalize(self, status, *, completed):
        if self.finalized:
            return self.summary
        self.finalized = True
        stem = "exec%04d" % self.plan_id
        arrays = {
            "plan_times_s": np.asarray(self.plan["times_s"], np.float64),
            "plan_eef_xyzw": np.asarray(self.plan["eef"], np.float64),
            "plan_q20_rad": np.asarray(self.plan["q20"], np.float64),
            "plan_q16_rad": np.asarray(self.plan["q16"], np.float64),
        }
        if self.sent:
            arrays.update({
                "send_monotonic_ns": np.asarray([row[0] for row in self.sent], np.int64),
                "send_time_ns": np.asarray([row[1] for row in self.sent], np.int64),
                "send_elapsed_s": np.asarray([row[2] for row in self.sent], np.float64),
                "send_eef_target_xyzw": np.asarray([row[3] for row in self.sent], np.float64),
            })
        if self.feedback:
            names = ("feedback_monotonic_ns", "feedback_time_ns", "feedback_elapsed_s",
                     "feedback_eef_target_xyzw", "feedback_q20_target_rad",
                     "feedback_q16_reference_rad", "feedback_eef_measured_xyzw",
                     "feedback_q20_measured_rad", "feedback_q16_measured_rad",
                     "feedback_dq16_measured_rad_s", "eef_position_error_m",
                     "eef_rotation_error_rad", "q20_abs_error_rad", "q16_abs_error_rad",
                     "feedback_arm_guard_failure")
            for index, name in enumerate(names):
                dtype = (np.int64 if index < 2 else
                         bool if name == "feedback_arm_guard_failure" else np.float64)
                arrays[name] = np.asarray([row[index] for row in self.feedback], dtype=dtype)
        npz_path = self.executions / (stem + ".npz")
        np.savez(str(npz_path), **arrays)

        send_period_ms = (np.diff(arrays["send_monotonic_ns"]) / 1e6
                          if "send_monotonic_ns" in arrays else np.asarray([]))
        feedback_period_ms = (np.diff(arrays["feedback_monotonic_ns"]) / 1e6
                              if "feedback_monotonic_ns" in arrays else np.asarray([]))
        eef_position = arrays.get("eef_position_error_m", np.empty((0, 2)))
        eef_rotation = arrays.get("eef_rotation_error_rad", np.empty((0, 2)))
        q20_error = arrays.get("q20_abs_error_rad", np.empty((0, 2, 20)))
        q16_error = arrays.get("q16_abs_error_rad", np.empty((0, 16)))
        guard_ok = bool(len(eef_position) and np.max(eef_position) <= self.eef_position_limit_m
                        and np.max(eef_rotation) <= self.eef_rotation_limit_rad)
        valid = bool(self.execute and completed and len(self.feedback))
        if not self.execute:
            diagnosis = "shadow_only_no_tracking_judgement"
        elif not completed:
            diagnosis = "execution_interrupted_or_guard_rejected"
        elif guard_ok:
            diagnosis = "robot_feedback_followed_runtime_eef_limits"
        else:
            diagnosis = "robot_execution_tracking_deviation"
        self.summary = {
            "schema": SCHEMA, "kind": "execution_feedback", "plan_id": self.plan_id,
            "execute": self.execute, "tracking_comparison_valid": valid,
            "status": str(status), "completed_chunk": bool(completed),
            "start_time_ns": self.start_time_ns,
            "start_monotonic_ns": self.start_monotonic_ns,
            "duration_recorded_s": (float(self.feedback[-1][2]) if self.feedback else 0.0),
            "plan_duration_s": float(self.plan["times_s"][-1]),
            "send_count": len(self.sent), "feedback_count": len(self.feedback),
            "arm_guard_failure_sample_count": int(np.count_nonzero(
                arrays.get("feedback_arm_guard_failure", np.asarray([], bool)))),
            "expected_arm_send_rate_hz": self.arm_rate_hz,
            "send_period_ms": _stats(send_period_ms),
            "send_gap_count_over_2x_period": int(np.count_nonzero(
                send_period_ms > 2000.0 / self.arm_rate_hz)),
            "feedback_period_ms": _stats(feedback_period_ms),
            "feedback_gap_count_over_40ms": int(np.count_nonzero(feedback_period_ms > 40.0)),
            "eef_position_error_m": _stats(eef_position),
            "eef_rotation_error_rad": _stats(eef_rotation),
            "q20_abs_error_rad": _stats(q20_error),
            "q16_abs_error_rad_reference_only": _stats(q16_error),
            "per_side_eef_position_max_m": (None if not len(eef_position)
                                              else dict(zip(SIDES, map(float, np.max(eef_position, axis=0))))),
            "per_side_eef_rotation_max_rad": (None if not len(eef_rotation)
                                                else dict(zip(SIDES, map(float, np.max(eef_rotation, axis=0))))),
            "runtime_eef_limits": {"position_m": self.eef_position_limit_m,
                                   "rotation_rad": self.eef_rotation_limit_rad},
            "within_runtime_eef_limits": guard_ok if valid else None,
            "diagnosis_hint": diagnosis,
            "plan_artifact": self.plan_artifact,
            "npz": str(npz_path),
            "notes": {"q16": "reference IK path; ServoP may choose another redundant IK branch",
                      "model_vs_execution": "inspect the linked plan artifact first; if its predicted and planned motion is correct but feedback errors are large, investigate execution"},
        }
        json_path = self.executions / (stem + ".json")
        self.summary["json"] = str(json_path)
        save_json(json_path, self.summary)
        return self.summary


class AsyncAuditSink:
    """Single background writer so filesystem latency never blocks ServoP."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.executor = ThreadPoolExecutor(max_workers=1,
                                           thread_name_prefix="xflow-audit-writer")
        self.jobs = []

    def submit_plan(self, plan_id, snapshot, answer):
        stem = "plan%04d" % int(plan_id)
        status = "planned" if answer.get("ok") else "rejected"
        artifact = {"plan_id": int(plan_id),
                    "npz": str(self.output_dir / "audit_plans" / (stem + ".npz")),
                    "json": str(self.output_dir / "audit_plans" / (stem + ".json")),
                    "status": status}
        future = self.executor.submit(save_plan_audit, self.output_dir, plan_id,
                                      snapshot, answer)
        self.jobs.append(("plan", int(plan_id), future))
        return artifact

    def submit_execution(self, recorder, status, completed):
        plan_id = recorder.plan_id
        future = self.executor.submit(recorder.finalize, status, completed=completed)
        self.jobs.append(("execution", plan_id, future))

    def close(self):
        self.executor.shutdown(wait=True)
        output = {"plans": [], "executions": [], "errors": []}
        for kind, plan_id, future in self.jobs:
            try:
                output["plans" if kind == "plan" else "executions"].append(
                    future.result())
            except BaseException as error:
                output["errors"].append({"kind": kind, "plan_id": plan_id,
                                         "error": type(error).__name__ + ": " + str(error)})
        output["plans"].sort(key=lambda value: value["plan_id"])
        output["executions"].sort(key=lambda value: value["plan_id"])
        return output


def build_run_summary(output_dir):
    """Pair plan and execution records and write a quick diagnosis index."""
    output_dir = Path(output_dir)
    plans = {}
    plan_dir = output_dir / "audit_plans"
    plan_paths = sorted(plan_dir.glob("plan*.json")) if plan_dir.is_dir() else []
    for path in plan_paths:
        value = json.loads(path.read_text())
        plans[int(value["plan_id"])] = value
    executions = {}
    execution_dir = output_dir / "audit_executions"
    execution_paths = (sorted(execution_dir.glob("exec*.json"))
                       if execution_dir.is_dir() else [])
    for path in execution_paths:
        value = json.loads(path.read_text())
        executions[int(value["plan_id"])] = value
    rows = []
    for plan_id in sorted(set(plans) | set(executions)):
        plan = plans.get(plan_id)
        execution = executions.get(plan_id)
        if plan is not None and plan["status"] == "rejected":
            classification = "model_or_planner_rejected_before_execution"
        elif execution is None:
            classification = "planned_but_not_executed_or_record_missing"
        elif not execution["execute"]:
            classification = "shadow_only"
        elif not execution["completed_chunk"]:
            classification = "execution_interrupted_or_guard_rejected"
        elif execution.get("within_runtime_eef_limits"):
            classification = "execution_followed_commands_check_model_or_plan"
        else:
            classification = "execution_tracking_deviation_check_robot"
        inference = {} if plan is None else plan.get("inference", {})
        rows.append({
            "plan_id": plan_id, "classification": classification,
            "plan_status": None if plan is None else plan["status"],
            "plan_reason": None if plan is None else plan.get("reason"),
            "model": inference.get("response_model", inference.get("model")),
            "deployment_id": inference.get("response_deployment_id",
                                            inference.get("deployment_id_expected")),
            "sample_ns": None if plan is None else plan.get("sample_ns"),
            "round_trip_ms": inference.get("response_round_trip_ms"),
            "execution_status": None if execution is None else execution["status"],
            "eef_position_error_m": None if execution is None else execution.get(
                "eef_position_error_m"),
            "eef_rotation_error_rad": None if execution is None else execution.get(
                "eef_rotation_error_rad"),
            "q20_abs_error_rad": None if execution is None else execution.get(
                "q20_abs_error_rad"),
            "plan_npz": None if plan is None else plan.get("npz"),
            "execution_json": None if execution is None else execution.get("json"),
        })
    summary = {"schema": SCHEMA, "kind": "run_index",
               "generated_time_ns": time.time_ns(),
               "plan_count": len(plans), "execution_count": len(executions),
               "records": rows,
               "interpretation": {
                   "model_or_plan": "inspect model_head_rgb, request/response keypoints and plan targets in audit_plans",
                   "robot_execution": "inspect sent targets, measured feedback and error arrays in audit_executions",
                   "warning": "automatic classification only separates tracking evidence; it cannot judge task semantics"}}
    path = output_dir / "audit_summary.json"
    save_json(path, summary)
    return {"path": str(path), "plan_count": len(plans),
            "execution_count": len(executions), "records": rows}
