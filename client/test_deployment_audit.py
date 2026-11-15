import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from deployment_audit import (ExecutionAuditRecorder, build_run_summary,
                              save_plan_audit)


def poses(x=0.4):
    return np.asarray([[x, 0.2, 0.1, 0.0, 0.0, 0.0, 1.0],
                       [x, -0.2, 0.1, 0.0, 0.0, 0.0, 1.0]], float)


def sample_and_answer():
    snapshot = {"sample_ns": 123456789, "q16": np.zeros(16),
                "eef_measured": poses(), "q20_measured": np.zeros((2, 20)),
                "eef_command": poses(), "q20_command": np.zeros((2, 20))}
    plan = {"times_s": np.asarray([0.1, 0.2]),
            "eef": np.stack([poses(0.41), poses(0.42)]),
            "q20": np.zeros((2, 2, 20)), "q16": np.zeros((2, 16)),
            "predicted_keypoints": np.ones((2, 2, 21, 3), np.float32),
            "current_keypoints": np.zeros((2, 21, 3), np.float32),
            "report": {"rpc": {"round_trip_ms": 100.0}},
            "inference_trace": {
                "schema": "xflow_inference_trace_v1", "sample_ns": 123456789,
                "task": "flower", "model": "hh_a12",
                "response_deployment_id": "deploy-test", "response_round_trip_ms": 100.0,
                "head_rgb": np.zeros((320, 320, 3), np.uint8),
                "history_rgb": np.ones((320, 320, 3), np.uint8),
                "request_keypoints": np.zeros((2, 21, 3), np.float32),
                "response_keypoints": np.ones((32, 2, 21, 3), np.float32),
                "response_keypoint_time_s": np.arange(1, 33, dtype=np.float32) / 120,
                "response_proprio_keypoints": np.zeros((2, 21, 3), np.float32)}}
    return snapshot, {"ok": True, "plan": plan}


class DeploymentAuditTests(unittest.TestCase):
    def test_plan_artifact_contains_exact_model_io_and_plan(self):
        snapshot, answer = sample_and_answer()
        with tempfile.TemporaryDirectory() as folder:
            artifact = save_plan_audit(folder, 7, snapshot, answer)
            with np.load(artifact["npz"], allow_pickle=False) as data:
                self.assertEqual(data["model_head_rgb"].shape, (320, 320, 3))
                self.assertEqual(data["model_response_keypoints"].shape,
                                 (32, 2, 21, 3))
                np.testing.assert_array_equal(data["sample_q16_rad"], np.zeros(16))
                np.testing.assert_array_equal(data["plan_eef_xyzw"],
                                              answer["plan"]["eef"])
            metadata = json.loads(Path(artifact["json"]).read_text())
            self.assertEqual(metadata["plan_id"], 7)
            self.assertEqual(metadata["inference"]["response_deployment_id"],
                             "deploy-test")

    def test_rejected_plan_still_preserves_model_response(self):
        snapshot, successful = sample_and_answer()
        trace = successful["plan"]["inference_trace"]
        answer = {"ok": False, "reason": "workspace envelope rejected",
                  "inference_trace": trace}
        with tempfile.TemporaryDirectory() as folder:
            artifact = save_plan_audit(folder, 8, snapshot, answer)
            with np.load(artifact["npz"], allow_pickle=False) as data:
                self.assertIn("model_response_keypoints", data)
                self.assertNotIn("plan_eef_xyzw", data)
            metadata = json.loads(Path(artifact["json"]).read_text())
            self.assertEqual(metadata["status"], "rejected")
            self.assertEqual(metadata["reason"], "workspace envelope rejected")

    def test_execution_record_separates_commands_feedback_and_errors(self):
        snapshot, answer = sample_and_answer(); plan = answer["plan"]
        with tempfile.TemporaryDirectory() as folder:
            artifact = save_plan_audit(folder, 2, snapshot, answer)
            recorder = ExecutionAuditRecorder(
                folder, 2, plan, poses(), np.zeros((2, 20)), np.zeros(16),
                execute=True, arm_rate_hz=300.0, plan_artifact=artifact)
            recorder.record_send(0.1, plan["eef"][0])
            recorder.record_feedback(0.1, plan["eef"][0], plan["q20"][0],
                                     plan["eef"][0], plan["q20"][0],
                                     np.zeros(16), np.zeros(16))
            summary = recorder.finalize("completed", completed=True)
            self.assertEqual(summary["eef_position_error_m"]["max"], 0.0)
            self.assertEqual(summary["q20_abs_error_rad"]["max"], 0.0)
            self.assertTrue(summary["within_runtime_eef_limits"])
            with np.load(summary["npz"], allow_pickle=False) as data:
                self.assertIn("send_eef_target_xyzw", data)
                self.assertIn("feedback_eef_measured_xyzw", data)
                self.assertIn("feedback_q20_measured_rad", data)
            index = build_run_summary(folder)
            self.assertEqual(index["records"][0]["classification"],
                             "execution_followed_commands_check_model_or_plan")

    def test_tracking_deviation_is_classified_as_execution_issue(self):
        snapshot, answer = sample_and_answer(); plan = answer["plan"]
        with tempfile.TemporaryDirectory() as folder:
            save_plan_audit(folder, 3, snapshot, answer)
            recorder = ExecutionAuditRecorder(
                folder, 3, plan, poses(), np.zeros((2, 20)), np.zeros(16),
                execute=True, arm_rate_hz=300.0)
            measured = plan["eef"][0].copy(); measured[0, 0] += 0.1
            recorder.record_feedback(0.1, plan["eef"][0], plan["q20"][0],
                                     measured, plan["q20"][0],
                                     np.zeros(16), np.zeros(16),
                                     arm_guard_failure=True)
            summary = recorder.finalize("completed", completed=True)
            self.assertFalse(summary["within_runtime_eef_limits"])
            self.assertEqual(summary["arm_guard_failure_sample_count"], 1)
            self.assertEqual(summary["diagnosis_hint"],
                             "robot_execution_tracking_deviation")


if __name__ == "__main__":
    unittest.main()
