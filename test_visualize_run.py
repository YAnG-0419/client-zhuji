import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

import numpy as np

from visualize_run import collect, png_bytes, project, render, rounded


HERE = Path(__file__).resolve().parent


def write_run(root: Path, *, with_execution=True, rejected=False):
    (root / "audit_plans").mkdir(parents=True)
    (root / "audit_executions").mkdir(parents=True)
    (root / "result.json").write_text(json.dumps({
        "execute": True, "stop_reason": "requested cycles completed",
        "arm_motion_commands_sent": 100,
        "args": {"prefetch": 0.5},
        "planner": {"model": "m", "task": "tidying", "contract": "realrobot",
                    "deployment_id": "abc", "speed_ratio": 1.0,
                    "image_mode": "virtual_cam0", "virtual_preset": "cam0_newk",
                    "shaping": {"enabled": True}},
    }))
    steps = 32
    image = np.zeros((320, 320, 3), np.uint8); image[100:200, 100:200] = 200
    keypoints = np.zeros((steps, 2, 21, 3))
    keypoints[..., 2] = 0.6
    keypoints[..., 0] = np.linspace(-0.1, 0.1, steps)[:, None, None]
    arrays = {
        "sample_q16_rad": np.zeros(16), "sample_eef_measured_xyzw": np.zeros((2, 7)),
        "sample_q20_measured_rad": np.zeros((2, 20)),
        "sample_eef_command_xyzw": np.zeros((2, 7)),
        "sample_q20_command_rad": np.zeros((2, 20)),
        "model_head_rgb": image, "model_request_keypoints": keypoints[0],
        "model_response_keypoints": keypoints.astype(np.float32),
        "model_response_keypoint_time_s": (np.arange(steps) + 1) / 30.0,
    }
    if not rejected:
        arrays.update({"plan_times_s": (np.arange(steps) + 1) / 30.0,
                       "plan_eef_xyzw": np.zeros((steps, 2, 7)),
                       "plan_q20_rad": np.zeros((steps, 2, 20)),
                       "plan_q16_rad": np.zeros((steps, 16))})
    np.savez(root / "audit_plans/plan0000.npz", **arrays)
    (root / "audit_plans/plan0000.json").write_text(json.dumps({
        "plan_id": 0, "status": "rejected" if rejected else "planned",
        "reason": "IK unreachable" if rejected else None, "history_sent": True,
        "inference": {"task": "tidying", "model": "m",
                      "response_round_trip_ms": 900.0,
                      "response_timings_ms": {"total": 790.0, "denoise": 640.0}},
        "planner_report": {"stage_ms": {"total_ms": 1800.0},
                           "timing": {"duration_s": 1.1, "stretch": 1.03,
                                      "binding": {"frame": 32}},
                           "anchor": {"left": {"local_offset_translation_norm_m": 0.004},
                                      "right": {"local_offset_translation_norm_m": 0.006}}},
    }))
    if with_execution:
        n = 40
        np.savez(root / "audit_executions/exec0000.npz",
                 plan_times_s=(np.arange(steps) + 1) / 30.0,
                 plan_eef_xyzw=np.zeros((steps, 2, 7)),
                 plan_q20_rad=np.zeros((steps, 2, 20)),
                 plan_q16_rad=np.zeros((steps, 16)),
                 feedback_elapsed_s=np.linspace(0, 1.1, n),
                 feedback_eef_target_xyzw=np.zeros((n, 2, 7)),
                 feedback_eef_measured_xyzw=np.zeros((n, 2, 7)),
                 eef_position_error_m=np.full((n, 2), 0.01),
                 q20_abs_error_rad=np.full((n, 2, 20), 0.02))
        (root / "audit_executions/exec0000.json").write_text(json.dumps({
            "plan_id": 0, "execute": True, "completed_chunk": True, "status": "completed",
            "duration_recorded_s": 1.1,
            "eef_position_error_m": {"p95": 0.012, "max": 0.015},
            "eef_rotation_error_rad": {"p95": 0.02, "max": 0.03},
            "q20_abs_error_rad": {"p95": 0.02, "max": 0.04}}))
    return root


class PngTests(unittest.TestCase):
    def test_png_round_trips_through_zlib(self):
        rgb = np.random.default_rng(0).integers(0, 255, (8, 5, 3), dtype=np.uint8)
        blob = png_bytes(rgb)
        self.assertTrue(blob.startswith(b"\x89PNG\r\n\x1a\n"))
        width, height, depth, colour = struct.unpack(">IIBB", blob[16:26])
        self.assertEqual((width, height, depth, colour), (5, 8, 8, 2))
        start = blob.index(b"IDAT") + 4
        length = struct.unpack(">I", blob[start - 8:start - 4])[0]
        raw = zlib.decompress(blob[start:start + length])
        rows = [raw[i * (1 + 5 * 3):(i + 1) * (1 + 5 * 3)] for i in range(8)]
        self.assertTrue(all(row[0] == 0 for row in rows))      # filter type 0
        np.testing.assert_array_equal(
            np.frombuffer(b"".join(row[1:] for row in rows), np.uint8).reshape(8, 5, 3), rgb)


class ProjectionTests(unittest.TestCase):
    def test_principal_point_and_behind_camera(self):
        K = np.array([[160.0, 0, 160.0], [0, 160.0, 160.0], [0, 0, 1.0]])
        points = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.0, -1.0]])
        pixels = project(points, K)
        np.testing.assert_allclose(pixels[0], [160.0, 160.0])
        np.testing.assert_allclose(pixels[1], [176.0, 160.0])
        self.assertTrue(np.all(np.isnan(pixels[2])))

    def test_rounded_emits_null_not_nan(self):
        self.assertEqual(rounded([1.23456, float("nan"), float("inf")], 2), [1.23, None, None])


class ReportTests(unittest.TestCase):
    def test_report_is_self_contained_and_json_safe(self):
        with tempfile.TemporaryDirectory() as folder:
            run = write_run(Path(folder) / "live-exec-test")
            out = render(run, HERE, run / "report.html")
            html = out.read_text()
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertNotIn("NaN", html)
        self.assertIn("data:image/png;base64,", html)
        self.assertIn("const DATA = {", html)

    def test_rejected_cycle_keeps_model_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            run = write_run(Path(folder) / "live-exec-test", with_execution=False,
                            rejected=True)
            payload = collect(run, HERE)
        cycle = payload["cycles"][0]
        self.assertEqual(cycle["status"], "rejected")
        self.assertEqual(cycle["reason"], "IK unreachable")
        self.assertIsNotNone(cycle["image"])            # what the model saw
        self.assertEqual(len(cycle["predicted_px"][0]), 32)
        self.assertNotIn("feedback", cycle)

    def test_executed_cycle_pairs_plan_and_feedback(self):
        with tempfile.TemporaryDirectory() as folder:
            run = write_run(Path(folder) / "live-exec-test")
            payload = collect(run, HERE)
        cycle = payload["cycles"][0]
        self.assertTrue(cycle["executed"])
        self.assertEqual(cycle["rtt_ms"], 900.0)
        self.assertEqual(cycle["anchor_mm"], [4.0, 6.0])
        self.assertEqual(len(cycle["feedback"]["t"]), 40)
        self.assertEqual(len(cycle["feedback"]["error_mm"]), 2)
        self.assertEqual(payload["prefetch"], 0.5)
        self.assertTrue(payload["shaping"]["enabled"])

    def test_keypoints_land_inside_the_stored_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            run = write_run(Path(folder) / "live-exec-test")
            payload = collect(run, HERE)
        pixels = np.asarray(payload["cycles"][0]["predicted_px"][0], dtype=float)
        self.assertTrue(np.all((pixels >= 0) & (pixels < 320)))

    def test_missing_records_are_reported_clearly(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder) / "empty"; run.mkdir()
            with self.assertRaises(SystemExit):
                collect(run, HERE)


if __name__ == "__main__":
    unittest.main()
