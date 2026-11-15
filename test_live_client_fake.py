"""Exercise live_client.run() --execute end to end against fake hardware."""
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import live_client


START = np.array([[0.40, 0.25, 0.02, *Rotation.from_euler("x", 0.1).as_quat()],
                  [0.42, -0.22, 0.07, *Rotation.from_euler("x", -0.1).as_quat()]])


class FakeRobot:
    def __init__(self, *args, **kwargs):
        self.info = {"accid": "DACH_TRON2A_447", "data": {"working_mode": "developer_mode"}}
        self.track = getattr(FakeRobot, "default_track", True)
        self.qstate = {"q": [0.0] * 16, "dq": [0.0] * 16}
        self.armed = False; self.commands_sent = 0; self.lease_at = 0.0
        self.backend = None; self.sent = []
        self.eef = START.copy()
        self.pose = self._pose()
        self.q_at = self.pose_at = time.monotonic()

    def _pose(self):
        pose = {s + "_position": list(self.eef[i, :3]) for i, s in enumerate(("left", "right"))}
        pose.update({s + "_quat": [self.eef[i, 6], *self.eef[i, 3:6]] for i, s in enumerate(("left", "right"))})
        return pose

    def snapshot(self):
        return {"q16": [0.0] * 16, "eef_base_xyz_xyzw": START.tolist(), "robot_info": self.info}

    def start_feedback(self):
        pass

    def healthy(self):
        self.pose = self._pose(); self.q_at = time.monotonic()
        return np.zeros(16), np.zeros(16)

    def arm_servop(self):
        self.armed = True; self.backend = "supervised_servop"

    def send(self, title, data, guid=None):
        assert title in ("request_servop", "request_emgy_stop")
        assert title != "request_servop" or self.armed
        if title == "request_emgy_stop":
            raise AssertionError("request_emgy_stop must never be sent")
        if not self.track:
            return
        for i, key in enumerate(("left_pos", "right_pos")):
            v = np.asarray(data[key])
            self.eef[i] = np.r_[v[:3], v[4:], v[3]]   # perfect tracking, wxyz -> xyzw
        self.sent.append(self.eef.copy())

    def emergency_stop(self):
        self.send("request_emgy_stop", {})

    def close(self):
        pass


class FakeHands:
    touched = False

    def __init__(self, target):
        self.node = types.SimpleNamespace(count_publishers=lambda topic: 1)

    def feedback(self, timeout=0):
        return {"left": [0.1] * 20, "right": [0.1] * 20}

    def validate_owner(self):
        pass

    def start(self):
        self.touched = True

    def stop(self):
        self.stopped = True

    def close(self):
        pass


class FakeHandSession:
    def __init__(self, hands, limits, out, result, tracking_log_only_stages=()):
        self.last = np.full((2, 20), 0.1); self.log = open(Path(out) / "hand.jsonl", "w")
        self.targets = []

    def tick(self, desired, stage, source_time=None):
        self.last = np.asarray(desired, float).copy(); self.targets.append(self.last.copy())
        return self.last.copy()


class FakeArmSession:
    def __init__(self, robot, model, t, eef, out, feedback_limits=None):
        self.robot = robot; self.feedback_log = open(Path(out) / "arm.jsonl", "w")

    def tick(self, desired, start, goal, source_time=None):
        q, dq = self.robot.healthy()
        return q, dq, 0.0, None


class FakeCamera:
    error = None

    def __init__(self, rclpy):
        self.camera_info = {"width": 640}

    def wait_ready(self):
        pass

    def set_execution_active(self, active):
        pass

    def sample(self, not_before, frames):
        return {"sample_ns": time.time_ns(), "head_rgb": np.zeros((2, 2, 3), np.uint8),
                "history_rgb": None, "history_missing_reason": "test", "camera_info": {}}

    def close(self):
        pass


class FakeConnection:
    """Planner stand-in: 32 steps moving the left hand along +x."""

    def __init__(self, distance=0.02):
        self.request = None; self.distance = distance

    def send(self, message):
        if message is not None:
            self.request = message

    def poll(self):
        return self.request is not None

    def recv(self):
        request, self.request = self.request, None
        eef = np.repeat(np.asarray(request["eef_command"])[None], 32, axis=0)
        eef[:, 0, 0] += np.linspace(0.0, self.distance, 32)
        return {"ok": True, "plan": {
            "times_s": (np.arange(32) + 1) / 30.0, "eef": eef,
            "q20": np.repeat(np.asarray(request["q20_command"])[None], 32, axis=0),
            "q16": np.zeros((32, 16)), "predicted_keypoints": np.zeros((32, 2, 21, 3)),
            "current_keypoints": np.zeros((2, 21, 3)),
            "report": {"rpc": {"round_trip_ms": 1.0}, "stage_ms": {}, "timing": {
                "duration_s": 32 / 30.0, "stretch": 1.0, "binding": {"frame": 32}},
                "anchor": {s: {"local_offset_translation_norm_m": 0.0, "local_offset_rotation_rad": 0.0,
                               "q_offset_abs_max_rad": 0.0, "decay_steps": 12} for s in ("left", "right")}}}}


def install_fakes(state):
    model = types.SimpleNamespace(lower=np.full(16, -3.0), upper=np.full(16, 3.0),
                                  set_q=lambda q: None, poses=lambda: START.copy(),
                                  model=types.SimpleNamespace(jnt_range=np.tile([-3.0, 3.0], (16, 1))),
                                  jids=np.arange(16))
    modules = {
        "transport": dict(replay_backend=lambda: (np, lambda: model, FakeRobot)),
        "replay_servop": dict(servo_transport=lambda T: T, QuatTrack=__import__("replay_quat").QuatTrack),
        "robot_home": dict(Hands=FakeHands, process_commands=lambda: ["wuji-hand-2-driver"]),
        "home_policy": dict(require_status=lambda info: None),
        "hand_only": dict(HandSession=FakeHandSession, observed_hands_type=lambda H: H,
                          limits_from_install=lambda: (np.tile([-1.0, 2.0], (2, 20, 1)), {}),
                          stop_hand_stack=lambda stack: None),
        "arm_only": dict(ArmSession=FakeArmSession),
        "movej_prepare": dict(check_servop_box=lambda poses: None),
        "wire_trace": dict(attach_trace=lambda robot: None, save_trace=lambda robot, out, result: None),
        "episode": dict(hand_names=lambda side: ["%s_finger%d_joint%d" % (side, f, j)
                                                 for f in range(1, 6) for j in range(1, 5)]),
        "rclpy": dict(init=lambda: None, shutdown=lambda: None),
    }
    saved = {}
    for name, attrs in modules.items():
        saved[name] = sys.modules.get(name)
        sys.modules[name] = types.SimpleNamespace(**attrs)
    return saved


def run_fake(robot_track=True, distance=0.02, cycles=2):
    quat_module = types.ModuleType("replay_quat")

    class QuatTrack:
        def __call__(self, poses):
            return [np.r_[p[:3], p[6], p[3:6]] for p in poses]
    quat_module.QuatTrack = QuatTrack
    sys.modules["replay_quat"] = quat_module
    state = {}
    saved = install_fakes(state)
    FakeRobot.default_track = robot_track
    originals = (live_client.start_planner, live_client.CameraFeed,
                 live_client.validate_camera_info, live_client.camera_info_from_json)
    connection = FakeConnection(distance)
    live_client.start_planner = lambda args: (types.SimpleNamespace(join=lambda timeout: None,
                                                                     is_alive=lambda: False),
                                              connection, {"fake": True, "history_execution_frames": 16})
    live_client.CameraFeed = FakeCamera
    live_client.validate_camera_info = lambda a, b: {}
    live_client.camera_info_from_json = lambda path: {}
    result = {"hand_target_updates": 0}
    error = None
    try:
        with tempfile.TemporaryDirectory() as folder:
            args = types.SimpleNamespace(task="flower", model="hh_a12", execute=True, step=False,
                                         cycles=cycles, steps=32, image_mode="virtual_cam0",
                                         virtual_preset="cam0_newk", denoise_steps=10, arm_cap=0.8,
                                         hand_cap=0.9, linear_cap=0.25, angular_cap=1.0,
                                         settle=0.05, rate=300.0, max_failures=3, relay_port=28765,
                                         speed_ratio=None)
            try:
                live_client.run(args, Path(folder), result)
            except Exception as exc:
                error = exc
    finally:
        (live_client.start_planner, live_client.CameraFeed,
         live_client.validate_camera_info, live_client.camera_info_from_json) = originals
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return result, error


class LiveClientFakeHardwareTest(unittest.TestCase):
    def test_execute_two_cycles_ends_in_controller_hold(self):
        result, error = run_fake()
        self.assertIsNone(error)
        self.assertTrue(result["completed"], result)
        self.assertEqual(result["cycles_executed"], 2)
        self.assertEqual(result["arm_stop"], "controller_hold")
        self.assertFalse(result["controller_hold"]["emergency_stop_sent"])
        self.assertTrue(result["controller_hold"]["developer_mode"])
        self.assertGreater(result["arm_motion_commands_sent"], 500)
        self.assertEqual(result["audit"]["plan_count"], 2)
        self.assertEqual(result["audit"]["execution_count"], 2)
        self.assertTrue(all(item["tracking_comparison_valid"]
                            for item in result["audit"]["executions"]))
        self.assertTrue(all(item["feedback_count"] > 20
                            for item in result["audit"]["executions"]))

    def test_guard_fault_holds_without_emergency_stop(self):
        # The robot ignores commands, so the EEF tracking guard must trip.
        result, error = run_fake(robot_track=False, distance=0.10, cycles=1)
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("tracking", str(error))
        self.assertEqual(result["arm_stop"], "controller_hold")
        self.assertFalse(result["controller_hold"]["emergency_stop_sent"])
        self.assertEqual(result["emergency_stops_suppressed"], 0)
        self.assertEqual(result["audit"]["execution_count"], 1)
        execution = result["audit"]["executions"][0]
        self.assertEqual(execution["arm_guard_failure_sample_count"], 1)
        self.assertGreater(execution["eef_position_error_m"]["max"], 0.04)


if __name__ == "__main__":
    unittest.main()
