#!/usr/bin/env python3
"""XFlow online client for zhuji (TRON2 ServoP arms + Wuji Hand 2).

Cycle (sync-hold): hold the commanded pose -> sample camera + SDK/Wuji state ->
plan in a separate process (XFlow inference, retarget, anchor, IK/collision)
-> execute the 32-step chunk at 30 Hz (time-stretched only where a speed cap
binds) -> hold and repeat.

Without --execute this is a live shadow run: the robot transport is read-only,
hands are never enabled and no ServoP/Wuji target is sent.  With --execute the
arms are streamed through supervised ServoP and the hands through wuji_control,
reusing the replay_hand guards (feedback speed, live clearance, joint range,
hand tracking).  Every stop, including faults, only stops ServoP streaming so
the controller holds in High-Level Dev mode; request_emgy_stop is never sent.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import multiprocessing
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

CLIENT = Path(__file__).resolve().parent
ROOT = CLIENT.parent
for _path in (CLIENT, ROOT / "replay_hand", ROOT / "replay", ROOT / "tools", ROOT):
    if str(_path) not in sys.path:
        sys.path.append(str(_path))

from image_preprocess import camera_info_dict, camera_info_from_json, validate_camera_info  # noqa: E402
from live_planner import worker_main  # noqa: E402
from motion_shaping import build_splice  # noqa: E402
from deployment_audit import (AsyncAuditSink, ExecutionAuditRecorder,
                              SCHEMA as AUDIT_SCHEMA, build_run_summary)  # noqa: E402

SIDES = ("left", "right")
LOCK = "/home/guest/wanchi/replay_tip_align_test_1030/.hands_replay.lock"
CAMERA_ARCHIVE = ROOT / "calibration/camera/color_camera_info_live.json"
EEF_TRACK_POS, EEF_TRACK_ROT = 0.04, 0.20      # replay_hand execution_cross limits
# Before sampling the next chunk, keep holding until the arms are within
# 2 cm / 3 deg of the command; abort only after 3 s of waiting.  Beyond the
# 4 cm / 0.2 rad tracking guard the run stops immediately (checked every tick).
HOLD_SETTLE_POS, HOLD_SETTLE_ROT, HOLD_SETTLE_TIMEOUT = 0.02, math.radians(3.0), 3.0
COMMAND_STEP_POS, COMMAND_STEP_ROT = 0.006, 0.03  # per 300 Hz send; bug catcher
OWNERS = ("wuji_teleop.switch", "wuji-teleop-switch", "tools/manus_teleop.sh", "robot_home.py",
          "axis_replay.py", "replay.py", "servo_test.py", "replay_hand.py",
          "replay_hand_pinocchio.py", "replay_servop.py", "replay_full.py",
          "replay_arm_only.py", "replay_hands_only.py")


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False,
                                     default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v)) + "\n")


def sdk_pose_xyzw(pose):
    return np.array([pose[s + "_position"] + pose[s + "_quat"][1:] + pose[s + "_quat"][:1]
                     for s in SIDES], dtype=float)


def pose_errors(a, b):
    position = np.linalg.norm(a[:, :3] - b[:, :3], axis=1)
    rotation = np.array([(Rotation.from_quat(a[i, 3:]).inv() * Rotation.from_quat(b[i, 3:])).magnitude()
                         for i in range(len(a))])
    return position, rotation


def slerp_xyzw(q0, q1, u):
    q0 = np.asarray(q0, float); q1 = np.asarray(q1, float)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        out = q0 + u * (q1 - q0)
    else:
        theta = math.acos(min(1.0, dot))
        out = (math.sin((1 - u) * theta) * q0 + math.sin(u * theta) * q1) / math.sin(theta)
    return out / np.linalg.norm(out)


def splice_plan(active, exec_start, sample, plan, caps, *, lead_s=0.05):
    """Continue the running chunk with a plan sampled while it was executing.

    Returns ``{"plan", "start", "info"}`` shaped like a planner result so the
    executor can swap in a new ChunkSampler, or ``None`` when the remaining plan
    is unusable or cannot be connected inside the caps.
    """
    dispatch = time.monotonic() + lead_s - exec_start
    moving = dispatch < active.duration - 1e-3
    progress = active.model_at(min(dispatch, active.duration)) \
        - active.model_at(sample["sample_monotonic"] - exec_start)
    if progress < 0:
        return None
    current_eef, current_q20 = active(dispatch)
    splice = build_splice(
        progress_model_s=progress,
        plan_model_t=np.asarray(plan["model_t"], float),
        plan_eef=np.asarray(plan["eef"], float), plan_q20=np.asarray(plan["q20"], float),
        plan_q16=np.asarray(plan["q16"], float), current_eef=current_eef,
        current_q20=current_q20, current_eef_velocity=active.velocity(dispatch),
        caps=caps, moving=moving)
    if splice is None:
        return None
    spliced = dict(plan)
    spliced.update(times_s=splice.times, model_t=splice.model_t, eef=splice.eef,
                   q20=splice.q20, _start_model_t=float(progress))
    spliced["report"] = dict(plan["report"], splice={
        "progress_model_s": splice.progress_model_s, "skipped_points": splice.skipped_points,
        "slowdown": splice.slowdown, "blend_s": splice.blend_s, "moving": moving,
        "peaks": splice.peaks, "dispatch_into_chunk_s": float(dispatch)})
    return {"plan": spliced, "start": (current_eef, current_q20),
            "info": spliced["report"]["splice"]}


def no_emergency_stop_transport(base):
    """Transport that never sends request_emgy_stop (operator decision 2026-11-14).

    Every stop -- normal end, Ctrl-C, guard trip, lease expiry, close -- becomes
    "stop streaming ServoP": the TRON2 controller latches the current pose after
    its ServoP command timeout and stays in High-Level Dev mode.  The physical
    e-stop on site remains the operator's backstop.
    """
    class NoEmergencyStopTransport(base):
        suppressed_emergency_stops = 0

        def emergency_stop(self):
            # Reached from inherited close()/watchdog paths: only stop streaming.
            self.armed = False
            self.suppressed_emergency_stops += 1

        def watch(self):
            while not self.closed.wait(.01):
                if self.armed and (self.fault or time.monotonic() - self.lease_at > .1):
                    self.fault = self.fault or "ServoP command lease expired"
                    self.armed = False
                    self.suppressed_emergency_stops += 1

        def send(self, title, data, guid=None):
            if title == "request_emgy_stop":
                raise RuntimeError("request_emgy_stop is disabled for the XFlow client")
            return super().send(title, data, guid)
    return NoEmergencyStopTransport


ENVELOPE_RECOVERY_MAX = 0.05     # rad; larger excursions stop instead of recovering
ENVELOPE_RECOVERY_INSIDE = 0.02  # rad; recovered pose sits this far inside the envelope
ENVELOPE_RECOVERY_SECONDS = 1.5


def recover_into_envelope(robot, model, session, hand_session, quat, rate, result):
    """Operator decision 2026-11-16: if a measured arm joint is slightly outside
    the SDK command envelope (ServoP overshoot at a limit), slowly stream ServoP
    to the pose with those joints clamped back inside, then verify.  Nothing is
    relaxed: certification still requires the measured start inside the envelope.
    Returns the commanded EEF to hold afterwards (or None when not needed).
    """
    q, _ = robot.healthy()
    lower, upper = np.asarray(model.lower[:14], float), np.asarray(model.upper[:14], float)
    excursion = np.maximum(np.maximum(lower - q[:14], q[:14] - upper), 0.0)
    if float(excursion.max()) <= 0.0:
        return None
    report = dict(joints=[int(j) for j in np.flatnonzero(excursion > 0)],
                  excursion_rad=float(excursion.max()), measured_q16=list(map(float, q)))
    result["envelope_recovery"] = report
    if excursion.max() > ENVELOPE_RECOVERY_MAX:
        raise RuntimeError("arm joint %.3f rad outside the command envelope; recover manually"
                           % excursion.max())
    target_q = q.copy()
    target_q[:14] = np.clip(q[:14], lower + ENVELOPE_RECOVERY_INSIDE, upper - ENVELOPE_RECOVERY_INSIDE)
    model.set_q(target_q)
    goal = np.asarray(model.poses(), float).copy()
    start = sdk_pose_xyzw(robot.pose)
    print("ENVELOPE RECOVERY: joints %s %.4f rad outside; moving back over %.1f s"
          % (report["joints"], excursion.max(), ENVELOPE_RECOVERY_SECONDS), flush=True)
    period, began, n, last_tick = 1.0 / rate, time.monotonic(), 0, 0.0
    hold_q20 = hand_session.last.copy()
    while True:
        now = time.monotonic()
        elapsed = now - began
        u = min(1.0, elapsed / ENVELOPE_RECOVERY_SECONDS)
        w = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        target = np.empty((2, 7))
        for side in range(2):
            target[side, :3] = start[side, :3] + w * (goal[side, :3] - start[side, :3])
            target[side, 3:] = slerp_xyzw(start[side, 3:], goal[side, 3:], w)
        left, right = quat(target)
        robot.send("request_servop", {"left_pos": left.tolist(), "right_pos": right.tolist()})
        robot.commands_sent += 1
        robot.lease_at = time.monotonic()
        if now - last_tick >= 0.02:
            last_tick = now
            q, dq, _, _ = session.tick(None, q, q)
            hand_session.tick(hold_q20, "xflow_hold")
            ep, er = pose_errors(sdk_pose_xyzw(robot.pose), target)
            if max(ep) > EEF_TRACK_POS or max(er) > EEF_TRACK_ROT:
                raise RuntimeError("EEF tracking error %.4f m during envelope recovery" % max(ep))
            inside = np.all(q[:14] >= lower) and np.all(q[:14] <= upper)
            if u >= 1.0 and inside and np.max(np.abs(dq[:14])) < 0.02:
                report.update(completed=True, elapsed_s=elapsed, final_q16=list(map(float, q)))
                print("ENVELOPE RECOVERY done in %.1f s" % elapsed, flush=True)
                return goal
            if elapsed > ENVELOPE_RECOVERY_SECONDS + 2.0:
                raise RuntimeError("envelope recovery did not bring the arm inside the envelope")
        n += 1
        deadline = began + n * period
        while time.monotonic() < deadline:
            time.sleep(min(0.0005, max(0.0, deadline - time.monotonic())))


def controller_hold(robot, result, reason, timeout_s=3.0):
    """Stop streaming and verify the controller hold; never sends an emergency stop.

    Reads the feedback cache directly so a transport fault (e.g. lease expiry)
    does not prevent verification.  Never raises; the outcome is recorded.
    """
    robot.armed = False
    report = dict(reason=str(reason), emergency_stop_sent=False, timeout_s=timeout_s,
                  required_stable_s=0.2, completed=False)
    result["controller_hold"] = report
    began, stable_since = time.monotonic(), None
    while True:
        now = time.monotonic()
        state, info = getattr(robot, "qstate", None), getattr(robot, "info", None) or {}
        mode = info.get("data", {}).get("working_mode")
        report["working_mode"] = mode
        if state is not None and now - getattr(robot, "q_at", 0.0) <= 0.15:
            speed = float(np.max(np.abs(np.asarray(state["dq"], float)[:14])))
            report["last_max_arm_speed_rad_s"] = speed
            report["max_arm_speed_rad_s"] = max(report.get("max_arm_speed_rad_s", 0.0), speed)
            stable_since = (now if stable_since is None else stable_since) if speed < 0.02 else None
        else:
            report["feedback_stale"] = True
        if now - began >= 0.25 and stable_since is not None and now - stable_since >= 0.2:
            report.update(completed=True, elapsed_s=now - began,
                          developer_mode=(mode == "developer_mode"),
                          held_q16=list(map(float, state["q"])))
            result["arm_stop"] = "controller_hold"
            print("ARMS HOLDING (working_mode=%s)" % mode, flush=True)
            return report
        if now - began > timeout_s:
            report["elapsed_s"] = now - began
            result["arm_stop"] = "controller_hold_unverified"
            print("WARNING: controller hold not verified within %.1f s (mode=%s); "
                  "no emergency stop was sent - watch the robot" % (timeout_s, mode), flush=True)
            return report
        time.sleep(0.01)


class ChunkSampler:
    """Interpolates the commanded start pose and the planned steps on one clock.

    ``model_t`` is each point's time on the model's own timeline, measured from
    the observation this chunk was planned from; prefetch uses it to work out
    how much of a newly returned plan the robot has already covered.
    """

    def __init__(self, start_eef, start_q20, plan, *, start_model_t=0.0):
        self.times = np.r_[0.0, np.asarray(plan["times_s"], float)]
        self.eef = np.concatenate([np.asarray(start_eef, float)[None], np.asarray(plan["eef"], float)])
        self.q20 = np.concatenate([np.asarray(start_q20, float)[None], np.asarray(plan["q20"], float)])
        model = plan.get("model_t")
        model = np.asarray(plan["times_s"], float) if model is None else np.asarray(model, float)
        self.model_t = np.r_[float(start_model_t), model[:len(self.times) - 1]]
        self.duration = float(self.times[-1])
        self.exec_start = None      # set by the executor when streaming begins

    def model_at(self, t):
        return float(np.interp(min(max(float(t), 0.0), self.duration), self.times, self.model_t))

    def velocity(self, t, step=0.02):
        """Finite-difference EEF translation velocity [2,3] at ``t`` seconds."""
        a = min(max(float(t) - 0.5 * step, 0.0), self.duration)
        b = min(max(float(t) + 0.5 * step, 0.0), self.duration)
        if b - a < 1e-6:
            return np.zeros((2, 3))
        return (self(b)[0][:, :3] - self(a)[0][:, :3]) / (b - a)

    def __call__(self, t):
        t = min(max(float(t), 0.0), self.duration)
        i = int(np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times) - 2))
        u = (t - self.times[i]) / max(self.times[i + 1] - self.times[i], 1e-9)
        eef = np.empty((2, 7))
        for side in range(2):
            eef[side, :3] = self.eef[i, side, :3] + u * (self.eef[i + 1, side, :3] - self.eef[i, side, :3])
            eef[side, 3:] = slerp_xyzw(self.eef[i, side, 3:], self.eef[i + 1, side, 3:], u)
        return eef, self.q20[i] + u * (self.q20[i + 1] - self.q20[i])


class CameraFeed:
    """Subscription-only head camera buffer with an execution clock for history."""

    def __init__(self, rclpy):
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
        self.node = rclpy.create_node("xflow_live_camera_%d" % os.getpid())
        self.lock = threading.Lock()
        self.images = []          # (stamp_ns, receive_monotonic, rgb, execution_ns)
        self.camera_info = None
        self.execution_active = False
        self.execution_ns = 0
        self.error = None
        self.node.create_subscription(Image, "/camera/top/color/image_raw", self._image,
                                      qos_profile_sensor_data)
        self.node.create_subscription(CameraInfo, "/camera/top/color/camera_info", self._info,
                                      qos_profile_sensor_data)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def _image(self, msg):
        try:
            if msg.encoding != "rgb8" or (msg.width, msg.height, msg.step) != (640, 480, 1920):
                raise ValueError("unexpected head image %s %dx%d" % (msg.encoding, msg.width, msg.height))
            stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            rgb = np.frombuffer(bytes(msg.data), np.uint8).reshape(480, 640, 3)
            with self.lock:
                if self.images and stamp <= self.images[-1][0]:
                    return
                if self.images and self.execution_active:
                    self.execution_ns += stamp - self.images[-1][0]
                self.images.append((stamp, time.monotonic(), rgb, self.execution_ns))
                while self.images and stamp - self.images[0][0] > 6_000_000_000:
                    self.images.pop(0)
        except BaseException as error:
            self.error = error

    def _info(self, msg):
        self.camera_info = camera_info_dict(msg.width, msg.height, msg.distortion_model,
                                            msg.k, msg.d, msg.header.frame_id)

    def set_execution_active(self, active):
        with self.lock:
            self.execution_active = bool(active)

    def wait_ready(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.error:
                raise RuntimeError("camera: " + str(self.error))
            if self.camera_info is not None and len(self.images) >= 5:
                return
            time.sleep(0.05)
        raise RuntimeError("no /camera/top/color/image_raw or camera_info; check the DDS environment")

    def sample(self, not_before_monotonic, history_frames):
        if self.error:
            raise RuntimeError("camera: " + str(self.error))
        with self.lock:
            fresh = [item for item in self.images if item[1] >= not_before_monotonic]
            if not fresh:
                return None
            current = fresh[-1]
            if time.monotonic() - current[1] > 0.15:
                raise RuntimeError("head image is stale")
            periods = np.diff([item[0] for item in self.images[-60:]])
            median = float(np.median(periods)) if len(periods) else 33_333_333.0
            tolerance = float(np.clip(median / 2, 16_700_000, 50_000_000))
            target = current[3] - history_frames / 30.0 * 1e9
            history, reason = None, None
            if target < 0:
                reason = "not enough executed time yet"
            else:
                best = min(self.images, key=lambda item: abs(item[3] - target))
                if abs(best[3] - target) > tolerance or best[0] == current[0]:
                    reason = "no image at the execution-clock offset"
                else:
                    history = best
        return {"sample_ns": int(current[0]), "sample_monotonic": float(current[1]),
                "head_rgb": current[2].copy(),
                "history_rgb": None if history is None else history[2].copy(),
                "history_sample_ns": None if history is None else int(history[0]),
                "history_missing_reason": reason, "camera_info": self.camera_info,
                "camera_fps": 1e9 / median}

    def close(self):
        self.executor.shutdown(timeout_sec=2)
        self.thread.join(timeout=2)
        self.node.destroy_node()


class StepGate:
    """--step: wait for Enter (execute) or q (quit) without blocking the sender."""

    def __init__(self):
        self.answer = None
        self.waiting = False

    def ask(self, text):
        self.answer, self.waiting = None, True
        print(text + "  [Enter=execute, s=skip, q=quit]", flush=True)

    def poll(self):
        if not self.waiting:
            return None
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip().lower()
            self.waiting = False
            return {"": "go", "s": "skip", "q": "quit"}.get(line, "skip")
        return None


def start_planner(args):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    kwargs = dict(root=str(ROOT), config_path=str(CLIENT / "deployment_config.json"),
                  model=args.model, task=args.task, relay_host="127.0.0.1",
                  relay_port=args.relay_port, image_mode=args.image_mode,
                  virtual_preset=args.virtual_preset, arm_cap=args.arm_cap,
                  hand_cap=args.hand_cap, linear_cap=args.linear_cap,
                  angular_cap=args.angular_cap, num_inference_steps=args.denoise_steps,
                  steps=args.steps, speed_ratio=args.speed_ratio,
                  shaping=None if args.shaping is None else {"enabled": args.shaping == "on"})
    process = context.Process(target=worker_main, args=(child, kwargs), daemon=True)
    process.start()
    if not parent.poll(180):
        process.terminate()
        raise RuntimeError("planner did not start within 180 s")
    hello = parent.recv()
    if not hello.get("ready"):
        raise RuntimeError("planner failed: " + hello.get("error", "unknown"))
    return process, parent, hello["describe"]


def run(args, out, result):
    from transport import replay_backend
    from replay_servop import servo_transport, QuatTrack
    from robot_home import Hands, process_commands
    from home_policy import require_status
    from hand_only import HandSession, observed_hands_type, limits_from_install, stop_hand_stack
    from arm_only import ArmSession
    from movej_prepare import check_servop_box
    from wire_trace import attach_trace, save_trace
    from episode import hand_names
    import rclpy

    robot = hands = stack = stack_log = session = hand_session = camera = None
    planner = connection = None
    execution_audit = None
    audit_sink = AsyncAuditSink(out)
    ros = False
    cycles_log = (out / "cycles.jsonl").open("w")
    result["audit"] = {"schema": AUDIT_SCHEMA, "plan_count": 0,
                       "execution_count": 0, "plans": [], "executions": [],
                       "writer": "single_background_thread_uncompressed_npz",
                       "tracking_records_are_diagnostic_only": not args.execute}

    def finish_execution_audit(status, completed):
        nonlocal execution_audit
        if execution_audit is None:
            return None
        audit_sink.submit_execution(execution_audit, status, completed)
        execution_audit = None
    try:
        commands = process_commands()
        if any(owner in c for c in commands for owner in OWNERS):
            raise RuntimeError("another robot controller is running; exit teleop/home/replay first")
        print("Starting planner (XFlow %s, task %s) ..." % (args.model, args.task), flush=True)
        planner, connection, describe = start_planner(args)
        result["planner"] = describe
        history_frames = int(describe["history_execution_frames"])
        print("PLANNER READY " + json.dumps(describe), flush=True)

        _, Model, Transport = replay_backend()
        model = Model()
        robot = no_emergency_stop_transport(servo_transport(Transport))(model.lower[:14], model.upper[:14])
        attach_trace(robot)
        state = robot.snapshot()
        save(out / "start_state.json", state)
        require_status(state["robot_info"])
        q0 = np.array(state["q16"])
        model.set_q(q0)
        start_eef = np.array(state["eef_base_xyz_xyzw"])
        position, rotation = pose_errors(model.poses(), start_eef)
        if max(position) > 0.001 or max(rotation) > 0.0035:
            raise RuntimeError("live model FK and SDK EEF feedback disagree")
        check_servop_box(start_eef)

        rclpy.init(); ros = True
        camera = CameraFeed(rclpy)
        names = {s + "_hand": dict.fromkeys(hand_names(s), 0.0) for s in SIDES}
        hands = observed_hands_type(Hands)(names)
        if not any("wuji-hand-2-driver" in c or "wuji-control" in c for c in commands):
            stack_log = (out / "stack.log").open("w")
            stack = subprocess.Popen(["ros2", "launch", str(ROOT / "tools/home_bimanual.launch.py")],
                                     stdout=stack_log, stderr=subprocess.STDOUT, start_new_session=True)
            print("STARTED disabled Wuji drivers (log %s)" % (out / "stack.log"), flush=True)
        q20_start = np.array([hands.feedback(timeout=25 if stack else 5)[s] for s in SIDES])
        hands.validate_owner()
        for s in SIDES:
            if hands.node.count_publishers("/wuji/" + s + "/command") != 1:
                raise RuntimeError("expected one wuji_control output publisher for " + s)
        limits, result["hand_limits"] = limits_from_install()
        camera.wait_ready()
        result["camera_gate"] = validate_camera_info(camera.camera_info, camera_info_from_json(CAMERA_ARCHIVE))
        robot.start_feedback()
        q, dq = robot.healthy()
        if np.max(abs(q - q0)) > 0.002 or np.max(abs(dq)) > 0.02:
            raise RuntimeError("robot moved during preflight")
        result["live_preflight_passed"] = True
        print("PREFLIGHT PASSED  mode=%s" % ("EXECUTE" if args.execute else "SHADOW (no commands)"), flush=True)

        if args.execute:
            print("ENABLING hands (program source) and arming ServoP hold at the measured pose", flush=True)
            hands.start()
            hand_session = HandSession(hands, limits, out, result,
                                       tracking_log_only_stages={"xflow_hold", "xflow_exec"})
            q20_start = hand_session.last.copy()
            q, dq = robot.healthy()
            if np.max(abs(q - q0)) > 0.002 or np.max(abs(dq)) > 0.02:
                raise RuntimeError("robot moved during hand enable")
            start_eef = sdk_pose_xyzw(robot.pose)
        # Created last: ArmSession requires <=0.1 s between supervision ticks.
        session = ArmSession(robot, model, np.array([0.0, 1.0]), np.stack([start_eef] * 2), out,
                             feedback_limits=model.model.jnt_range[model.jids])
        session.record_accuracy_only = True
        if args.execute:
            robot.arm_servop()
        quat = QuatTrack()
        if args.execute:
            recovered = recover_into_envelope(robot, model, session, hand_session, quat, args.rate, result)
            if recovered is not None:
                start_eef = recovered
        command_eef, command_q20 = start_eef.copy(), q20_start.copy()
        last_sent = None
        chunk = None; exec_start = 0.0; chunk_q16_start = q0.copy()
        settle_since = time.monotonic(); pending = None; pending_sample = None
        hold_waiting = False
        ready_plan = None; ready_start = None; ready_sample = None; gate = StepGate()
        planner_caps = dict(describe["caps"], **describe["accel_caps"])
        cycle = failures = plan_sequence = 0
        period = 1.0 / args.rate
        began = time.monotonic(); n = 0; last_tick = 0.0
        stop_reason = None
        print("HOLDING; first inference ...", flush=True)
        while stop_reason is None:
            now = time.monotonic()
            # ----- command for this tick -----
            if chunk is not None:
                elapsed = now - exec_start
                target_eef, target_q20 = chunk(elapsed)
                if elapsed >= chunk.duration:
                    if args.execute:
                        command_eef, command_q20 = chunk.eef[-1].copy(), chunk.q20[-1].copy()
                    else:  # shadow: the robot did not move, keep anchoring to measurement
                        command_eef = sdk_pose_xyzw(robot.pose)
                        command_q20 = np.array([hands.feedback()[s] for s in SIDES])
                    finish_execution_audit("completed", True)
                    chunk = None; camera.set_execution_active(False)
                    settle_since = now
                    cycle += 1
                    if args.cycles and cycle >= args.cycles:
                        stop_reason = "requested cycles completed"
            if chunk is None:
                target_eef, target_q20 = command_eef, command_q20
            if args.execute:
                check_servop_box(target_eef)
                if last_sent is not None:
                    step_p, step_r = pose_errors(target_eef, last_sent)
                    if max(step_p) > COMMAND_STEP_POS or max(step_r) > COMMAND_STEP_ROT:
                        raise RuntimeError("command step %.4f m / %.4f rad exceeds bound" % (max(step_p), max(step_r)))
                left, right = quat(target_eef)
                robot.send("request_servop", {"left_pos": left.tolist(), "right_pos": right.tolist()})
                robot.commands_sent += 1
                robot.lease_at = time.monotonic()
                last_sent = target_eef.copy()
                if execution_audit is not None:
                    execution_audit.record_send(now - exec_start, target_eef)

            # ----- 50 Hz supervision -----
            if now - last_tick >= 0.02:
                last_tick = now
                q, dq, _, _ = session.tick(None, chunk_q16_start, chunk_q16_start)
                measured_eef = sdk_pose_xyzw(robot.pose)
                if args.execute:
                    ep, er = pose_errors(measured_eef, target_eef)
                    result["max_eef_tracking_m"] = max(result.get("max_eef_tracking_m", 0.0), float(max(ep)))
                    result["max_eef_tracking_rad"] = max(result.get("max_eef_tracking_rad", 0.0), float(max(er)))
                    if max(ep) > EEF_TRACK_POS or max(er) > EEF_TRACK_ROT:
                        if execution_audit is not None:
                            # Preserve the exact violating arm sample without
                            # sending another hand target after the guard trips.
                            fault_q20 = np.array([hands.feedback()[s] for s in SIDES])
                            execution_audit.record_feedback(
                                now - exec_start, target_eef, target_q20,
                                measured_eef, fault_q20, q, dq,
                                arm_guard_failure=True)
                        raise RuntimeError("EEF tracking error %.4f m / %.4f rad" % (max(ep), max(er)))
                    measured_q20 = hand_session.tick(target_q20, "xflow_exec" if chunk is not None else "xflow_hold")
                else:
                    measured_q20 = np.array([hands.feedback()[s] for s in SIDES])
                if execution_audit is not None:
                    execution_audit.record_feedback(
                        now - exec_start, target_eef, target_q20, measured_eef,
                        measured_q20, q, dq)
                if camera.error:
                    raise RuntimeError("camera: " + str(camera.error))

                # Request the next chunk once the robot has settled in hold.
                # Hold-settle gate (operator decision 2026-11-15): the loaded arms sag
                # 1-2 cm below the ServoP command at rest and settle after each chunk,
                # so exceeding HOLD_SETTLE_POS/ROT first means "keep holding and wait";
                # it aborts after HOLD_SETTLE_TIMEOUT, or at once beyond the 4 cm
                # tracking guard checked above every tick.
                hold_ready = False
                prefetch_ready = (args.prefetch > 0.0 and args.execute and chunk is not None
                                  and pending is None and ready_plan is None
                                  and stop_reason is None
                                  and now - exec_start >= args.prefetch * chunk.duration)
                if (chunk is None and pending is None and ready_plan is None and stop_reason is None
                        and now - settle_since >= args.settle and np.max(abs(dq[:14])) < 0.05):
                    hold_ready = True
                    if args.execute:
                        drift_p, drift_r = pose_errors(measured_eef, command_eef)
                        if max(drift_p) > HOLD_SETTLE_POS or max(drift_r) > HOLD_SETTLE_ROT:
                            hold_ready = False
                            if now - settle_since > HOLD_SETTLE_TIMEOUT:
                                raise RuntimeError("held pose stayed %.3f m / %.3f rad from the command for %.1f s"
                                                   % (max(drift_p), max(drift_r), HOLD_SETTLE_TIMEOUT))
                            if not hold_waiting:
                                print("  hold settling: %.1f mm / %.1f deg from command, waiting"
                                      % (1000 * max(drift_p), math.degrees(max(drift_r))), flush=True)
                            hold_waiting = True
                if hold_ready or prefetch_ready:
                    hold_waiting = False
                    not_before = (exec_start if prefetch_ready
                                  else settle_since + args.settle * 0.5)
                    image = camera.sample(not_before, history_frames)
                    if image is not None:
                        if prefetch_ready:
                            # Sampled mid-motion: anchor to what was commanded at
                            # that instant; build_splice realigns at dispatch.
                            anchor_eef, anchor_q20 = chunk(image["sample_monotonic"] - exec_start)
                        else:
                            anchor_eef = command_eef.copy()
                            anchor_q20 = hand_session.last.copy() if args.execute else measured_q20
                        snapshot = dict(image, q16=q.copy(), eef_measured=measured_eef,
                                        q20_measured=measured_q20, eef_command=anchor_eef,
                                        q20_command=anchor_q20)
                        plan_id = plan_sequence
                        plan_sequence += 1
                        connection.send({k: v for k, v in snapshot.items()})
                        pending = time.monotonic()
                        pending_sample = dict(eef=measured_eef.copy(), q20=measured_q20.copy(),
                                              eef_command=command_eef.copy(), q20_command=np.array(anchor_q20),
                                              history=image["history_rgb"] is not None,
                                              history_reason=image["history_missing_reason"],
                                              plan_id=plan_id, snapshot=snapshot,
                                              prefetch=bool(prefetch_ready),
                                              sample_monotonic=image["sample_monotonic"],
                                              sample_chunk=chunk, sample_exec_start=exec_start)

                if pending is not None and connection.poll():
                    answer = connection.recv()
                    latency = time.monotonic() - pending
                    pending = None
                    plan_id = pending_sample["plan_id"]
                    artifact = audit_sink.submit_plan(
                        plan_id, pending_sample["snapshot"], answer)
                    row = dict(cycle=cycle, plan_id=plan_id, planning_s=latency,
                               plan_artifact=artifact, ok=answer["ok"],
                               history_sent=pending_sample["history"],
                               history_missing_reason=pending_sample["history_reason"])
                    if not answer["ok"]:
                        failures += 1
                        row["reason"] = answer["reason"]
                        print("PLAN REJECTED (%d/%d): %s" % (failures, args.max_failures, answer["reason"]), flush=True)
                        if failures >= args.max_failures:
                            stop_reason = "too many consecutive rejected plans"
                        settle_since = time.monotonic()
                    else:
                        failures = 0
                        plan = answer["plan"]
                        plan["_plan_id"] = plan_id
                        plan["_plan_artifact"] = artifact
                        rep = plan["report"]
                        row.update(rtt_ms=rep["rpc"]["round_trip_ms"], stage_ms=rep["stage_ms"],
                                   duration_s=rep["timing"]["duration_s"], stretch=rep["timing"]["stretch"],
                                   binding=rep["timing"]["binding"],
                                   ik_projection=rep.get("ik_projection"),
                                   anchor={s: dict(t_mm=1000 * rep["anchor"][s]["local_offset_translation_norm_m"],
                                                   r_deg=math.degrees(rep["anchor"][s]["local_offset_rotation_rad"]),
                                                   q_rad=rep["anchor"][s]["q_offset_abs_max_rad"],
                                                   decay=rep["anchor"][s]["decay_steps"]) for s in SIDES})
                        targets_path = out / ("plan%04d_targets.npz" % plan_id)
                        np.savez_compressed(targets_path, times_s=plan["times_s"],
                                            eef=plan["eef"], q20=plan["q20"], q16=plan["q16"],
                                            predicted_keypoints=plan["predicted_keypoints"],
                                            current_keypoints=plan["current_keypoints"])
                        row["targets_npz"] = str(targets_path)
                        if pending_sample["prefetch"]:
                            moved = False   # expected: the splice realigns at dispatch
                        else:
                            drift_p, drift_r = pose_errors(measured_eef, pending_sample["eef"])
                            moved = max(drift_p) > 0.003 or max(drift_r) > math.radians(1.5) or \
                                np.max(abs(measured_q20 - pending_sample["q20"])) > 0.08
                        print("PLAN %03d  rtt %.0f ms  plan %.2f s  chunk %.2f s (x%.2f %s)  anchor L %.1fmm/%.1fdeg R %.1fmm/%.1fdeg" % (
                            cycle, rep["rpc"]["round_trip_ms"], latency, rep["timing"]["duration_s"],
                            rep["timing"]["stretch"], rep["timing"]["binding"],
                            row["anchor"]["left"]["t_mm"], row["anchor"]["left"]["r_deg"],
                            row["anchor"]["right"]["t_mm"], row["anchor"]["right"]["r_deg"]), flush=True)
                        if moved:
                            row["discarded"] = "robot moved since the sample"
                            settle_since = time.monotonic()
                        else:
                            ready_plan = plan
                            ready_sample = pending_sample
                            ready_start = (pending_sample["eef_command"], pending_sample["q20_command"])
                            if args.step:
                                gate.ask("cycle %d ready" % cycle)
                    cycles_log.write(json.dumps(row, default=float) + "\n"); cycles_log.flush()

                if ready_plan is not None:
                    decision = gate.poll() if args.step else "go"
                    if decision == "quit":
                        stop_reason = "operator quit"
                    elif decision == "skip":
                        ready_plan = None; settle_since = time.monotonic()
                    elif decision == "go":
                        plan_to_execute = ready_plan
                        spliced = None
                        if ready_sample["prefetch"] and chunk is not None:
                            spliced = splice_plan(chunk, exec_start, ready_sample,
                                                  plan_to_execute, planner_caps)
                            if spliced is None:
                                print("  splice not possible; letting the chunk finish first",
                                      flush=True)
                        if spliced is not None:
                            finish_execution_audit("spliced", True)
                            cycle += 1
                            if args.cycles and cycle >= args.cycles:
                                stop_reason = "requested cycles completed"
                            plan_to_execute = spliced["plan"]
                            start_eef_q20 = spliced["start"]
                            row_splice = spliced["info"]
                            print("  SPLICE at %.2f s: skipped %d steps, %.2f s left%s"
                                  % (time.monotonic() - exec_start, row_splice["skipped_points"],
                                     float(plan_to_execute["times_s"][-1]),
                                     "" if row_splice["moving"] else " (from rest)"), flush=True)
                        elif ready_sample["prefetch"]:
                            ready_plan = None; ready_sample = None
                            continue       # wait for the running chunk, then re-sample
                        else:
                            start_eef_q20 = ready_start
                        # Start exactly where the plan was anchored (same commanded pose/q20).
                        chunk = ChunkSampler(start_eef_q20[0], start_eef_q20[1], plan_to_execute,
                                             start_model_t=plan_to_execute.get("_start_model_t", 0.0))
                        ready_plan = None; ready_sample = None
                        exec_start = time.monotonic(); chunk_q16_start = q.copy()
                        execution_audit = ExecutionAuditRecorder(
                            out, plan_to_execute["_plan_id"], plan_to_execute,
                            start_eef_q20[0], start_eef_q20[1], q.copy(), execute=args.execute,
                            arm_rate_hz=args.rate,
                            plan_artifact=plan_to_execute.get("_plan_artifact"),
                            eef_position_limit_m=EEF_TRACK_POS,
                            eef_rotation_limit_rad=EEF_TRACK_ROT)
                        camera.set_execution_active(True)
                        if not args.execute:
                            print("  (shadow) chunk not sent", flush=True)

            # ----- pacing -----
            n += 1
            deadline = began + n * period
            if time.monotonic() > deadline + period:
                n = int((time.monotonic() - began) / period)
                continue
            while time.monotonic() < deadline:
                time.sleep(min(0.0005, max(0.0, deadline - time.monotonic())))
        if chunk is not None and stop_reason == "requested cycles completed":
            remaining = chunk.duration - (time.monotonic() - exec_start)
            if remaining > 0:
                print("  finishing the last chunk (%.2f s)" % remaining, flush=True)
            while time.monotonic() - exec_start < chunk.duration:
                now = time.monotonic()
                target_eef, target_q20 = chunk(now - exec_start)
                if args.execute:
                    left, right = quat(target_eef)
                    robot.send("request_servop", {"left_pos": left.tolist(),
                                                  "right_pos": right.tolist()})
                    robot.commands_sent += 1
                    robot.lease_at = time.monotonic()
                if now - last_tick >= 0.02:
                    last_tick = now
                    session.tick(None, chunk_q16_start, chunk_q16_start)
                    if args.execute:
                        hand_session.tick(target_q20, "xflow_exec")
                time.sleep(1.0 / args.rate)
            finish_execution_audit("completed", True)
        result["stop_reason"] = stop_reason
        result["cycles_executed"] = cycle
        print("STOPPING: " + stop_reason, flush=True)
        if args.execute and robot.commands_sent:
            controller_hold(robot, result, stop_reason)
        result["completed"] = True
    except KeyboardInterrupt:
        result["stop_reason"] = "Ctrl-C"
        print("Ctrl-C: holding the arms", flush=True)
        if args.execute and robot is not None and robot.commands_sent and robot.armed:
            controller_hold(robot, result, "operator Ctrl-C")
        result["completed"] = True
    finally:
        errors = []
        if execution_audit is not None:
            try:
                finish_execution_audit(result.get("stop_reason", "fault_or_cleanup"), False)
            except Exception as exc:
                errors.append("deployment audit: " + str(exc))
        cycles_log.close()
        if connection is not None:
            try:
                connection.send(None)
            except Exception:
                pass
        if planner is not None:
            planner.join(timeout=3)
            if planner.is_alive():
                planner.terminate()
        if robot and robot.commands_sent and "controller_hold" not in result:
            # Fault path: stop streaming and let the controller hold; no request_emgy_stop.
            failure = sys.exc_info()[1]
            controller_hold(robot, result, ("fault: %s: %s" % (type(failure).__name__, failure))
                            if failure else "cleanup")
        if hands:
            if hands.touched:
                try:
                    hands.stop(); result["hands_stop"] = "confirmed"
                except Exception as exc:
                    errors.append("hand stop: " + str(exc))
            try:
                hands.close()
            except Exception as exc:
                errors.append("hand close: " + str(exc))
        if camera:
            camera.close()
        if session:
            session.feedback_log.close()
        if hand_session:
            hand_session.log.close()
        if stack:
            try:
                stop_hand_stack(stack)
            except Exception as exc:
                errors.append("stack stop: " + str(exc))
        if stack_log:
            stack_log.close()
        if robot:
            result["arm_motion_commands_sent"] = robot.commands_sent
            result["emergency_stops_suppressed"] = robot.suppressed_emergency_stops
            try:
                robot.close()
            except Exception as exc:
                errors.append("robot close: " + str(exc))
            finally:
                save_trace(robot, out, result)
        if ros:
            rclpy.shutdown()
        audit_output = audit_sink.close()
        result["audit"]["plans"] = audit_output["plans"]
        result["audit"]["executions"] = audit_output["executions"]
        result["audit"]["plan_count"] = len(audit_output["plans"])
        result["audit"]["execution_count"] = len(audit_output["executions"])
        if audit_output["errors"]:
            result["audit"]["errors"] = audit_output["errors"]
            errors.append("deployment audit writer failed")
        if errors:
            result["cleanup_errors"] = errors
            raise RuntimeError("; ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="flower | whiteboard | tidying | lemon | chili")
    parser.add_argument("--model", default=None,
                        help="model name on the service; default: the service's default_model")
    parser.add_argument("--speed-ratio", type=float, default=None,
                        help="training-motion speed / robot speed; default from the server "
                             "(hh 4, rb 1). Also sets the history offset.")
    parser.add_argument("--execute", action="store_true", help="move the robot; default is a live shadow run")
    parser.add_argument("--step", action="store_true", help="ask for Enter before every chunk")
    parser.add_argument("--cycles", type=int, default=0, help="0 = until Ctrl-C")
    parser.add_argument("--steps", type=int, default=32, help="model steps executed per chunk (2..32)")
    parser.add_argument("--image-mode", choices=("virtual_cam0", "stretch_undistort"), default="virtual_cam0")
    parser.add_argument("--virtual-preset", choices=("cam0_newk", "hh_435", "hh_335"), default="cam0_newk")
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--arm-cap", type=float, default=0.8, help="planned arm joint speed cap rad/s")
    parser.add_argument("--hand-cap", type=float, default=0.9, help="planned hand joint speed cap rad/s")
    parser.add_argument("--linear-cap", type=float, default=0.25, help="planned EEF speed cap m/s")
    parser.add_argument("--angular-cap", type=float, default=1.0, help="planned EEF angular cap rad/s")
    parser.add_argument("--settle", type=float, default=0.2, help="hold time before sampling, s")
    parser.add_argument("--prefetch", type=float, default=0.0,
                        help="0 = stop-and-go (default); 0<F<1 samples the next chunk after "
                             "F of the running one and splices the result in")
    parser.add_argument("--shaping", choices=("off", "on"), default=None,
                        help="ease-in/out + acceleration caps; default from deployment_config.json")
    parser.add_argument("--rate", type=float, default=300.0)
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--relay-port", type=int, default=28765)
    args = parser.parse_args()
    if args.speed_ratio is not None and not 0.25 <= args.speed_ratio <= 8.0:
        parser.error("--speed-ratio must be within [0.25, 8] (history must fit the 6 s camera buffer)")
    if not 0.0 <= args.prefetch < 0.95:
        parser.error("--prefetch must be within [0, 0.95)")
    if not 50 <= args.rate <= 500 or not 2 <= args.steps <= 32 or args.cycles < 0:
        parser.error("invalid rate/steps/cycles")
    if not (0 < args.arm_cap <= 1.0 and 0 < args.hand_cap <= 1.0 and 0 < args.linear_cap <= 0.5
            and 0 < args.angular_cap <= 2.0):
        parser.error("speed caps exceed the replay limits")
    out = CLIENT / "runs" / ("live-%s-%s-%d" % ("exec" if args.execute else "shadow",
                                                time.strftime("%Y%m%d-%H%M%S"), os.getpid()))
    out.mkdir(parents=True)
    result = {"completed": False, "execute": args.execute, "args": vars(args), "arm_motion_commands_sent": 0,
              "hand_target_updates": 0}
    print("RUN DIR", out, flush=True)

    def interrupt(signum, frame):
        raise KeyboardInterrupt()
    old = signal.signal(signal.SIGTERM, interrupt)
    lock = open(LOCK, "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, out, result)
    except BaseException as exc:
        result["error"] = type(exc).__name__ + ": " + str(exc)
        print("XFLOW LIVE FAILED:", result["error"], flush=True)
    finally:
        lock.close()
        signal.signal(signal.SIGTERM, old)
        try:
            result["audit_index"] = build_run_summary(out)
        except BaseException as exc:
            result["audit_index_error"] = type(exc).__name__ + ": " + str(exc)
        save(out / "result.json", result)
        print("REPORT", out / "result.json", flush=True)
    return 0 if result.get("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
