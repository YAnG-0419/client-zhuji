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

SIDES = ("left", "right")
LOCK = "/home/guest/wanchi/replay_tip_align_test_1030/.hands_replay.lock"
CAMERA_ARCHIVE = ROOT / "calibration/camera/color_camera_info_live.json"
EEF_TRACK_POS, EEF_TRACK_ROT = 0.04, 0.20      # replay_hand execution_cross limits
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
    """Interpolates the commanded start pose and the planned steps on one clock."""

    def __init__(self, start_eef, start_q20, plan):
        self.times = np.r_[0.0, np.asarray(plan["times_s"], float)]
        self.eef = np.concatenate([np.asarray(start_eef, float)[None], np.asarray(plan["eef"], float)])
        self.q20 = np.concatenate([np.asarray(start_q20, float)[None], np.asarray(plan["q20"], float)])
        self.duration = float(self.times[-1])

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
        return {"sample_ns": int(current[0]), "head_rgb": current[2].copy(),
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
                  steps=args.steps)
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
    ros = False
    cycles_log = (out / "cycles.jsonl").open("w")
    try:
        commands = process_commands()
        if any(owner in c for c in commands for owner in OWNERS):
            raise RuntimeError("another robot controller is running; exit teleop/home/replay first")
        print("Starting planner (XFlow %s, task %s) ..." % (args.model, args.task), flush=True)
        planner, connection, describe = start_planner(args)
        result["planner"] = describe
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
        command_eef, command_q20 = start_eef.copy(), q20_start.copy()
        last_sent = None
        chunk = None; exec_start = 0.0; chunk_q16_start = q0.copy()
        settle_since = time.monotonic(); pending = None; pending_sample = None
        ready_plan = None; ready_start = None; gate = StepGate()
        cycle = failures = 0
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
                        raise RuntimeError("EEF tracking error %.4f m / %.4f rad" % (max(ep), max(er)))
                    measured_q20 = hand_session.tick(target_q20, "xflow_exec" if chunk is not None else "xflow_hold")
                else:
                    measured_q20 = np.array([hands.feedback()[s] for s in SIDES])
                if camera.error:
                    raise RuntimeError("camera: " + str(camera.error))

                # Request the next chunk once the robot has settled in hold.
                if (chunk is None and pending is None and ready_plan is None and stop_reason is None
                        and now - settle_since >= args.settle and np.max(abs(dq[:14])) < 0.05):
                    image = camera.sample(settle_since + args.settle * 0.5, 16)
                    if image is not None:
                        anchor_q20 = hand_session.last.copy() if args.execute else measured_q20
                        snapshot = dict(image, q16=q.copy(), eef_measured=measured_eef,
                                        q20_measured=measured_q20, eef_command=command_eef.copy(),
                                        q20_command=anchor_q20)
                        drift_p, drift_r = pose_errors(measured_eef, command_eef)
                        if args.execute and (max(drift_p) > 0.02 or max(drift_r) > math.radians(3)):
                            raise RuntimeError("held pose drifted %.3f m from the command" % max(drift_p))
                        connection.send({k: v for k, v in snapshot.items()})
                        pending = time.monotonic()
                        pending_sample = dict(eef=measured_eef.copy(), q20=measured_q20.copy(),
                                              eef_command=command_eef.copy(), q20_command=np.array(anchor_q20),
                                              history=image["history_rgb"] is not None,
                                              history_reason=image["history_missing_reason"])

                if pending is not None and connection.poll():
                    answer = connection.recv()
                    latency = time.monotonic() - pending
                    pending = None
                    row = dict(cycle=cycle, planning_s=latency, ok=answer["ok"],
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
                        rep = plan["report"]
                        row.update(rtt_ms=rep["rpc"]["round_trip_ms"], stage_ms=rep["stage_ms"],
                                   duration_s=rep["timing"]["duration_s"], stretch=rep["timing"]["stretch"],
                                   binding=rep["timing"]["binding"],
                                   anchor={s: dict(t_mm=1000 * rep["anchor"][s]["local_offset_translation_norm_m"],
                                                   r_deg=math.degrees(rep["anchor"][s]["local_offset_rotation_rad"]),
                                                   q_rad=rep["anchor"][s]["q_offset_abs_max_rad"],
                                                   decay=rep["anchor"][s]["decay_steps"]) for s in SIDES})
                        np.savez_compressed(out / ("cycle%03d.npz" % cycle), times_s=plan["times_s"],
                                            eef=plan["eef"], q20=plan["q20"], q16=plan["q16"],
                                            predicted_keypoints=plan["predicted_keypoints"],
                                            current_keypoints=plan["current_keypoints"])
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
                        # Start exactly where the plan was anchored (same commanded pose/q20).
                        chunk = ChunkSampler(ready_start[0], ready_start[1], ready_plan)
                        ready_plan = None
                        exec_start = time.monotonic(); chunk_q16_start = q.copy()
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
        if errors:
            result["cleanup_errors"] = errors
            raise RuntimeError("; ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, help="flower | whiteboard | tidying | lemon | chili")
    parser.add_argument("--model", default="hh_a12")
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
    parser.add_argument("--rate", type=float, default=300.0)
    parser.add_argument("--max-failures", type=int, default=3)
    parser.add_argument("--relay-port", type=int, default=28765)
    args = parser.parse_args()
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
        save(out / "result.json", result)
        print("REPORT", out / "result.json", flush=True)
    return 0 if result.get("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
