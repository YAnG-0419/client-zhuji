"""Pure timestamp, latency, and history-clock policy for XFlow chunks.

No ROS, socket, or robot API is imported here. The deployment default is a
``sync_hold`` cycle: sample, hold while inference runs, then schedule all 32
returned steps at 30 Hz from ``dispatch_ns + reserve``. Native model time is
kept only for a diagnostic realtime-cutoff comparison.

For ``frame_per_step_30hz`` the history image is selected on the *execution*
clock 16 execution periods (0.533 s) before the head image. Time spent in a
synchronous hold does not advance that clock.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple


FRAME_PER_STEP_30HZ = "frame_per_step_30hz"
NATIVE_TIME = "native_time"
SYNC_HOLD = "sync_hold"
REALTIME_CUTOFF = "realtime_cutoff"
EXECUTION_CLOCK = "execution"
WALL_CLOCK = "wall"
DEFAULT_MODE = FRAME_PER_STEP_30HZ
DEFAULT_LATENCY_POLICY = SYNC_HOLD
DEFAULT_STEPS = 32
EXECUTION_FPS = 30.0
EXECUTION_HISTORY_FRAMES = 16


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(name + " must be a number")
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


@dataclass(frozen=True)
class ModelTiming:
    contract: str
    fps: float
    history_offset_frames: int
    keypoint_time_s: Tuple[float, ...]

    @classmethod
    def from_info(cls, model_info: Dict[str, Any]) -> "ModelTiming":
        contract = str(model_info.get("contract", ""))
        fps = _finite_float(model_info.get("fps", 30.0), "model fps")
        history = model_info.get("history_offset_frames")
        try:
            times = tuple(float(value) for value in model_info.get("keypoint_time_s"))
        except (TypeError, ValueError):
            raise ValueError("keypoint_time_s must be a numeric sequence")
        if not contract:
            raise ValueError("model contract is required")
        if fps <= 0:
            raise ValueError("model fps must be positive")
        if type(history) is not int or history < 0:
            raise ValueError("history_offset_frames must be a nonnegative integer")
        if (len(times) != 32 or not all(math.isfinite(value) for value in times)
                or times[0] <= 0
                or any(right <= left for left, right in zip(times, times[1:]))):
            raise ValueError("keypoint_time_s must contain 32 increasing positive values")
        return cls(contract=contract, fps=fps, history_offset_frames=history,
                   keypoint_time_s=times)

    def history_report(self, history_clock=EXECUTION_CLOCK) -> dict:
        if history_clock not in (EXECUTION_CLOCK, WALL_CLOCK):
            raise ValueError("history_clock must be execution or wall")
        if history_clock == EXECUTION_CLOCK:
            frames = EXECUTION_HISTORY_FRAMES
            seconds = frames / EXECUTION_FPS
            policy = "16 completed execution periods; synchronous holds are skipped"
        else:
            frames = self.history_offset_frames
            seconds = frames / self.fps
            policy = "model-advertised wall-camera offset"
        return {
            "clock": history_clock,
            "capture_offset_frames": frames,
            "capture_offset_s": seconds,
            "capture_policy": policy,
            "model_advertised_history_offset_frames": self.history_offset_frames,
            "model_advertised_history_offset_s": self.history_offset_frames / self.fps,
        }


@dataclass(frozen=True)
class ExecutionSchedule:
    mode: str
    sample_ns: int
    exec_start_ns: int
    source_indices: Tuple[int, ...]
    deadlines_ns: Tuple[int, ...]
    model_time_s: Tuple[float, ...]
    report: dict

    def __post_init__(self) -> None:
        indices = tuple(int(value) for value in self.source_indices)
        deadlines = tuple(int(value) for value in self.deadlines_ns)
        model_times = tuple(float(value) for value in self.model_time_s)
        if (len(indices) < 1 or len(deadlines) != len(indices)
                or len(model_times) != len(indices)
                or any(right <= left for left, right in zip(indices, indices[1:]))
                or any(right <= left for left, right in zip(deadlines, deadlines[1:]))
                or not all(math.isfinite(value) for value in model_times)):
            raise ValueError("invalid execution schedule")
        if indices[0] < 0 or indices[-1] >= 32:
            raise ValueError("source index outside the 32-step horizon")
        object.__setattr__(self, "source_indices", indices)
        object.__setattr__(self, "deadlines_ns", deadlines)
        object.__setattr__(self, "model_time_s", model_times)

    @property
    def duration_s(self) -> float:
        anchor = self.exec_start_ns if self.mode == FRAME_PER_STEP_30HZ else self.sample_ns
        return (int(self.deadlines_ns[-1]) - int(anchor)) / 1e9

    def future(self, now_ns: int, reserve_s: float = 0.0) -> "ExecutionSchedule":
        reserve_s = _finite_float(reserve_s, "reserve_s")
        if reserve_s < 0:
            raise ValueError("reserve_s must be nonnegative")
        threshold = int(now_ns) + int(round(reserve_s * 1e9))
        keep = tuple(index for index, deadline in enumerate(self.deadlines_ns)
                     if deadline >= threshold)
        if not keep:
            raise TimeoutError("prediction has no future step after the planning reserve")
        report = dict(self.report)
        report["future_selection"] = {
            "now_ns": int(now_ns), "reserve_s": reserve_s,
            "discarded_steps": len(self.deadlines_ns) - len(keep),
            "kept_steps": len(keep),
        }
        return ExecutionSchedule(self.mode, self.sample_ns, self.exec_start_ns,
                                 tuple(self.source_indices[index] for index in keep),
                                 tuple(self.deadlines_ns[index] for index in keep),
                                 tuple(self.model_time_s[index] for index in keep),
                                 report)


def build_schedule(model: ModelTiming, *, sample_ns: int,
                   exec_start_ns: Optional[int] = None,
                   dispatch_ns: Optional[int] = None,
                   reserve_s: float = 0.0,
                   mode: str = DEFAULT_MODE,
                   latency_policy: str = DEFAULT_LATENCY_POLICY,
                   history_clock: str = EXECUTION_CLOCK,
                   steps: int = DEFAULT_STEPS) -> ExecutionSchedule:
    """Build a non-executing schedule and fail on ambiguous latency policy."""
    if type(sample_ns) is not int or sample_ns < 0:
        raise ValueError("sample_ns must be a nonnegative integer")
    if type(steps) is not int or not 1 <= steps <= 32:
        raise ValueError("steps must be an integer within [1, 32]")
    if latency_policy not in (SYNC_HOLD, REALTIME_CUTOFF):
        raise ValueError("unknown latency policy: " + str(latency_policy))
    if history_clock not in (EXECUTION_CLOCK, WALL_CLOCK):
        raise ValueError("history_clock must be execution or wall")
    reserve_s = _finite_float(reserve_s, "reserve_s")
    if reserve_s < 0:
        raise ValueError("reserve_s must be nonnegative")
    indices = tuple(range(steps))
    model_times = model.keypoint_time_s[:steps]

    if mode == FRAME_PER_STEP_30HZ:
        if latency_policy != SYNC_HOLD:
            raise ValueError("frame_per_step_30hz only supports sync_hold")
        if dispatch_ns is not None:
            if type(dispatch_ns) is not int or dispatch_ns < sample_ns:
                raise ValueError("dispatch_ns must be an integer >= sample_ns")
            derived = int(dispatch_ns) + int(round(reserve_s * 1e9))
            if exec_start_ns is not None and exec_start_ns != derived:
                raise ValueError("exec_start_ns conflicts with dispatch_ns + reserve")
            exec_start_ns = derived
        if exec_start_ns is None or type(exec_start_ns) is not int or exec_start_ns < sample_ns:
            raise ValueError("sync_hold requires exec_start_ns or dispatch_ns >= sample_ns")
        offsets_ns = tuple(int(round((index + 1) * (1e9 / EXECUTION_FPS)))
                           for index in indices)
        deadlines = tuple(int(exec_start_ns) + offset for offset in offsets_ns)
        anchor = int(exec_start_ns)
        timing_definition = "deadline[i] = dispatch_ns + reserve + round((i+1)/30 * 1e9)"
    elif mode == NATIVE_TIME:
        if latency_policy != REALTIME_CUTOFF:
            raise ValueError("native_time is diagnostic and requires realtime_cutoff")
        if exec_start_ns is not None and exec_start_ns != sample_ns:
            raise ValueError("native_time is anchored at sample_ns; do not retime it")
        offsets_ns = tuple(int(round(model_time * 1e9)) for model_time in model_times)
        deadlines = tuple(int(sample_ns) + offset for offset in offsets_ns)
        anchor = int(sample_ns)
        exec_start_ns = int(sample_ns)
        timing_definition = "diagnostic: deadline[i] = sample_ns + keypoint_time_s[i]"
    else:
        raise ValueError("unknown timing mode: " + str(mode))

    report = {
        "mode": mode, "latency_policy": latency_policy,
        "default_mode": DEFAULT_MODE, "default_latency_policy": DEFAULT_LATENCY_POLICY,
        "steps": steps, "default_steps": DEFAULT_STEPS,
        "execution_fps": EXECUTION_FPS,
        "execution_period_s": 1.0 / EXECUTION_FPS,
        "timing_definition": timing_definition,
        "sample_ns": int(sample_ns),
        "dispatch_ns": None if dispatch_ns is None else int(dispatch_ns),
        "reserve_s": reserve_s,
        "anchor_ns": anchor,
        "first_deadline_ns": int(deadlines[0]),
        "last_deadline_ns": int(deadlines[-1]),
        "duration_s": (int(deadlines[-1]) - anchor) / 1e9,
        "native_model_time_s": list(model_times),
        "native_time_is_diagnostic_only": True,
        "selected_mode_is_execution_default": mode == DEFAULT_MODE,
        "history": model.history_report(history_clock),
        "hardware_commands_sent": 0,
        "execution_ready": False,
    }
    return ExecutionSchedule(mode, int(sample_ns), int(exec_start_ns), indices,
                             deadlines, model_times, report)
