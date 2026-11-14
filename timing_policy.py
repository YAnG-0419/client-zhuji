"""Pure timestamp policy for XFlow keypoint chunks.

This module has no ROS, socket, or robot imports.  It deliberately separates
the model's native prediction timestamps from the client execution policy:

``frame_per_step_30hz`` (default)
    Execute every returned step for one 30 Hz period.  For a 32-step chunk,
    deadlines are ``exec_start + (1..32) / 30`` seconds.

``native_time`` (diagnostic only)
    Preserve ``keypoint_time_s`` from the service info and anchor it at the
    observation sample timestamp.  It is retained for comparisons, not as the
    default deployment schedule.

The model history image contract remains unchanged.  A humanhand model still
receives the frame four camera periods before the current image.  Under the
frame-per-step execution interpretation that span is *equivalent* to sixteen
30 Hz execution periods; this 16-period/0.533 s value is reported
only as an execution-timescale diagnostic and must not be used to select the
history image.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple


FRAME_PER_STEP_30HZ = "frame_per_step_30hz"
NATIVE_TIME = "native_time"
DEFAULT_MODE = FRAME_PER_STEP_30HZ
DEFAULT_STEPS = 32
EXECUTION_FPS = 30.0


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
        fps = _finite_float(model_info.get("fps"), "model fps")
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

    @property
    def history_capture_offset_s(self) -> float:
        """Actual history-image offset required by the inference contract."""
        return self.history_offset_frames / self.fps

    def history_report(self) -> dict:
        execution_period = 1.0 / EXECUTION_FPS
        native_first_step = float(self.keypoint_time_s[0])
        execution_to_native_increment_ratio = execution_period / native_first_step
        equivalent_span_s = (
            self.history_capture_offset_s * execution_to_native_increment_ratio
        )
        equivalent_periods = equivalent_span_s / execution_period
        return {
            "capture_offset_frames": self.history_offset_frames,
            "capture_offset_s": self.history_capture_offset_s,
            "capture_policy": "read from info; unchanged by execution schedule",
            "frame_per_step_equivalent_execution_periods": equivalent_periods,
            "frame_per_step_equivalent_span_s": equivalent_span_s,
            "equivalent_value_is_diagnostic_only": True,
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
                   mode: str = DEFAULT_MODE,
                   steps: int = DEFAULT_STEPS) -> ExecutionSchedule:
    if type(sample_ns) is not int or sample_ns < 0:
        raise ValueError("sample_ns must be a nonnegative integer")
    if type(steps) is not int or not 1 <= steps <= 32:
        raise ValueError("steps must be an integer within [1, 32]")
    indices = tuple(range(steps))
    model_times = model.keypoint_time_s[:steps]

    if mode == FRAME_PER_STEP_30HZ:
        if exec_start_ns is None or type(exec_start_ns) is not int or exec_start_ns < sample_ns:
            raise ValueError("frame_per_step_30hz requires exec_start_ns >= sample_ns")
        # Round absolute offsets, rather than accumulating a truncated 33.333 ms
        # period; the 32nd deadline is therefore exactly nearest to 32/30 s.
        offsets_ns = tuple(int(round((index + 1) * (1e9 / EXECUTION_FPS)))
                           for index in indices)
        deadlines = tuple(int(exec_start_ns) + offset for offset in offsets_ns)
        anchor = int(exec_start_ns)
        timing_definition = "deadline[i] = exec_start_ns + round((i+1)/30 * 1e9)"
    elif mode == NATIVE_TIME:
        if exec_start_ns is not None and exec_start_ns != sample_ns:
            raise ValueError("native_time is anchored at sample_ns; do not retime it")
        offsets_ns = tuple(int(round(model_time * 1e9))
                           for model_time in model_times)
        deadlines = tuple(int(sample_ns) + offset for offset in offsets_ns)
        anchor = int(sample_ns)
        exec_start_ns = int(sample_ns)
        timing_definition = "diagnostic: deadline[i] = sample_ns + keypoint_time_s[i]"
    else:
        raise ValueError("unknown timing mode: " + str(mode))

    report = {
        "mode": mode,
        "default_mode": DEFAULT_MODE,
        "steps": steps,
        "default_steps": DEFAULT_STEPS,
        "execution_fps": EXECUTION_FPS,
        "execution_period_s": 1.0 / EXECUTION_FPS,
        "timing_definition": timing_definition,
        "anchor_ns": anchor,
        "first_deadline_ns": int(deadlines[0]),
        "last_deadline_ns": int(deadlines[-1]),
        "duration_s": (int(deadlines[-1]) - anchor) / 1e9,
        "native_model_time_s": list(model_times),
        "native_time_is_diagnostic_only": True,
        "selected_mode_is_execution_default": mode == DEFAULT_MODE,
        "history": model.history_report(),
        "hardware_commands_sent": 0,
        "execution_ready": False,
    }
    return ExecutionSchedule(mode, int(sample_ns), int(exec_start_ns), indices,
                             deadlines, model_times, report)

