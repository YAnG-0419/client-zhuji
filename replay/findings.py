"""Downgrade software checks to recorded findings so a demonstration completes.

Replay normally refuses to proceed whenever an accuracy, tracking or state
check fails. For a demonstration the operator wants the opposite default: keep
the robot moving and write down what was out of tolerance. `Findings` is that
record, and `permit()` is the single predicate every downgraded check calls.

Two things are not software policy and stay enforced:

* The TRON2 firmware does not execute a MoveJ outside its joint envelope. It
  acknowledges the command with result=success and then holds position, which
  is why a target beyond the envelope is clipped to the boundary rather than
  sent as-is (`clip_to_envelope`).
* `stop.request` and Ctrl-C are the operator's stop and always abort.
"""
import json
import time


class Findings:
    """Append-only log of checks that were downgraded rather than enforced.

    Each entry is one JSON object per line in `violations.jsonl`, so a run can
    be inspected while it is still going. Terminal output is deliberately
    sparse: the first few of each kind, then a periodic count, because a
    4216-frame episode can produce thousands of entries of the same kind.
    """

    ECHO_LIMIT = 3
    ECHO_EVERY = 500

    def __init__(self, path):
        self._file = path.open('w')
        self.counts = {}
        self.extremes = {}

    def record(self, kind, detail, value=None, **fields):
        count = self.counts[kind] = self.counts.get(kind, 0)+1
        if value is not None:
            value = float(value)
            self.extremes[kind] = max(self.extremes.get(kind, value), value)
        entry = {'kind': kind, 'detail': detail, 'wall_s': time.time()}
        if value is not None:
            entry['value'] = value
        entry.update(fields)
        self._file.write(json.dumps(entry, allow_nan=False, default=float)+'\n')
        self._file.flush()
        if count <= self.ECHO_LIMIT:
            print('FINDING %s: %s' % (kind, detail), flush=True)
        elif count % self.ECHO_EVERY == 0:
            print('FINDING %s x%d' % (kind, count), flush=True)
        return entry

    def summary(self):
        return {'counts': dict(self.counts), 'worst_value': dict(self.extremes),
                'detail': 'One line per finding in violations.jsonl',
                'note': 'Checks downgraded by --permissive; the run continued past each one'}

    def close(self):
        self._file.close()


_active = None


def begin(path):
    """Start recording findings; returns the Findings so the caller can close it."""
    global _active
    _active = Findings(path)
    return _active


def end():
    global _active
    if _active is not None:
        _active.close()
    _active = None


def active():
    return _active


def permit(kind, detail, value=None, **fields):
    """Record a downgraded check and return True, or return False to enforce it.

    Call sites read as `if not permit(...): raise ...`, so the strict behaviour
    is what happens without --permissive and the text of each check stays next
    to the condition it describes.
    """
    if _active is None:
        return False
    _active.record(kind, detail, value, **fields)
    return True


def clip_to_envelope(lower, upper, q, where, names=None):
    """Clip a MoveJ target into the executable envelope, recording any excess.

    Verified on hardware 2026-11-04 (`runs/wrist_envelope_probe_20261104_180929.json`):
    wrist_roll_R was commanded to 0.70 rad (inside the envelope) and moved
    0.6959 rad, then to 0.78 rad (outside it) and moved 0.00000 rad, both
    acknowledged `success`. Reaching the boundary is therefore the most the
    hardware can do, and it is preferred over stalling at an unreachable one.
    """
    import numpy as np
    target = np.clip(q, lower, upper)
    excess = np.abs(target-q)
    worst = float(excess.max()) if excess.size else 0.
    if worst > 1e-9:
        outside = [int(j) for j in np.flatnonzero(excess > 1e-9)]
        labels = [names[j] for j in outside] if names else outside
        permit('movej_target_clipped',
               '%s: %s beyond the executable envelope by up to %.5f rad; moving to the limit'
               % (where, labels, worst),
               value=worst, joints=labels)
    return target
