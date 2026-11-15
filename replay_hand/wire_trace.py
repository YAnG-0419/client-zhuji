"""Bounded in-memory robot WebSocket trace; disk writes only after execution.

Capture exact serialized payloads and raw replies, including the feedback frame
which a guard rejects before the ordinary telemetry logger can record it.
"""
from collections import deque
import json
import threading
import time


class WireTrace:
    def __init__(self, socket, capacity=8192):
        self.socket = socket
        self.rows = deque(maxlen=capacity)
        self.lock = threading.Lock()
        self.total = 0

    def __getattr__(self, key):
        return getattr(self.socket, key)

    def record(self, direction, started, payload, error=None):
        row = dict(direction=direction, monotonic_start_s=started,
                   monotonic_end_s=time.monotonic(), unix_s=time.time(),
                   payload=payload if isinstance(payload, str) else repr(payload))
        if error is not None:
            row['error'] = type(error).__name__+': '+str(error)
        with self.lock:
            row['sequence'] = self.total
            self.total += 1
            self.rows.append(row)

    def send(self, payload, *args, **kwargs):
        began = time.monotonic()
        try:
            answer = self.socket.send(payload, *args, **kwargs)
        except BaseException as exc:
            self.record('send', began, payload, exc)
            raise
        self.record('send', began, payload)
        return answer

    def recv(self, *args, **kwargs):
        began = time.monotonic()
        # Timeouts are frequent and not a robot reply; preserve their original
        # exception and let the existing transport handle them unchanged.
        payload = self.socket.recv(*args, **kwargs)
        self.record('recv', began, payload)
        return payload

    def save(self, path):
        with self.lock:
            rows, total = list(self.rows), self.total
        with path.open('w') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False)+'\n')
        return dict(path=path.name, events=total, retained=len(rows),
                    dropped=max(0, total-len(rows)), policy='bounded raw send/receive tail; no disk I/O in send loop')


def attach_trace(robot):
    robot.wire_trace = WireTrace(robot.ws)
    robot.ws = robot.wire_trace


def save_trace(robot, out, result):
    trace = getattr(robot, 'wire_trace', None)
    if trace is not None:
        try:
            result['wire_trace'] = trace.save(out/'robot_wire.jsonl')
        except Exception as exc:
            result['wire_trace_error'] = type(exc).__name__+': '+str(exc)
