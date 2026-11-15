"""Replay transport limited to targets the TRON2 firmware actually executes.

The URDF mechanical ranges are wider than the SDK MoveJ envelope. A target
outside that envelope is acknowledged with result=success and then silently
not executed, which previously surfaced only as an approach settle timeout
(runs/20261104_145739_102574 and runs/20261104_150124_104108). Planning and
sending therefore use the intersection of both ranges.
"""
import json
import time
import uuid

import numpy as np
from replay_thresholds import MODEL_CLEARANCE_M, INTERNAL_WRIST_CLEARANCE_M


def validate_movej(data, lower, upper):
    q = np.asarray(data['joint'], dtype=float)
    duration = float(data['time'])
    if q.shape != (14,) or not np.isfinite(q).all():
        raise ValueError('MoveJ requires 14 finite arm joints')
    if np.any(q < lower+1e-5) or np.any(q > upper-1e-5):
        raise ValueError('MoveJ target exceeds the executable joint range')
    if not np.isfinite(duration) or duration < .25:
        raise ValueError('MoveJ duration must be finite and at least 0.25 seconds')
    return q, duration


def replay_backend():
    from robot_home import backend
    from home_policy import require_status
    _, BaseModel, BaseTransport = backend()

    class Model(BaseModel):
        def __init__(self):
            super().__init__()
            # URDF mechanical ranges intersected with the SDK MoveJ envelope:
            # a target the firmware will not execute must fail during offline
            # planning, not as a silent no-motion timeout on hardware.
            from movej_limits import LOWER, UPPER
            self.lower, self.upper = self.model.jnt_range[self.jids].T.copy()
            self.lower[:14] = np.maximum(self.lower[:14], LOWER)
            self.upper[:14] = np.minimum(self.upper[:14], UPPER)
            self.internal_wrist_pairs = {
                frozenset(('wrist_yaw_'+side+'_Link', 'wrist_roll_'+side+'_Link'))
                for side in ('L', 'R')}

        def check_path(self, qs, clearance=MODEL_CLEARANCE_M, include_initial=True):
            return super().check_path(qs, clearance=max(clearance, MODEL_CLEARANCE_M),
                                      include_initial=include_initial)

        def check_interval(self, start, end, clearance=MODEL_CLEARANCE_M):
            return super().check_interval(start, end, clearance=max(clearance, MODEL_CLEARANCE_M))

        def clearance(self, sweep_delta=None):
            import mujoco
            delta = np.zeros(16) if sweep_delta is None else np.abs(sweep_delta)
            best, closest = float('inf'), None
            for g, h in self.pairs:
                names = [self.body_name(self.model.geom_bodyid[x]) for x in (g, h)]
                allowance = float((self.radius[g]+self.radius[h]) @ delta)/2
                distance = float(mujoco.mj_geomDistance(self.model, self.data, g, h, .5+allowance, None))-allowance
                required = (INTERNAL_WRIST_CLEARANCE_M
                            if frozenset(names) in self.internal_wrist_pairs else MODEL_CLEARANCE_M)
                # Normalize each pair's surplus to the common 5 mm gate.
                # The constant offset preserves swept and start-loss bounds.
                score = distance + MODEL_CLEARANCE_M - required
                if score < best:
                    best, closest = score, names
            return best, closest

    class Transport(BaseTransport):
        def __init__(self, joint_lower, joint_upper, *args, **kwargs):
            self.joint_lower = np.asarray(joint_lower, dtype=float).copy()
            self.joint_upper = np.asarray(joint_upper, dtype=float).copy()
            if self.joint_lower.shape != (14,) or self.joint_upper.shape != (14,):
                raise ValueError('Expected mechanical ranges for 14 arm joints')
            super().__init__(*args, **kwargs)

        def send(self, title, data, guid=None):
            if title != 'request_movej':
                return super().send(title, data, guid)
            q, duration = validate_movej(data, self.joint_lower, self.joint_upper)
            self.healthy()
            require_status(self.info)
            with self.write_lock:
                if not self.armed or self.backend != 'supervised_movej' or self.fault or self.closed.is_set():
                    raise RuntimeError('MoveJ transport is not healthy and armed')
                packet = {'accid': self.info['accid'], 'title': title,
                          'timestamp': int(time.time()*1000), 'guid': guid or uuid.uuid4().hex,
                          'data': {'joint': q.tolist(), 'time': duration}}
                self.ws.send(json.dumps(packet, allow_nan=False))

    return np, Model, Transport
