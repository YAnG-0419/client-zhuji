"""Pinocchio-only IK adapter; MuJoCo remains an independent collision/FK check."""
import hashlib
from pathlib import Path
import sys

import numpy as np
from replay_thresholds import IK_POSITION_TOL, IK_ROTATION_TOL
from scipy.spatial.transform import Rotation


class PinocchioSolver:
    def __init__(self, root):
        sys.path.insert(0, str(root / 'Pinocchio'))
        import pin_model
        self.pin = pin_model.PinArms()
        self.calls, self.max_iterations = 0, 0
        self.max_position_error_m = self.max_rotation_error_rad = 0.
        self.provenance = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (Path(pin_model.__file__), root/'record/config/model/robot.urdf')}

    def solve(self, collision_model, seed, target):
        # Use the intersection: the command/collision model and Pinocchio must
        # both accept every solved joint. Never fall back to MuJoCo IK.
        self.pin.lower = np.maximum(self.pin.lower, collision_model.lower)
        self.pin.upper = np.minimum(self.pin.upper, collision_model.upper)
        q, iterations = self.pin.solve(np.asarray(seed).copy(), target,
                                       branch=np.asarray(seed).copy())
        p, r = self.pin.fk(q)
        ep = float(max(np.linalg.norm(p-target[:, :3], axis=1)))
        er = float(max((Rotation.from_matrix(r).inv()*Rotation.from_quat(target[:, 3:])).magnitude()))
        self.calls += 1
        self.max_iterations = max(self.max_iterations, *iterations)
        self.max_position_error_m = max(self.max_position_error_m, ep)
        self.max_rotation_error_rad = max(self.max_rotation_error_rad, er)
        if not np.isfinite(q).all() or ep > IK_POSITION_TOL or er > IK_ROTATION_TOL:
            raise ValueError('Pinocchio IK residual exceeds tolerance: %.6f m / %.6f rad' % (ep, er))
        if np.any(q < self.pin.lower) or np.any(q > self.pin.upper):
            raise ValueError('Pinocchio IK solution outside executable joint limits')
        return q

    def report(self):
        return dict(implementation='Pinocchio Jacobian damped least squares with seed nullspace pull; '
                    'bounded SciPy fallback uses Pinocchio FK',
                    mujoco_ik_used=False, calls=self.calls, max_iterations=self.max_iterations,
                    max_position_error_m=self.max_position_error_m,
                    max_rotation_error_rad=self.max_rotation_error_rad, dependencies=self.provenance)
