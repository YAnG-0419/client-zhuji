"""Offline boundary tests for the operator-selected replay thresholds."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'replay'), str(ROOT/'replay_hand'), str(ROOT/'tools')]


@pytest.mark.parametrize('position,angle,accepted', [(.015, 1.5, True), (.021, 0., False), (0., 2.1, False)])
def test_cross_seed_accepts_requested_pose_tolerance(tmp_path, monkeypatch, position, angle, accepted):
    import planner_cross
    import replay_thresholds as thresholds
    q = np.zeros(16)
    goal = np.zeros((2, 7)); goal[:, 6] = 1.
    actual = goal.copy(); actual[1, 0] = position
    actual[1, 3:] = Rotation.from_euler('z', angle, degrees=True).as_quat()
    model = SimpleNamespace(lower=np.full(16, -1.), upper=np.full(16, 1.),
                            set_q=lambda q: None, poses=lambda: actual.copy(),
                            clearance=lambda: (.03, ['a', 'b']))
    base = SimpleNamespace(solve_eef=lambda model, seed, target: seed,
                           IK_POSITION_TOL=thresholds.IK_POSITION_TOL,
                           IK_ROTATION_TOL=thresholds.IK_ROTATION_TOL)
    monkeypatch.setattr(planner_cross, 'candidates', lambda *a, **kw: [('test', q)])
    args = (base, model, ROOT, np.array([0., .02]), np.stack([goal, goal]), tmp_path)
    if accepted:
        path, _ = planner_cross.solve_seed_path(*args)
        assert path.shape == (2, 16)
    else:
        with pytest.raises(ValueError, match='first-frame'):
            planner_cross.solve_seed_path(*args)


@pytest.mark.parametrize('internal,distance,accepted', [
    (True, .0009, False), (True, .0011, True), (True, .0058, True),
    (False, .003, False), (False, .0049, False), (False, .0051, True)])
def test_pair_specific_collision_requirements(monkeypatch, internal, distance, accepted):
    import mujoco
    from transport import replay_backend
    _, Model, _ = replay_backend()
    model = Model()
    pair = next(p for p in model.pairs if (frozenset(
        model.body_name(model.model.geom_bodyid[g]) for g in p) in model.internal_wrist_pairs) == internal)
    model.pairs = [pair]
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *args: distance)
    if accepted:
        result = model.check_path(np.zeros((1, 16)), clearance=.001)
        assert result['required_clearance_m'] == .005
    else:
        with pytest.raises(ValueError, match='clearance'):
            model.check_path(np.zeros((1, 16)), clearance=.001)


def test_changed_start_cannot_spend_clearance_below_five_mm(monkeypatch):
    import replay_movej_path as path
    monkeypatch.setattr(path, 'check_segment', lambda *args: .006)
    radius = np.r_[.1, np.zeros(15)]
    model = SimpleNamespace(pairs=[(0, 1)], joint_radius_bounds=lambda *a, **kw: radius,
                            lower=np.full(16, -1.), upper=np.full(16, 1.))
    states = iter(np.r_[x, np.zeros(15)] for x in (0., .01, .02, .03))
    with pytest.raises(RuntimeError, match='5 mm'):
        path.check_current_segment(model, lambda: (next(states), np.zeros(16)), np.zeros(16))


@pytest.mark.parametrize('distance,accepted', [(.0049, False), (.0051, True)])
def test_live_hand_body_collision_boundary(monkeypatch, distance, accepted):
    from full_sync import CombinedSession
    from arm_only import ArmSession
    q = np.zeros(16)
    monkeypatch.setattr(ArmSession, 'tick', lambda *a: (q, q, 0., None))
    session = CombinedSession.__new__(CombinedSession)
    session.t = np.array([0., 1.]); session.hand_data = np.zeros((2, 2, 20))
    session.hand_stage = 'test'
    session.hand_session = SimpleNamespace(tick=lambda *a: np.zeros((2, 20)))
    session.hand_checker = SimpleNamespace(set_q=lambda *a: None,
                                          clearance=lambda: (distance, ['hand', 'body']))
    if accepted:
        session.tick(None, q, q)
    else:
        with pytest.raises(RuntimeError, match='Hand/body live clearance'):
            session.tick(None, q, q)
