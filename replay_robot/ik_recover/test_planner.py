import math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from replay_robot.ik_recover.planner import Limits, RecoveryPlanner, Reference


class LinearModel:
    lower = np.full(16, -1.)
    upper = np.full(16, 1.)

    def set_q(self, q):
        self.q = q

    def poses(self):
        eef = np.zeros((2, 7))
        eef[:, 0] = self.q[[0, 7]]
        eef[:, 6] = 1.
        return eef


def planner(position_limit=.035, gap=.041, sweep=None):
    t = np.arange(5) * .02
    goals = np.zeros((5, 2, 7))
    goals[:, :, 6] = 1.
    goals[:, 0, 0] = [0., .005, .040, .015, .020]
    hands = np.arange(200).reshape(5, 2, 20)

    def solve(seed, target):
        seed[0] = min(target[0, 0], .016)
        return seed

    def error(actual, target):
        p = np.linalg.norm(actual[:, :3] - target[:, :3], axis=1)
        r = (Rotation.from_quat(actual[:, 3:]).inv() *
             Rotation.from_quat(target[:, 3:])).magnitude()
        return p, r

    return RecoveryPlanner(t, goals, hands, Reference(t, goals), LinearModel(),
                           solve, solve, error, lambda _: None,
                           sweep or (lambda q, h: {}),
                           (.020, math.radians(2), math.radians(3)),
                           Limits(position_limit, math.radians(3), gap))


def test_bridge_keeps_timestamps_anchors_and_synchronized_hands():
    checked = []
    solver = planner(sweep=lambda q, h: checked.append((q.copy(), h.copy())) or {})
    spans = list(solver.sections(np.zeros(16)))
    start, end, path, _, detail = spans[1]
    assert (start, end) == (1, 3)
    assert detail['peak_error']['position_mm'] == pytest.approx([30, 0])
    np.testing.assert_array_equal(checked[0][1], solver.hands[1:4])
    np.testing.assert_allclose(path[:, 0], [.005, .010, .015])
    np.testing.assert_array_equal(path[0], spans[0][2][-1])
    np.testing.assert_array_equal(path[-1], spans[2][2][0])


@pytest.mark.parametrize('kwargs', [dict(position_limit=.023), dict(gap=.021)])
def test_failure_does_not_yield_a_partial_bridge(kwargs):
    spans = planner(**kwargs).sections(np.zeros(16))
    assert next(spans)[1] == 1
    with pytest.raises(ValueError, match='No recovery'):
        next(spans)


def test_a_collision_rejects_the_bridge():
    def blocked(q, h):
        raise ValueError('collision clearance')

    spans = planner(sweep=blocked).sections(np.zeros(16))
    next(spans)
    with pytest.raises(ValueError, match='collision clearance'):
        next(spans)


def test_original_source_knots_between_replay_frames_are_checked():
    solver = planner()
    raw_t = np.array([0., .007, .020])
    raw = solver.goals[:3].copy()
    raw[:, 0, 0] = [0., .04, 0.]
    solver.intervals[0] = Reference(raw_t, raw).interval(0., .020)
    with pytest.raises(ValueError, match='40.00 mm'):
        solver.check_section(0, np.zeros((2, 16)), .035, math.radians(3))
