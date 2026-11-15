from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clearance import certify_path


class CoarseBound:
    def check_path(self, qs, clearance):
        assert clearance == .005
        raise ValueError('Conservative interval bound is below 5 mm')


def test_uncertain_coarse_bound_can_be_certified_without_relaxing_margin():
    qs = np.zeros((3, 16)); qs[:, 0] = [0., .01, .02]
    seen = []
    def refine(model, start, goal):
        seen.append((start.copy(), goal.copy()))
        return .00646
    report = certify_path(CoarseBound(), qs, refine)
    assert report['required_clearance_m'] == .005
    assert report['conservative_min_clearance_m'] == .00646
    assert report['refined_segments'] == [1, 2]
    assert len(seen) == 2
    assert np.array_equal(seen[-1][1], qs[-1])


def test_uncertified_segment_remains_rejected():
    def refine(model, start, goal):
        raise ValueError('Endpoint clearance 4.9 mm')
    with pytest.raises(ValueError, match='waypoint segment 1'):
        certify_path(CoarseBound(), np.zeros((2, 16)), refine)


def test_refinement_cannot_return_a_bound_below_five_mm():
    with pytest.raises(ValueError, match='below 5 mm'):
        certify_path(CoarseBound(), np.zeros((2, 16)), lambda *args: .0049)
