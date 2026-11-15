from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))
from transport import validate_movej


def test_target_inside_the_supplied_range_is_accepted():
    q = np.zeros(14); q[13] = .7
    actual, seconds = validate_movej({'joint': q.tolist(), 'time': 20.},
                                      np.full(14, -1.047), np.full(14, 1.047))
    assert actual[13] == .7 and seconds == 20.


@pytest.mark.parametrize('value', [1.1, float('nan')])
def test_invalid_or_out_of_range_target_is_rejected(value):
    q = np.zeros(14); q[13] = value
    with pytest.raises(ValueError):
        validate_movej({'joint': q.tolist(), 'time': 20.}, np.full(14, -1.047), np.full(14, 1.047))


def test_model_limits_are_intersected_with_the_sdk_movej_envelope():
    """A firmware-refused target must fail planning, not time out on hardware."""
    from transport import replay_backend
    # replay_backend() puts the wanchi modules, including movej_limits, on sys.path.
    _, Model, _ = replay_backend()
    from movej_limits import LOWER, UPPER
    model = Model()
    urdf_lower, urdf_upper = model.model.jnt_range[model.jids].T
    assert np.all(model.lower[:14] >= LOWER-1e-12)
    assert np.all(model.upper[:14] <= UPPER+1e-12)
    assert np.all(model.lower[:14] >= urdf_lower[:14]-1e-12)
    assert np.all(model.upper[:14] <= urdf_upper[:14]+1e-12)
    # The wrists are where the SDK envelope is strictly narrower than the URDF.
    wrists = [5, 6, 12, 13]
    assert np.all(model.upper[wrists] < urdf_upper[wrists])
    # The recorded 0.7834 rad wrist pitch that stalled 20261104_150124_104108
    # is outside the executable range and must now be rejected up front.
    q = np.zeros(14); q[12] = .7834
    with pytest.raises(ValueError, match='executable joint range'):
        validate_movej({'joint': q.tolist(), 'time': 20.}, model.lower[:14], model.upper[:14])


def test_invalid_duration_is_rejected():
    with pytest.raises(ValueError, match='duration'):
        validate_movej({'joint': [0.]*14, 'time': float('nan')}, np.full(14, -1.), np.full(14, 1.))


def test_wrist_internal_one_mm_exception_keeps_other_pairs_at_five_mm(monkeypatch):
    import mujoco
    from transport import replay_backend
    _, Model, _ = replay_backend()
    model = Model()
    names = lambda pair: frozenset(model.body_name(model.model.geom_bodyid[g]) for g in pair)
    internal = next(pair for pair in model.pairs if names(pair) in model.internal_wrist_pairs)
    other = next(pair for pair in model.pairs if names(pair) not in model.internal_wrist_pairs)
    monkeypatch.setattr(mujoco, 'mj_geomDistance', lambda *args: .003)
    model.pairs = [internal]
    assert model.clearance()[0] == pytest.approx(.007)  # Internal 3 mm exceeds the restored 1 mm requirement.
    model.pairs = [other]
    assert model.clearance()[0] == pytest.approx(.003)  # Same distance fails the cross-link 5 mm gate.
