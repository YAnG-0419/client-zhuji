"""Offline tests for the --permissive downgrade mechanism."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'tools'))

import findings


@pytest.fixture(autouse=True)
def _clean():
    findings.end()
    yield
    findings.end()


def test_permit_is_false_until_recording_begins():
    """Without --permissive every check keeps its original abort behaviour."""
    assert findings.permit('anything', 'detail') is False
    assert findings.active() is None


def test_permit_records_and_allows_once_begun(tmp_path):
    findings.begin(tmp_path/'violations.jsonl')
    assert findings.permit('ik_residual', 'too far', value=.02) is True
    assert findings.permit('ik_residual', 'still far', value=.05) is True
    summary = findings.active().summary()
    assert summary['counts']['ik_residual'] == 2
    assert summary['worst_value']['ik_residual'] == .05
    findings.end()
    lines = [json.loads(l) for l in (tmp_path/'violations.jsonl').read_text().splitlines()]
    assert [l['kind'] for l in lines] == ['ik_residual', 'ik_residual']
    assert lines[1]['value'] == .05 and 'wall_s' in lines[1]


def test_findings_are_flushed_while_the_run_continues(tmp_path):
    """A long run must be inspectable before it finishes."""
    findings.begin(tmp_path/'violations.jsonl')
    findings.permit('hand_tracking_error', 'drifting', value=.3)
    assert (tmp_path/'violations.jsonl').read_text().count('\n') == 1
    findings.end()


def test_clip_to_envelope_reports_the_excess_and_returns_a_reachable_target(tmp_path):
    findings.begin(tmp_path/'violations.jsonl')
    lower, upper = np.full(14, -.7505), np.full(14, .7505)
    q = np.zeros(14); q[13] = .7834; q[5] = -.79
    target = findings.clip_to_envelope(lower, upper, q, 'segment 7', ['j%d' % i for i in range(14)])
    assert target[13] == pytest.approx(.7505) and target[5] == pytest.approx(-.7505)
    assert np.all(target >= lower) and np.all(target <= upper)
    entry = json.loads((tmp_path/'violations.jsonl').read_text().splitlines()[0])
    assert entry['kind'] == 'movej_target_clipped'
    assert entry['value'] == pytest.approx(.0395, abs=1e-4)   # 0.79 - 0.7505
    assert sorted(entry['joints']) == ['j13', 'j5']
    findings.end()


def test_clip_to_envelope_is_silent_when_nothing_is_outside(tmp_path):
    findings.begin(tmp_path/'violations.jsonl')
    q = np.zeros(14); q[13] = .5
    target = findings.clip_to_envelope(np.full(14, -.7505), np.full(14, .7505), q, 'segment 1')
    assert target[13] == .5
    assert (tmp_path/'violations.jsonl').read_text() == ''
    findings.end()


def test_clip_without_recording_still_returns_an_executable_target():
    """Clipping protects the firmware contract even in strict mode."""
    q = np.zeros(14); q[13] = .9
    target = findings.clip_to_envelope(np.full(14, -.7505), np.full(14, .7505), q, 'x')
    assert target[13] == pytest.approx(.7505)


def test_guard_values_measures_without_enforcing():
    """stage_guard aborts past its thresholds; guard_values only reports."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'replay_module', str(Path(__file__).resolve().parents[1]/'replay.py'))
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    from home_policy import stage_guard
    start, goal = [0.]*16, [1.]*16
    q = [0.]*16; q[0] = .5; q[1] = -.4      # far off the straight path
    dq = [0.]*16; dq[0] = 2.
    with pytest.raises(RuntimeError):
        stage_guard(q, dq, start, goal, range(14))
    values = replay.guard_values(q, dq, start, goal, range(14))
    assert values['max_speed_rad_s'] == 2.
    assert values['path_deviation_rad'] > np.radians(3)
    assert 0. <= values['progress'] <= 1.
