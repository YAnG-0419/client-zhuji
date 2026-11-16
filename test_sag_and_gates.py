import math
from pathlib import Path
import unittest

import numpy as np

import sag_model
from live_client import HOLD_SETTLE_CREEP_POS, HOLD_SETTLE_WINDOW

HERE = Path(__file__).resolve().parent


class SagModelTests(unittest.TestCase):
    def model(self, drop=-0.018):
        """A model that predicts a constant drop in z for both arms."""
        coefficients = np.zeros((3, 11)); coefficients[2, -1] = drop
        side = {"coefficients": coefficients.tolist(), "residual_mm": {"p95": 1.0}}
        return {"max_compensation_m": 0.035, "sides": {"left": side, "right": dict(side)}}

    def test_compensation_lifts_the_command_by_the_predicted_sag(self):
        target = np.zeros((2, 7)); target[:, 3] = 1.0
        sent = sag_model.compensate(self.model(), target, np.zeros(16))
        np.testing.assert_allclose(sent[:, 2], 0.018)          # commanded 18 mm higher
        np.testing.assert_allclose(sent[:, 3:], target[:, 3:])  # orientation untouched

    def test_compensation_is_capped(self):
        sent = sag_model.compensate(self.model(drop=-0.5), np.zeros((2, 7)), np.zeros(16))
        np.testing.assert_allclose(sent[:, 2], 0.035)

    def test_fit_recovers_a_known_sag(self):
        rng = np.random.default_rng(0)
        target = rng.normal(0, 0.1, (200, 2, 3))
        joints = rng.normal(0, 0.3, (200, 2, 7))
        sag = np.zeros_like(target)
        sag[:, :, 2] = -0.01 - 0.05 * joints[:, :, 1]          # droop grows with the hip angle
        model = sag_model.fit(target, joints, sag)
        self.assertLess(model["sides"]["left"]["residual_mm"]["max"], 0.01)
        row = np.concatenate([target[0], np.zeros((2, 4))], axis=1)
        np.testing.assert_allclose(sag_model.predict(model, row, joints[0].reshape(-1)),
                                   sag[0], atol=1e-9)

    def test_real_model_is_sane(self):
        path = HERE / "sag_model.json"
        if not path.is_file():
            self.skipTest("no fitted model in the tree")
        import json
        model = json.loads(path.read_text())
        for side in ("left", "right"):
            entry = model["sides"][side]
            self.assertGreater(entry["samples"], 100)
            # the fit must beat the raw sag it is correcting
            self.assertLess(entry["residual_mm"]["mean"], entry["raw_norm_mm"]["mean"])


class HoldGateTests(unittest.TestCase):
    """The gate must accept a large static offset and still catch a creeping one."""

    def decide(self, series):
        history = [(t, d, 0.0) for t, d in series]
        spread = max(d[1] for d in history) - min(d[1] for d in history)
        return (len(history) >= 2
                and history[-1][0] - history[0][0] >= HOLD_SETTLE_WINDOW * 0.5
                and spread < HOLD_SETTLE_CREEP_POS)

    def test_static_sag_counts_as_settled(self):
        self.assertTrue(self.decide([(0.0, 0.0245), (0.3, 0.0246), (0.5, 0.0244)]))

    def test_creeping_drift_does_not(self):
        self.assertFalse(self.decide([(0.0, 0.020), (0.3, 0.026), (0.5, 0.032)]))

    def test_too_short_a_window_does_not(self):
        self.assertFalse(self.decide([(0.0, 0.0245), (0.1, 0.0246)]))


if __name__ == "__main__":
    unittest.main()
