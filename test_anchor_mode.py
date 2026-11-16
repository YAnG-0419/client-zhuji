import json
from pathlib import Path
import unittest

import numpy as np

from execution_postprocess import anchor_weights, apply_local_anchor

HERE = Path(__file__).resolve().parent


def frames(n=32, *, drift=0.0):
    """A fitted root trajectory that stays put (plus optional per-step drift)."""
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, 2, 3] = drift * np.arange(n)
    return out


class AnchorModeTests(unittest.TestCase):
    def test_weights_per_mode(self):
        self.assertEqual(anchor_weights(4, "constant").tolist(), [1.0] * 4)
        self.assertEqual(anchor_weights(4, "none").tolist(), [0.0] * 4)
        decay = anchor_weights(32, "decay", 12)
        self.assertAlmostEqual(decay[0], 1.0)
        self.assertAlmostEqual(decay[-1], 0.0)

    def test_constant_mode_does_not_chase_a_fit_bias(self):
        """A fit that is a fixed 17 mm below the measured pose must not move the hand."""
        measured = np.eye(4)
        fit = frames()
        fit[:, 2, 3] -= 0.017
        limits = np.stack([np.full(20, -2.0), np.full(20, 2.0)], axis=1)
        q = np.zeros((32, 20))
        for mode, expected in (("decay", -0.017), ("constant", 0.0)):
            roots, _, report = apply_local_anchor(
                fit, q, measured, np.zeros(20), limits, mode=mode, decay_steps=12,
                apply_to_hand_q=True)
            self.assertAlmostEqual(roots[0][2, 3], 0.0, places=9)        # starts measured
            self.assertAlmostEqual(roots[-1][2, 3], expected, places=9)  # ends at fit or holds
            self.assertAlmostEqual(report["local_offset_translation_norm_m"], 0.017, places=9)

    def test_constant_mode_still_follows_model_motion(self):
        measured = np.eye(4)
        fit = frames(drift=-0.001)          # the model really does descend 1 mm per step
        fit[:, 2, 3] -= 0.017               # on top of a constant fit bias
        limits = np.stack([np.full(20, -2.0), np.full(20, 2.0)], axis=1)
        roots, _, _ = apply_local_anchor(fit, np.zeros((32, 20)), measured, np.zeros(20),
                                         limits, mode="constant", decay_steps=12,
                                         apply_to_hand_q=True)
        self.assertAlmostEqual(roots[-1][2, 3], -0.031, places=9)   # 31 steps x 1 mm only


class PlannerWiringTests(unittest.TestCase):
    def test_config_block_and_cli_choices_agree(self):
        config = json.loads((HERE / "deployment_config.json").read_text())
        self.assertIn(config["execution"]["anchor"]["mode"], ("decay", "constant", "none"))
        source = (HERE / "live_planner.py").read_text()
        self.assertIn("mode=self.anchor_mode", source)
        self.assertNotIn('mode="decay"', source)
        self.assertIn('"anchor_mode": self.anchor_mode', source)
        self.assertIn("--anchor-mode", (HERE / "live_client.py").read_text())


if __name__ == "__main__":
    unittest.main()
