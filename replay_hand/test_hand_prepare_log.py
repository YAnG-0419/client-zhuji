import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import replay as base
from hand_prepare_log import prepare_hands


class HandLogTests(unittest.TestCase):
    def run_case(self, error):
        clock = [0.]
        session = base.Session.__new__(base.Session)
        session.hand_data = np.zeros((1, 2, 20))
        session.last_hand = np.zeros((2, 20))
        def tick(*a, **kw):
            if error > .2:
                raise RuntimeError('Hand tracking error exceeds 0.2 rad')
            h = np.zeros((2, 20)); h[1, 5] = error
            return np.zeros(16), np.zeros(16), 0., h
        session.tick = tick
        result = {}
        with TemporaryDirectory() as folder, patch.object(base, 'HAND_FIRST_FRAME_TOL', .2), \
                patch('time.monotonic', lambda: clock[0]), \
                patch('time.sleep', lambda dt: clock.__setitem__(0, clock[0]+dt)):
            try:
                prepare_hands(session, 10., np.zeros(16), Path(folder), result, .2)
            except RuntimeError:
                if error <= .2:
                    raise
            self.assertIs(session.tick, tick)
            rows = [json.loads(s) for s in (Path(folder)/'hand_preparation.jsonl').read_text().splitlines()]
        return result['hand_preparation'], rows

    def test_reported_residual_passes_and_is_identified(self):
        r, rows = self.run_case(.18001)
        self.assertTrue(r['completed'])
        self.assertEqual(r['last_feedback']['worst_joint'], 'right_finger2_joint2')
        self.assertEqual(rows[-1]['event'], 'end')

    def test_runtime_failure_propagates_and_tick_is_restored(self):
        r, _ = self.run_case(.21)
        self.assertNotIn('completed', r)
        self.assertIn('0.2 rad', r['error'])


if __name__ == '__main__':
    unittest.main()
