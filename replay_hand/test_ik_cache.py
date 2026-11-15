"""Offline regression tests for cache identity and independently checked FK."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

import ik_cache
from ik_cache import read_cache, write_cache, validate_seed, signature, cache_path
from replay import seed_matches_target


class Model:
    lower = np.full(16, -2.)
    upper = np.full(16, 2.)
    dependencies = {'model.xml': 'model-digest'}

    def set_q(self, q):
        self.q = np.array(q)

    def poses(self):
        a = np.zeros((2, 7)); a[:, 6] = 1.
        a[:, 0] = self.q[[0, 7]]
        return a


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.t = np.array([0., .02, .04])
        self.q = np.zeros((3, 16)); self.q[:, 0] = [0., .01, .02]
        self.eef = []
        for q in self.q:
            self.model.set_q(q); self.eef.append(self.model.poses())
        self.eef = np.array(self.eef)
        self.sig = {'episode': 'a', 'model': 'b', 'code': 'c'}

    def read(self, path, sig=None):
        return read_cache(path, sig or self.sig, self.t, self.eef, self.model, .02, .035)

    def test_round_trip_and_signature_rejection(self):
        with TemporaryDirectory() as folder:
            p = Path(folder)/'cache.npz'
            write_cache(p, self.sig, self.t, self.eef, self.q, {'test': True})
            q, report = self.read(p)
            np.testing.assert_array_equal(q, self.q)
            self.assertEqual(report['max_position_error_m'], 0.)
            for key in self.sig:
                changed = dict(self.sig); changed[key] = 'changed'
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    self.read(p, changed)

    def test_payload_corruption_rejected(self):
        with TemporaryDirectory() as folder:
            p = Path(folder)/'cache.npz'
            write_cache(p, self.sig, self.t, self.eef, self.q, {})
            with np.load(p) as z:
                data = {k: z[k] for k in z.files}
            data['q16'][1, 0] += .001
            np.savez(p, **data)
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                self.read(p)

    def test_fk_bounds_continuity_and_head_are_rechecked(self):
        mutations = [lambda q: q.__setitem__((1, 0), .05),
                     lambda q: q.__setitem__((1, 1), 2.1),
                     lambda q: q.__setitem__((1, 1), .21),
                     lambda q: q.__setitem__((1, 14), .001),
                     lambda q: q.__setitem__((1, 0), np.nan),
                     lambda q: q.__setitem__((0, 0), .002)]
        for mutate in mutations:
            q = self.q.copy(); mutate(q)
            with TemporaryDirectory() as folder:
                p = Path(folder)/'cache.npz'
                write_cache(p, self.sig, self.t, self.eef, q, {})
                with self.assertRaises(ValueError): self.read(p)

    def test_time_and_targets_cannot_be_swapped(self):
        with TemporaryDirectory() as folder:
            p = Path(folder)/'cache.npz'
            for t, eef in [(self.t*2, self.eef), (self.t, self.eef+.001)]:
                write_cache(p, self.sig, t, eef, self.q, {})
                with self.assertRaises(ValueError): self.read(p)

    def test_loader_hit_avoids_solver_and_bad_cache_falls_back(self):
        hands = np.zeros((3, 2, 20))
        meta = dict(placement_policy='preserve_common_frame_v1')
        values = (self.t, self.eef, hands, self.q, dict(meta))
        with TemporaryDirectory() as folder:
            out = Path(folder); p = out/'cache.npz'
            write_cache(p, self.sig, self.t, self.eef, self.q, {})
            with patch('episode_cross.load_cross_episode', side_effect=lambda *a: (self.t, self.eef, hands, dict(meta))), \
                    patch('transport.replay_backend', return_value=(None, Model, None)), \
                    patch('ik_cache.signature', return_value=self.sig), \
                    patch('ik_cache.cache_path', return_value=p), \
                    patch('replay_hand.cross_loader', return_value=values) as cold:
                loaded = ik_cache.load_full_episode(out/'input.npz', 50., out)
                self.assertTrue(loaded[-1]['ik_cache']['hit'])
                cold.assert_not_called()
                p.write_bytes(b'broken archive')
                loaded = ik_cache.load_full_episode(out/'input.npz', 50., out)
                cold.assert_called_once()
                self.assertFalse(loaded[-1]['ik_cache']['hit'])

    def test_accept_existing_seed_only_if_fk_and_bounds_pass(self):
        self.assertTrue(seed_matches_target(self.model, self.q[1], self.eef[1]))
        q = self.q[1].copy(); q[0] += .1
        self.assertFalse(seed_matches_target(self.model, q, self.eef[1]))
        for value in (2.1, np.nan):
            q = self.q[1].copy(); q[1] = value
            self.assertFalse(seed_matches_target(self.model, q, self.eef[1]))

    def test_valid_near_bound_seed_is_not_repeatedly_nudged(self):
        q = self.q[1].copy(); q[1] = self.model.lower[1]+1.5e-5
        self.assertTrue(seed_matches_target(self.model, q, self.eef[1]))
        q[1] = self.model.lower[1]+.5e-5
        self.assertFalse(seed_matches_target(self.model, q, self.eef[1]))

    def test_fingerprint_tracks_inputs_models_bounds_and_code(self):
        names = ['replay_hand/ik_cache.py', 'replay_hand/planner_cross.py',
                 'replay_hand/episode_cross.py', 'replay_hand/anchor_cross.py',
                 'replay_hand/replay_hand.py', 'replay_hand/full_sync.py',
                 'replay/replay.py', 'replay/transport.py', 'replay/replay_thresholds.py',
                 'tools/robot_home.py', 'poses/home.json']
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for name in names:
                p = root/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('a')
            meta = dict(episode_sha256='input', fk={}, placement_policy='preserve_common_frame_v1')
            hands = np.zeros((3, 2, 20))
            def sig(): return signature(root, self.model, self.t, self.eef, hands, meta, 50.)
            original = cache_path(root, sig())
            meta['episode_sha256'] = 'changed'
            self.assertNotEqual(original, cache_path(root, sig()))
            meta['episode_sha256'] = 'input'
            (root/names[0]).write_text('changed-code')
            self.assertNotEqual(original, cache_path(root, sig()))
            (root/names[0]).write_text('a')
            self.model.lower = self.model.lower-.1
            self.assertNotEqual(original, cache_path(root, sig()))


if __name__ == '__main__':
    unittest.main()
