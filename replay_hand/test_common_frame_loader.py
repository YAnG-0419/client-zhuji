"""Run in the deployed replay Python environment; hardware backends are mocked."""
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
import replay_hand as entry


class CommonFrameLoaderTests(unittest.TestCase):
    def test_common_frame_never_reanchors(self):
        t=np.array([0.,1.]);eef=np.zeros((2,2,7));eef[:,:,6]=1
        hands=np.zeros((2,2,20));meta={'placement_policy':'preserve_common_frame_v1'}
        seed=np.zeros((2,16))
        with patch.object(entry,'load_cross_episode',return_value=(t,eef,hands,meta)), \
             patch.object(entry,'backend',return_value=(None,lambda:object(),None)), \
             patch.object(entry,'anchor_episode',side_effect=AssertionError('Must not anchor')), \
             patch.object(entry,'solve_seed_path',return_value=(seed,{})) as solve:
            result=entry.cross_loader(Path('unused'),50,Path('unused'))
        self.assertIs(result[1],eef)
        self.assertIsNone(solve.call_args.kwargs['preferred'])
        self.assertTrue(result[4]['cross_anchor']['bimanual_relative_pose_preserved'])

    def test_unknown_policy_fails_closed(self):
        with patch.object(entry,'load_cross_episode',return_value=(None,None,None,{'placement_policy':'typo'})), \
             patch.object(entry,'backend',return_value=(None,lambda:object(),None)):
            with self.assertRaisesRegex(ValueError,'Unknown placement'):
                entry.cross_loader(Path('unused'),50,Path('unused'))


if __name__=='__main__':unittest.main()
