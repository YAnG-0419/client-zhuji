import unittest
from unittest.mock import Mock,patch
import numpy as np
from hand_approach import plan_approach


class PreparationRouteTests(unittest.TestCase):
    def test_requires_both_arm_and_hand_certification(self):
        start=np.zeros(16);goal=np.ones(16)*.1;mid=np.ones(16)*.05
        model=Mock();model.check_path.return_value={'ok':True}
        checker=Mock();checker.check_path.side_effect=[ValueError('thumb at torso'),{'ok':True}]
        choices=[('direct',np.stack([start,goal])),('via',np.stack([start,mid,goal]))]
        with patch('hand_approach.candidates',return_value=iter(choices)):
            path,report=plan_approach(model,checker,start,goal,np.zeros((2,20)))
        self.assertEqual(report['selected'],'via')
        self.assertEqual(len(path),3)
        self.assertFalse(report['attempts'][0]['accepted'])
        self.assertEqual(model.check_path.call_count,2)
        self.assertEqual(checker.check_path.call_count,2)

    def test_no_route_is_not_silently_accepted(self):
        start=np.zeros(16);model=Mock();model.check_path.side_effect=ValueError('collision')
        checker=Mock()
        with patch('hand_approach.candidates',return_value=iter([('direct',np.stack([start,start]))])):
            with self.assertRaisesRegex(ValueError,'No certified'):
                plan_approach(model,checker,start,start,np.zeros((2,20)))
        checker.check_path.assert_not_called()


if __name__=='__main__':unittest.main()
