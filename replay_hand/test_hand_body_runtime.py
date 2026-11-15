import unittest
from unittest.mock import Mock,patch
import numpy as np
from full_sync import CombinedSession


class HandBodyRuntimeTests(unittest.TestCase):
    def session(self,clearance):
        session=CombinedSession.__new__(CombinedSession)
        session.t=np.array([0.,1.]);session.hand_data=np.zeros((2,2,20));session.hand_stage='test'
        session.hand_session=Mock();session.hand_session.tick.return_value=np.full((2,20),.1)
        session.hand_checker=Mock();session.hand_checker.clearance.return_value=(clearance,['l_pinky','r_palm'])
        return session

    def test_actual_hand_and_arm_feedback_are_checked(self):
        session=self.session(.03);q=np.arange(16)/100
        with patch('full_sync.ArmSession.tick',return_value=(q,np.zeros(16),0.,None)):
            result=session.tick(None,q,q,.4)
        np.testing.assert_array_equal(session.hand_checker.set_q.call_args.args[0],q)
        np.testing.assert_array_equal(session.hand_checker.set_q.call_args.args[1],np.full((2,20),.1))
        self.assertEqual(session.hand_clearance_last['source_time_s'],.4)

    def test_contact_propagates_to_existing_stop_cleanup(self):
        session=self.session(-.001);q=np.zeros(16)
        with patch('full_sync.ArmSession.tick',return_value=(q,q,0.,None)):
            with self.assertRaisesRegex(RuntimeError,'Hand/body live clearance'):
                session.tick(None,q,q)


if __name__=='__main__':unittest.main()
