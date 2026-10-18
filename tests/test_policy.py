import unittest
from airavoid.policy import TemporalPolicy


class PolicyTests(unittest.TestCase):
    def estimate(self,state='nonthreat',reason='receding'):
        return {'state':state,'reason':reason}

    def test_short_contraction_does_not_clear_target(self):
        p=TemporalPolicy()
        for t in [0,.08,.16,.24,.32]:
            self.assertEqual(p.apply(1,t,self.estimate())['state'],'uncertain')

    def test_persistent_crossing_can_become_non_threat(self):
        p=TemporalPolicy()
        for i in range(16):e=p.apply(1,i*.08,self.estimate(reason='crossing'))
        self.assertEqual(e['state'],'nonthreat')

    def test_missing_or_ambiguous_evidence_cancels_confirmation(self):
        p=TemporalPolicy()
        for i in range(12):p.apply(1,i*.08,self.estimate())
        p.apply(1,.96,self.estimate('uncertain','lost'))
        self.assertEqual(p.apply(1,1.04,self.estimate())['state'],'uncertain')

    def test_warning_is_confirmed_faster_than_clearance(self):
        p=TemporalPolicy()
        for t in [0,.08,.16,.24]:e=p.apply(1,t,self.estimate('threat','bearing_growth'))
        self.assertEqual(e['state'],'threat')

    def test_track_ids_have_independent_confirmation(self):
        p=TemporalPolicy();p.apply(1,0,self.estimate());p.apply(1,.2,self.estimate())
        self.assertEqual(p.apply(2,1.3,self.estimate())['state'],'uncertain')
