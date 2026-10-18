import unittest
import numpy as np
from airavoid.geometry import Config,ThreatEstimator,advisory


class GeometryTests(unittest.TestCase):
    def history(self,vx=0.,vy=0.,growth=.08,span=15.,noise=0.):
        rng=np.random.default_rng(12)
        return [(float(t),400+vx*t+float(rng.normal(0,noise)),200+vy*t,span*np.exp(growth*t)) for t in np.arange(0,4.01,.1)]

    def test_constant_bearing_growth_off_center_warns(self):
        e=ThreatEstimator().assess(self.history(),4)
        self.assertEqual(e['state'],'threat');self.assertGreater(e['tau'],10)

    def test_crossing_growing_target(self):
        e=ThreatEstimator().assess(self.history(vx=60),4)
        self.assertEqual(e['state'],'nonthreat')

    def test_no_growth_is_uncertain(self):
        self.assertEqual(ThreatEstimator().assess(self.history(growth=0),4)['state'],'uncertain')

    def test_camera_unknown_abstains(self):
        self.assertEqual(ThreatEstimator().assess(self.history(),4,False)['reason'],'camera')

    def test_small_target_abstains(self):
        self.assertEqual(ThreatEstimator().assess(self.history(span=2),4)['reason'],'tiny')

    def test_receding(self):
        e=ThreatEstimator().assess(self.history(growth=-.12,span=30),4)
        self.assertEqual(e['state'],'uncertain')
        self.assertEqual(e['reason'],'shrink_ambiguous')

    def test_legacy_stable_shape_recession_hypothesis(self):
        model=ThreatEstimator(Config(allow_recession_clearance=True))
        self.assertEqual(model.assess(self.history(growth=-.12,span=30),4)['state'],'nonthreat')

    def test_gap_and_identity_ambiguity(self):
        self.assertEqual(ThreatEstimator().assess(self.history(),5)['reason'],'lost')
        self.assertEqual(ThreatEstimator().assess(self.history(),4,True,True)['reason'],'association')

    def test_advice_never_asserts_safe_direction(self):
        self.assertIsNone(advisory([])['recommendation'])

if __name__=='__main__':unittest.main()
