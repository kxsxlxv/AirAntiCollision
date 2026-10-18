import unittest
import numpy as np
from airavoid.geometry import ThreatEstimator


class IndependentScaleTests(unittest.TestCase):
    def test_tracking_box_growth_does_not_replace_detector_size(self):
        times=np.arange(0,4.01,.1)
        appearance=[(float(t),400.,250.,float(15*np.exp(.3*t))) for t in times]
        independent=[(float(t),400.,250.,15.) for t in times[::2]]
        self.assertEqual(ThreatEstimator().assess(appearance,4)['state'],'threat')
        self.assertEqual(ThreatEstimator().assess(appearance,4,scale_history=independent)['state'],'uncertain')

    def test_sparse_independent_size_abstains(self):
        h=[(float(t),400.,250.,20.) for t in np.arange(0,4.01,.1)]
        e=ThreatEstimator().assess(h,4,scale_history=h[::10])
        self.assertEqual(e['reason'],'scale_history')

    def test_stale_independent_size_abstains(self):
        h=[(float(t),400.,250.,20.) for t in np.arange(0,4.01,.1)]
        self.assertEqual(ThreatEstimator().assess(h,4,scale_history=h[:22])['reason'],'scale_stale')
