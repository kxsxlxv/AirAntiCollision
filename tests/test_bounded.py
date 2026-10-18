import unittest
import numpy as np
from airavoid.bounded import bounded_line,BoundedProjectiveEstimator
from airavoid.geometry import Config


class BoundedTests(unittest.TestCase):
    def test_known_slope_enclosed_for_arbitrary_bounded_errors(self):
        rng=np.random.default_rng(45);t=np.arange(0,4.01,.1)
        for _ in range(30):
            y=10+3*t+rng.uniform(-2,2,len(t))
            fit=bounded_line(t,y-2,y+2)
            self.assertIsNotNone(fit)
            self.assertLessEqual(fit['slope_low'],3+1e-9)
            self.assertGreaterEqual(fit['slope_high'],3-1e-9)

    def test_inconsistent_measurements_rejected(self):
        t=np.array([0.,1.,2.]);y=np.array([0.,0.,10.])
        self.assertIsNone(bounded_line(t,y-.1,y+.1))

    def test_true_plane_miss_enclosed(self):
        h=[]
        for t in np.arange(0,4.01,.1):
            z=25-t;x=2+(8-2)/25*t
            h.append((float(t),float(640+1500*x/z),360.,float(1500/z)))
        e=BoundedProjectiveEstimator(Config(bounded_pixel_error=.25,bounded_span_fraction=.01)).assess(h,4)
        self.assertLessEqual(e['miss_low'],8);self.assertGreaterEqual(e['miss_high'],8)
        self.assertEqual(e['state'],'nonthreat')

    def test_close_constant_bearing_can_warn_under_bounds(self):
        h=[(float(t),700.,360.,float(800/(10-t))) for t in np.arange(0,4.01,.1)]
        e=BoundedProjectiveEstimator().assess(h,4)
        self.assertEqual(e['state'],'threat')
