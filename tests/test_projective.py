import unittest
import numpy as np
from airavoid.projective import ProjectiveEstimator


class ProjectiveTests(unittest.TestCase):
    def projected(self,miss=0.,focal=800):
        # W=1, Z0=20, Vz=-1, X0=2, Vx=(miss-X0)/20.
        h=[]
        for t in np.arange(0,4.001,.1):
            z=20-t;x=2+(miss-2)/20*t
            h.append((float(t),float(640+focal*x/z),360.,float(focal/z)))
        return h

    def test_exact_plane_intercept_and_time(self):
        e=ProjectiveEstimator().assess(self.projected(miss=8),4)
        self.assertAlmostEqual(e['miss_spans'],8,places=6)
        self.assertAlmostEqual(e['tau'],16,places=6)
        self.assertEqual(e['state'],'nonthreat')

    def test_constant_bearing_risk(self):
        e=ProjectiveEstimator().assess(self.projected(),4)
        self.assertEqual(e['state'],'threat')

    def test_focal_length_does_not_change_exact_miss(self):
        a=ProjectiveEstimator().assess(self.projected(miss=8,focal=500),4)
        b=ProjectiveEstimator().assess(self.projected(miss=8,focal=1200),4)
        self.assertAlmostEqual(a['miss_spans'],b['miss_spans'],places=6)

    def test_unknown_camera_does_not_become_observable(self):
        self.assertEqual(ProjectiveEstimator().assess(self.projected(),4,False)['state'],'uncertain')
