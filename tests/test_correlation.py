import unittest
import numpy as np
from airavoid.geometry import robust_line


class CorrelationTests(unittest.TestCase):
    def test_covariance_correction_never_narrows_or_changes_fit(self):
        rng=np.random.default_rng(11);times=np.arange(0,4.01,.08)
        noise=np.convolve(rng.normal(size=len(times)+10),np.ones(11)/3,'valid')
        values=200+3*times+noise
        iid=robust_line(times,values,1.)
        corrected=robust_line(times,values,1.,.5)
        self.assertAlmostEqual(iid[0],corrected[0]);self.assertAlmostEqual(iid[1],corrected[1])
        self.assertGreaterEqual(corrected[2],iid[2])
