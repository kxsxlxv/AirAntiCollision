import unittest
import numpy as np
from airavoid.calibrated import CameraCalibration,CalibratedCPAEstimator


class CalibratedTests(unittest.TestCase):
    def setUp(self):
        self.k=CameraCalibration(40000,720,1500.,20000.,360.,True,'Analytical test only')
        self.est=CalibratedCPAEstimator(self.k)

    def history(self,r,v):
        rows=[]
        for t in np.arange(0,4.001,.1):
            x,y,z=np.array(r)+np.array(v)*t
            rows.append((float(t),20000+1500*x/z,360+1500*y/z,1500/z))
        return rows

    def test_wide_bearing_counterexample_recovers_true_cpa(self):
        z=25.;x=z*np.tan(np.deg2rad(85));r=np.array([x,0,z]);v=np.array([(8-x)/z,0,-1.])
        e=self.est.assess(self.history(r,v),4)
        tau=-r@v/(v@v);miss=np.linalg.norm(r+tau*v)
        self.assertAlmostEqual(e['miss_spans'],miss,places=8)
        self.assertAlmostEqual(e['tau'],tau-4,places=8)
        self.assertNotEqual(e['state'],'nonthreat')

    def test_calibrated_constant_bearing_close_warns(self):
        e=self.est.assess(self.history([8.,2.,25.],[-.32,-.08,-1.]),4)
        self.assertEqual(e['state'],'threat')

    def test_calibrated_far_crossing(self):
        e=self.est.assess(self.history([0.,0.,25.],[1.,0.,-1.]),4)
        self.assertEqual(e['state'],'nonthreat')

    def test_camera_or_stale_history_abstains(self):
        h=self.history([0,0,25],[0,0,-1])
        self.assertEqual(self.est.assess(h,4,False)['reason'],'camera')
        self.assertEqual(self.est.assess(h,5)['reason'],'lost')

    def test_past_cpa_does_not_clear_shrinking_target(self):
        e=self.est.assess(self.history([0.,0.,25.],[0.,0.,1.]),4)
        self.assertEqual(e['state'],'uncertain');self.assertEqual(e['reason'],'cpa_past')

    def test_invalid_calibration_rejected(self):
        with self.assertRaises(ValueError):self.k.validate_resolution(1280,720)
        with self.assertRaises(ValueError):CameraCalibration(1280,720,float('nan'),640,360,True,'test')
        with self.assertRaises(ValueError):CameraCalibration(1280,720,700,640,360,False,'test')
