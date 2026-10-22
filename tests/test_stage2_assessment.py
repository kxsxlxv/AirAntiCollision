import unittest
import numpy as np
from airavoid.assessment import (Assessor,Measurement,Context,relative_motion_observer,
                                 wcv_taumod,wcv_future,scope_result)
from airavoid.calibrated import CameraCalibration
from airavoid.detector_p2 import letterbox,DetectorP2


class Stage2AssessmentTests(unittest.TestCase):
    def test_airtrack_pair_time_is_at_current_later_sample(self):
        # Independent fronto-parallel projection: Z=20-t, half width=1.
        h=[]
        for t in np.arange(0,4.01,.1):
            s=100/(20-t)
            h.append(Measurement(float(t),(50-s,50-s,50+s,50+s)))
        e=Assessor('airtrack_cues').assess(h,Context(4,100,100,'fixed',True))
        self.assertAlmostEqual(e['published_pair_time_at_later_sample_s'],16.)
        self.assertAlmostEqual(e['tau_s'],16.)
        self.assertEqual(e['decision'],'uncertain')

    def test_lai_lin_factors_and_zero_motion_interval(self):
        h=[Measurement(0,(79,49,81,51)),Measurement(1,(68,48,72,52))]
        e=Assessor('lai_lin').assess(h,Context(1,100,100,'fixed',True))
        expansion=12/np.hypot(4,12)
        expected=.5*expansion+.333*.6+.167
        self.assertAlmostEqual(e['risk_index'],expected)
        self.assertEqual(e['decision'],'potential_conflict')
        h=[Measurement(0,(49,49,51,51)),Measurement(1,(49,49,51,51))]
        e=Assessor('lai_lin').assess(h,Context(1,100,100,'fixed',True))
        self.assertIsNone(e['bearing_factor'])
        self.assertEqual(e['risk_index_interval'],[.333,.5])

    def test_missing_measurement_is_never_safe(self):
        for method in ['lai_lin','bauer_disc','wcv_taumod','rv_c3bf','log']:
            e=Assessor(method).assess([],Context(4,100,100))
            self.assertEqual(e['decision'],'unobserved')
        h=[Measurement(0,(0,0,10,10)),Measurement(4,(0,0,20,20),measured=False,source='optical')]
        self.assertEqual(Assessor('lai_lin').assess(h,Context(4,100,100))['decision'],'unobserved')

    def test_future_prediction_is_not_an_expansion_measurement(self):
        a=Measurement(0,(1,1,10,10));b=Measurement(1,(1,1,100,100),measured=False,source='tracker_predict')
        self.assertEqual(Assessor('lai_lin').assess([a,b],Context(1,100,100))['decision'],'unobserved')

    def test_calibration_and_metric_state_cannot_be_guessed(self):
        h=[Measurement(float(t),(10,10,20,20)) for t in np.arange(0,4.01,.1)]
        c=Context(4,100,100,'fixed',True)
        self.assertEqual(Assessor('bauer_disc').assess(h,c)['reason_codes'],['missing_verified_intrinsics'])
        self.assertEqual(Assessor('wcv_taumod').assess(h,c)['reason_codes'],['missing_verified_metric_relative_state'])
        self.assertEqual(Assessor('rv_c3bf').assess(h,c)['reason_codes'],['missing_calibrated_derotated_target_feature_flow'])

    def test_flow_observer_recovers_direction_and_rejects_single_point(self):
        p=np.array([[-.1,-.05],[.1,-.05],[.1,.05],[-.1,.05]])
        xi=np.array([.2,.03,-.1]);flow=xi[:2]-p*xi[2]
        fit,details=relative_motion_observer(p,flow,alpha=1e-12)
        np.testing.assert_allclose(fit,xi,atol=1e-10)
        self.assertIsNone(relative_motion_observer(p[:1],flow[:1])[0])

    def test_nasa_predicate_distinguishes_crossing_and_vertical_separation(self):
        kw=dict(dthr=1.5,zthr=1.5,tthr=35.,tcoa=35.)
        self.assertTrue(wcv_taumod([0,0,20],[0,0,-1],**kw)[0])
        self.assertFalse(wcv_taumod([10,0,20],[0,0,-1],**kw)[0])
        self.assertFalse(wcv_taumod([0,10,20],[0,0,-1],**kw)[0])
        self.assertTrue(wcv_taumod([0,1,0],[0,0,0],**kw)[0])

    def test_scope_keeps_rejected_candidate_for_falsification(self):
        r={'decision':'apparent_nonconflict','reason_codes':['published_index']}
        e=scope_result(r,Context(1,100,100,'auto',True))
        self.assertEqual(e['raw_decision'],'apparent_nonconflict')
        self.assertEqual(e['decision'],'uncertain')
        self.assertEqual(r['decision'],'apparent_nonconflict')

    def test_future_well_clear_is_distinct_from_current_predicate(self):
        kw=dict(dthr=1.5,zthr=1.5,tthr=35.,tcoa=35.)
        r=[0,0,55];v=[0,0,-1]
        self.assertFalse(wcv_taumod(r,v,**kw)[0])
        self.assertTrue(wcv_future(r,v,60,**kw)[0])
        self.assertFalse(wcv_future(r,v,10,**kw)[0])

    def test_p2_letterbox_geometry_and_taxonomy(self):
        image=np.zeros((1080,1920,3),np.uint8);image[:,:,2]=255
        x,ratio,left,top=letterbox(image)
        self.assertEqual(x.shape,(1,3,800,1440))
        self.assertAlmostEqual(ratio,800/1080)
        self.assertEqual(left,9);self.assertEqual(top,0)
        self.assertEqual(float(x[0,0,0,left]),1.)
        parser=DetectorP2.__new__(DetectorP2);parser.conf=.03
        with self.assertRaises(ValueError):
            parser._decode(np.array([[0.,0.,10.,10.,.5,1.]]),1080,1920,ratio,left,top)


if __name__=='__main__':unittest.main()
