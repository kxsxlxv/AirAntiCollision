import unittest
import numpy as np
from airavoid.assessment import (Assessor,Measurement,Context,relative_motion_observer,
                                 wcv_taumod,wcv_future,scope_result,bauer_disc_coordinates)
from airavoid.calibrated import CameraCalibration
from airavoid.detector_p2 import letterbox,DetectorP2


class Stage2AssessmentTests(unittest.TestCase):
    @staticmethod
    def disc_ellipse_history(position,velocity):
        # Independent tangency of a horizontal circle (R=1) and a vertical
        # ellipse (depth diameter=1, height=.25), not assessor equations.
        history=[];f=1500.;center=20000.
        for time in np.arange(0,4.0001,.1):
            x,y,z=np.asarray(position)+time*np.asarray(velocity)
            horizontal=np.sort((x*z+np.array([-1,1])*.5*np.sqrt(x*x+z*z-.25))/(z*z-.25))
            ry=.125;rz=.5
            vertical=np.sort((y*z+np.array([-1,1])*np.sqrt(ry*ry*z*z+rz*rz*y*y-ry*ry*rz*rz))/(z*z-rz*rz))
            box=(center+f*horizontal[0],center+f*vertical[0],center+f*horizontal[1],center+f*vertical[1])
            history.append(Measurement(float(time),box))
        calibration=CameraCalibration(40000,40000,f,center,center,True,'independent analytic ellipse fixture')
        return history,Context(4,40000,40000,'fixed',True,calibration,True,constant_shape_verified=True)

    def test_bauer_virtual_disc_reconstruction_uses_three_axes_without_known_height(self):
        h,c=self.disc_ellipse_history([5,10,25],[-.2,-.4,-1])
        positions,horizontal,diagnostic=bauer_disc_coordinates(h,c.calibration)
        # Physical R is one metre in this fixture only; API receives no target R/H.
        expected=np.array([5,10,25])+np.array([-.2,-.4,-1])*4
        np.testing.assert_allclose(positions[-1],expected,atol=.012)
        self.assertEqual(horizontal.shape,(41,2))
        e=Assessor('bauer_disc').assess(h,c)
        self.assertAlmostEqual(e['tau_s'],21.,delta=.02)
        self.assertLess(e['miss_distance'],.02)
        self.assertTrue(e['model_3d_CPA_available'])
        self.assertFalse(e['true_3d_CPA_available'])
        self.assertFalse(e['vertical_CPA_in_height_units_available'])

    def test_bauer_vertical_velocity_changes_time_and_distance_of_minimum(self):
        h,c=self.disc_ellipse_history([0,10,25],[0,-.8,-1])
        e=Assessor('bauer_disc').assess(h,c)
        # Independently, squared range is (10-.8*t)^2+(25-t)^2,
        # derivative vanishes at absolute t=33/1.64, not at horizontal t=25.
        absolute_minimum=33/1.64
        expected_distance=np.hypot(10-.8*absolute_minimum,25-absolute_minimum)
        self.assertAlmostEqual(e['tau_s'],absolute_minimum-4,delta=.03)
        self.assertAlmostEqual(e['miss_distance'],expected_distance,delta=.03)
        self.assertAlmostEqual(e['horizontal_tau_s'],21.,delta=.01)
        self.assertLess(e['horizontal_miss_distance'],.01)
        self.assertEqual(e['decision'],'apparent_nonconflict')

    def test_bauer_constant_vertical_separation_does_not_divide_by_vertical_speed(self):
        h,c=self.disc_ellipse_history([0,12,25],[0,0,-1])
        e=Assessor('bauer_disc').assess(h,c)
        self.assertAlmostEqual(e['tau_s'],21.,delta=.03)
        self.assertAlmostEqual(e['miss_distance'],12.,delta=.03)
        self.assertEqual(e['decision'],'apparent_nonconflict')

    def test_bauer_unphysical_virtual_disc_horizon_is_not_safe(self):
        # A very close, very high target creates a virtual disc outside the
        # calibrated pinhole domain. It must return an explicit invalid result.
        k=CameraCalibration(100,100,10,50,50,True,'deliberate singular fixture')
        h=[Measurement(float(t),(30,190,70,210)) for t in np.arange(0,4.001,.1)]
        e=Assessor('bauer_disc').assess(h,Context(4,100,100,'fixed',True,k))
        self.assertEqual(e['decision'],'uncertain')
        self.assertEqual(e['reason_codes'],['invalid_virtual_disc_projection'])

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
