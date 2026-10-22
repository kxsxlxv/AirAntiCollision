import contextlib
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from airavoid.cli import parse_args
from airavoid.assessment_runtime import AssessmentRuntime,validate_detection_classes,detector_contract,scope_for_runtime
from airavoid.detector_p2 import DetectorP2,letterbox


class RuntimeTests(unittest.TestCase):
    def args(self,*extra):
        with contextlib.redirect_stderr(io.StringIO()):
            return parse_args(['input.mov',*extra])

    def test_five_class_default_and_optin_p2_are_distinct(self):
        a=self.args('--weights','five.onnx')
        self.assertEqual(a.detector_variant,'A');self.assertTrue(a.tiled)
        b=self.args('--detector-variant','B')
        self.assertFalse(b.tiled);self.assertFalse(b.gpu_preprocess)
        self.assertIn('yolo26m_p2_local_fp32.engine',b.weights)

    def test_p2_rejects_incompatible_preprocessing_and_augmentations(self):
        for option in ['--gpu-preprocess','--tiled','--adaptive-tiles','--zoom-proposals','--appearance-bridge']:
            with self.subTest(option=option),self.assertRaises(SystemExit):
                self.args('--detector-variant','B',option)

    def test_published_method_requires_explicit_selection_and_no_log_hac(self):
        a=self.args('--assessment','lai_lin')
        self.assertEqual(a.correlation_seconds,0.)
        self.assertIsNone(self.args().assessment)
        with self.assertRaises(SystemExit):self.args('--assessment','lai_lin','--geometry','projective')

    def test_taxonomy_validation_rejects_wrong_or_fractional_classes(self):
        validate_detection_classes([[1,2,3,4,.9,4]],'A')
        for bad in [1,.2,np.nan]:
            with self.subTest(class_id=bad),self.assertRaises(ValueError):
                validate_detection_classes([[1,2,3,4,.9,bad]],'B')

    def test_runtime_validates_actual_tensor_shape(self):
        detector=SimpleNamespace(backend='tensorrt',input_name='x',output_name='y',
            buffers={'x':np.zeros((1,3,416,704)),'y':np.zeros((1,300,6))})
        with self.assertRaisesRegex(ValueError,'contract mismatch'):detector_contract(detector,'B')

    def test_p2_letterbox_preserves_center_and_pad_for_portraitish_video(self):
        image=np.full((960,1280,3),255,np.uint8)
        x,ratio,left,top=letterbox(image)
        self.assertEqual(x.shape,(1,3,800,1440));self.assertAlmostEqual(ratio,800/960)
        self.assertEqual((left,top),(186,0));self.assertAlmostEqual(float(x[0,0,0,0]),114/255,places=6)
        self.assertEqual(float(x[0,0,400,720]),1.)

    def test_p2_parser_cannot_silently_accept_five_class_export(self):
        detector=DetectorP2.__new__(DetectorP2);detector.conf=.03
        with self.assertRaisesRegex(ValueError,'class zero'):
            detector._decode(np.array([[1,2,10,20,.9,2]],float),100,100,1,0,0)

    def test_sot_prediction_is_not_independent_expansion(self):
        runtime=AssessmentRuntime('lai_lin');camera={'mode':'auto','valid':True}
        tr=SimpleNamespace(id=1,last=0.,box=[20,20,40,40],confidence=.7,source='detector',clipped=False,ambiguous_until=0.)
        runtime.assess(tr,0.,camera,(100,100))
        tr.last=1.;tr.box=[20,20,80,80];tr.source='sot'
        e=runtime.assess(tr,1.,camera,(100,100))
        self.assertEqual(e['research']['decision'],'unobserved')
        self.assertEqual(e['research']['evidence_age_s'],1.)
        self.assertIsNone(e['growth']);self.assertIsNone(e['velocity'])

    def test_research_nonconflict_is_scoped_on_moving_camera(self):
        runtime=AssessmentRuntime('lai_lin');camera={'mode':'auto','valid':True}
        tr=SimpleNamespace(id=1,last=0.,box=[1,1,11,11],confidence=.7,source='detector',clipped=False,ambiguous_until=0.)
        runtime.assess(tr,0.,camera,(100,100))
        tr.last=.1;tr.box=[2,1,12,11]
        e=runtime.assess(tr,.1,camera,(100,100))
        self.assertEqual(e['state'],'uncertain')
        self.assertEqual(e['research']['raw_decision'],'apparent_nonconflict')
        self.assertIn('unresolved_camera_rotation_translation',e['research']['reason_codes'])
        self.assertIsNone(e['miss_spans'])

    def test_lai_initial_area_anchor_survives_bounded_recent_history(self):
        runtime=AssessmentRuntime('lai_lin');camera={'mode':'fixed','valid':True}
        tr=SimpleNamespace(id=1,last=0.,box=[1,1,11,11],confidence=.7,source='detector',clipped=False,ambiguous_until=0.)
        for i in range(620):
            tr.last=i/25;tr.box=[1,1,11+i/100,11+i/100]
            result=runtime.assess(tr,tr.last,camera,(100,100))['research']
        self.assertEqual(len(runtime.histories[1]),500)
        self.assertEqual(result['initial_area_time_s'],0.)
        expected_change=(10+619/100)**2-100
        self.assertAlmostEqual(result['area_factor'],expected_change/np.hypot(100,expected_change))

    def test_low_lai_index_is_not_physical_clearance_even_with_fixed_camera(self):
        runtime=AssessmentRuntime('lai_lin');camera={'mode':'fixed','valid':True}
        tr=SimpleNamespace(id=1,last=0.,box=[1,1,11,11],confidence=.7,source='detector',clipped=False,ambiguous_until=0.)
        runtime.assess(tr,0.,camera,(100,100))
        tr.last=.1;tr.box=[2,1,12,11]
        result=runtime.assess(tr,.1,camera,(100,100))['research']
        self.assertEqual(result['raw_decision'],'apparent_nonconflict')
        self.assertEqual(result['decision'],'uncertain')
        self.assertIn('low_index_is_not_clearance_evidence',result['reason_codes'])

    def test_nominal_point_observer_clearance_needs_aircraft_footprint(self):
        raw={'method_id':'rv_c3bf','decision':'apparent_nonconflict',
            'raw_decision':'apparent_nonconflict','reason_codes':['nominal_cone']}
        scoped=scope_for_runtime(raw)
        self.assertEqual(raw['decision'],'apparent_nonconflict')
        self.assertEqual(scoped['decision'],'uncertain')
        self.assertIn('point_cone_without_finite_aircraft_footprint_is_not_clearance',scoped['reason_codes'])

    def test_clipped_measurement_cannot_seed_new_area_anchor(self):
        runtime=AssessmentRuntime('lai_lin');camera={'mode':'fixed','valid':True}
        tr=SimpleNamespace(id=1,last=0.,box=[1,1,11,11],confidence=.7,source='detector',clipped=True,ambiguous_until=0.)
        runtime.assess(tr,0.,camera,(100,100))
        tr.clipped=False
        for i in [1,2]:
            tr.last=i/25;tr.box=[2,2,22,22]
            result=runtime.assess(tr,tr.last,camera,(100,100))['research']
        self.assertEqual(result['initial_area_time_s'],1/25)
        self.assertEqual(result['area_factor'],0.)


if __name__=='__main__':unittest.main()
