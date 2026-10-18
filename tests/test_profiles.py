import sys
import unittest
from unittest.mock import patch
from airavoid.cli import parse_args


class ProfileTests(unittest.TestCase):
    def args(self,*extra):
        with patch.object(sys,'argv',['airavoid','input.mp4','--weights','model.onnx',*extra]):return parse_args()

    def test_conservative_defaults_are_explicit_and_onnx_does_not_use_cuda_preprocessing(self):
        a=self.args()
        self.assertEqual(a.conf,.03);self.assertTrue(a.tiled);self.assertFalse(a.gpu_preprocess)
        self.assertEqual((a.correlation_seconds,a.image_edge_margin,a.tile_edge_margin),(1.,1.,2.))

    def test_reference_and_explicit_overrides(self):
        a=self.args('--profile','reference')
        self.assertEqual(a.conf,.18);self.assertFalse(a.tiled)
        self.assertEqual((a.correlation_seconds,a.image_edge_margin,a.tile_edge_margin),(0.,0.,0.))
        a=self.args('--conf','.12','--full-frame','--cpu-preprocess','--image-edge-margin','3')
        self.assertEqual(a.conf,.12);self.assertFalse(a.tiled);self.assertEqual(a.image_edge_margin,3.)

    def test_other_geometry_does_not_silently_enable_log_covariance(self):
        a=self.args('--geometry','projective');self.assertEqual(a.correlation_seconds,0.)
