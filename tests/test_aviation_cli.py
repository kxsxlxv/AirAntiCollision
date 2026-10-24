"""Showcase layout selection must not change the scientific/default pipeline."""
import contextlib
import io
import unittest
from airavoid.cli import parse_args


class AviationCLITests(unittest.TestCase):
    def args(self,*extra):
        with contextlib.redirect_stderr(io.StringIO()):
            return parse_args(['input.mov',*extra])

    def test_scientific_default_remains_detector_A_and_original_geometry(self):
        a=self.args()
        self.assertFalse(a.demo_showcase);self.assertEqual(a.detector_variant,'A')
        self.assertEqual(a.geometry,'log');self.assertIsNone(a.assessment)

    def test_new_showcase_and_legacy_showcase_are_explicitly_selectable(self):
        a=self.args('--demo-showcase','--detector-variant','B','--nps-range-demo')
        self.assertEqual(a.demo_layout,'aviation');self.assertTrue(a.nps_range_demo)
        self.assertEqual(self.args('--demo-showcase','--demo-layout','classic').demo_layout,'classic')

    def test_unknown_layout_is_rejected(self):
        with self.assertRaises(SystemExit):self.args('--demo-showcase','--demo-layout','unknown')

    def test_render_process_cannot_silently_use_another_renderer(self):
        with self.assertRaises(SystemExit):self.args('--demo-showcase','--render-process')
