import unittest
from scripts.camera_parallax_probe import run_probe


class CameraParallaxIntegrationTests(unittest.TestCase):
    def test_good_actual_background_fit_cannot_authorize_pass(self):
        r=run_probe()
        self.assertGreaterEqual(r['valid_background_fits'],50)
        self.assertEqual(r['raw_unstabilized_final_state'],'threat')
        self.assertGreater(r['old_confirmed_states'].get('nonthreat',0),0)
        self.assertEqual(r['guarded_confirmed_states'].get('nonthreat',0),0)
