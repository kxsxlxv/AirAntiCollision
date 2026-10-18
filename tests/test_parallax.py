import unittest
import numpy as np
from airavoid.geometry import ThreatEstimator,Config,apply_camera_scope


class ParallaxTests(unittest.TestCase):
    def test_perfect_background_shift_can_create_false_pass_on_true_collision(self):
        raw=[(float(t),640.,360.,1000/(50*(1-t/20))) for t in np.arange(0,4.001,.1)]
        stabilized=[(t,u+20*t,v,s) for t,u,v,s in raw]
        estimator=ThreatEstimator(Config(correlation_seconds=1.))
        self.assertEqual(estimator.assess(raw,4)['state'],'threat')
        corrected=estimator.assess(stabilized,4)
        self.assertEqual(corrected['state'],'nonthreat')
        scoped=apply_camera_scope(corrected,{'mode':'auto','valid':True})
        self.assertEqual(scoped['state'],'uncertain');self.assertEqual(scoped['reason'],'parallax')
        self.assertEqual(scoped['image_rule_candidate_state'],'nonthreat')

    def test_declared_fixed_setup_and_potential_warnings_keep_their_scope(self):
        estimate={'state':'nonthreat','reason':'crossing'}
        self.assertEqual(apply_camera_scope(estimate,{'mode':'fixed'})['state'],'nonthreat')
        self.assertEqual(apply_camera_scope(estimate,{'mode':'auto'},True)['state'],'nonthreat')
        self.assertEqual(apply_camera_scope({'state':'threat','reason':'bearing_growth'},{'mode':'auto'})['state'],'threat')
