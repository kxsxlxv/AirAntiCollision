import copy
import unittest
from airavoid.demo_range import NPSRangeDemo


class RangeDemoTests(unittest.TestCase):
    def sample(self,span=20):
        obj={'id':2,'box':[640-span/2,470,640+span/2,480],
             'demo_observation':{'measured':True,'clipped':False,'ambiguous':False}}
        return {'targets':[{'id':2,'state':'MONITORING'}]},[obj]

    def test_declared_size_scales_range_and_larger_image_is_closer(self):
        def estimate(size,span):
            d,o=self.sample(span);NPSRangeDemo(size).update(d,o,0,(1280,960));return d['targets'][0]['range_approx_m']
        self.assertAlmostEqual(estimate(3,20),2*estimate(1.5,20))
        self.assertAlmostEqual(estimate(1.5,20)/estimate(1.5,40),2,places=2)

    def test_predictions_and_ambiguous_extents_have_no_range(self):
        for bad in ['measured','ambiguous','clipped']:
            d,o=self.sample();o[0]['demo_observation'][bad]=bad!='measured'
            NPSRangeDemo().update(d,o,0,(1280,960));self.assertIsNone(d['targets'][0]['range_approx_m'])

    def test_range_does_not_change_cue_state_or_direction(self):
        d,o=self.sample();d['direction']={'status':'unresolved','vector':None};before=copy.deepcopy(d)
        NPSRangeDemo().update(d,o,0,(1280,960))
        self.assertEqual(d['direction'],before['direction']);self.assertEqual(d['targets'][0]['state'],'MONITORING')
        self.assertFalse(d['range_preset']['calibrated']);self.assertFalse(d['range_preset']['used_for_collision_or_direction'])

    def test_invalid_assumptions_rejected(self):
        for span,fov in [(0,120),(1.5,180),(float('nan'),120)]:
            with self.assertRaises(ValueError):NPSRangeDemo(span,fov)
