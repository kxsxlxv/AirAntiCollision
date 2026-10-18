import unittest
from airavoid.geometry import advisory


class AdvisoryTests(unittest.TestCase):
    def objects(self,scale=1.):
        return [{'id':i,'assessment':{'state':state,'reason':reason,'drift':vx*scale,'velocity':[vx*scale,0.],
                                    'span':10*scale,'camera_valid':True}} for i,state,reason,vx in
                [(1,'threat','bearing_growth',0.),(2,'nonthreat','crossing',30.)]]

    def test_common_flow_can_worsen_another_target(self):
        a=advisory(self.objects())
        self.assertLess(a['scores']['right'],0)
        self.assertEqual(a['bottleneck_track_ids']['right'],2)
        self.assertGreaterEqual(a['scores']['left'],0)
        self.assertIsNone(a['recommendation'])

    def test_same_view_resized_keeps_comparison(self):
        a=advisory(self.objects(),1280);b=advisory(self.objects(1.5),1920)
        for key in a['scores']:self.assertAlmostEqual(a['scores'][key],b['scores'][key])

    def test_bad_extent_is_unresolved_even_if_confident(self):
        objects=self.objects();objects[1]['assessment']['reason']='aspect'
        a=advisory(objects)
        self.assertEqual(a['unresolved_targets'],1);self.assertEqual(a['evaluated_targets'],1)
