import unittest
import numpy as np
from airavoid.tracking import Tracker
from airavoid.geometry import ThreatEstimator


class ClippingTests(unittest.TestCase):
    def test_clipped_span_cannot_seed_future_geometry(self):
        tracker=Tracker(image_edge_margin=1);identity=np.eye(3)
        for i in range(25):tracker.update([[5,40,25,50,.8,0]],i*.1,identity,(1280,720))
        tr=tracker.tracks[0];self.assertGreater(len(tr.history),20)
        tracker.update([[0,40,20,50,.8,0]],2.5,identity,(1280,720))
        self.assertTrue(tr.clipped);self.assertEqual(len(tr.history),0);self.assertEqual(len(tr.scale_history),0)
        tracker.update([[5,40,25,50,.8,0]],2.6,identity,(1280,720))
        self.assertFalse(tr.clipped);self.assertEqual(len(tr.history),1)

    def test_history_duration_counts_only_valid_post_clipping_measurements(self):
        tracker=Tracker(image_edge_margin=1);identity=np.eye(3);estimator=ThreatEstimator()
        tracker.update([[0,100,20,110,.8,0]],0.,identity,(1280,720))
        for i in range(1,51):tr=tracker.update([[3,100,23,110,.8,0]],i*.04,identity,(1280,720))[0]
        self.assertEqual(estimator.assess(tr.history,2.)['reason'],'history')
        tr=tracker.update([[3,100,23,110,.8,0]],2.04,identity,(1280,720))[0]
        self.assertNotEqual(estimator.assess(tr.history,2.04)['reason'],'history')

    def test_right_bottom_and_sot_outside_source_abstain(self):
        for box in [[1270,100,1280,110,.8,0],[100,710,110,720,.8,0],[-5,100,15,120,.8,0,'sot']]:
            tracker=Tracker(image_edge_margin=1)
            tr=tracker.update([box],0,np.eye(3),(1280,720))[0]
            self.assertTrue(tr.clipped)
