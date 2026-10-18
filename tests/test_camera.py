import unittest
import numpy as np
import cv2
from airavoid.tracking import CameraMotion,Tracker


class CameraTests(unittest.TestCase):
    def test_reacquisition_resets_reference(self):
        rng=np.random.default_rng(18)
        texture=rng.integers(0,256,(360,640,3),dtype=np.uint8)
        camera=CameraMotion('auto')
        camera.update(texture,0,[])
        transform,state=camera.update(texture,.1,[])
        self.assertTrue(state['valid']);self.assertTrue(state['evidence_reset'])
        transform,state=camera.update(texture,.2,[])
        self.assertFalse(state['evidence_reset'])
        blank=np.full_like(texture,128)
        transform,state=camera.update(blank,.3,[])
        self.assertFalse(state['valid']);self.assertTrue(state['evidence_reset'])
        camera.update(texture,.4,[])
        transform,state=camera.update(texture,.5,[])
        self.assertTrue(state['valid']);self.assertTrue(state['evidence_reset'])
        np.testing.assert_allclose(transform,np.eye(3),atol=1e-5)

    def test_reset_keeps_identity_but_discards_old_geometry(self):
        tracker=Tracker();tracks=tracker.update([[10,10,20,20,.9,0]],0,np.eye(3))
        ident=tracks[0].id
        tracker.reset_evidence()
        self.assertEqual(tracks[0].id,ident);self.assertEqual(len(tracks[0].history),0)
