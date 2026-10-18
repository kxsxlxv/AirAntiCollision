import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from airavoid.proposals import SkyProposals
from airavoid.provenance import file_sha256
from airavoid.media import TimestampWriter
from airavoid.lifecycle import RunSession
from scripts.run_sky_demo import require_cfr
from airavoid.cli import parse_args


class SkyAlignmentTests(unittest.TestCase):
    def test_zoom_crop_boundary_uses_same_truncation_guard_as_native_tiles(self):
        class DetectorStub:
            conf=.03;tile_edge_margin=2.
            def _infer(self,crop):return np.array([[0,20,10,30,.9,0],[40,40,60,50,.8,0]],float)
        image=np.zeros((96,128,3),np.uint8);image[45:48,60:63]=255
        proposals=SkyProposals(None,25,all_region=True)
        result=proposals.augment(DetectorStub(),image,[],0,0)
        self.assertEqual(len(result),1);self.assertAlmostEqual(result[0][4],.8)
        self.assertEqual(result[0][6],'zoom');proposals.close()

    def test_mask_free_zoom_matches_explicit_all_region_control(self):
        with tempfile.TemporaryDirectory() as folder:
            mask=Path(folder)/'mask.mkv'
            writer=cv2.VideoWriter(str(mask),cv2.VideoWriter_fourcc(*'FFV1'),25,(128,96))
            writer.write(np.full((96,128,3),255,np.uint8));writer.release()
            image=np.zeros((96,128,3),np.uint8);image[45:48,60:63]=255
            class CropRecorder:
                conf=.03
                def __init__(self):self.crops=[]
                def _infer(self,crop):self.crops.append(crop.copy());return []
            left=SkyProposals(None,25,all_region=True);right=SkyProposals(mask,25,all_region=True)
            a=CropRecorder();b=CropRecorder()
            self.assertEqual(left.augment(a,image,[],999,0),right.augment(b,image,[],0,0))
            self.assertTrue(a.crops);self.assertEqual(left.last_proposals,right.last_proposals)
            for x,y in zip(a.crops,b.crops):np.testing.assert_array_equal(x,y)
            self.assertEqual(left.provenance['status'],'no_sky_model');left.close();right.close()
            self.assertTrue(parse_args(['input.mp4','--zoom-proposals']).zoom_proposals)

    def test_same_fps_wrong_source_or_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            mask=Path(folder)/'mask.mkv'
            writer=cv2.VideoWriter(str(mask),cv2.VideoWriter_fourcc(*'FFV1'),25,(64,48))
            for _ in range(3):writer.write(np.full((48,64,3),255,np.uint8))
            writer.release()
            path=Path(str(mask)+'.json');manifest={'input_sha256':'right','mask_sha256':file_sha256(mask)}
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'source SHA'):
                SkyProposals(mask,25,source_hash='wrong',source_frames=3)
            with self.assertRaisesRegex(ValueError,'frame count'):
                SkyProposals(mask,25,source_hash='right',source_frames=4)
            source=SkyProposals(mask,25,source_hash='right',source_frames=3)
            self.assertEqual(source.provenance['status'],'sha256_verified');source.close()
            manifest['mask_sha256']='tampered';path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'mask SHA'):
                SkyProposals(mask,25,source_hash='right',source_frames=3)

    def test_sky_preflight_accepts_cfr_and_rejects_vfr_before_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            for name,times,end in [('cfr',[0,.08,.16],.24),('vfr',[0,.08,.28],.36)]:
                path=Path(folder)/(name+'.mp4');writer=TimestampWriter(path,12.5,(160,96))
                for t in times:writer.write(np.zeros((96,160,3),np.uint8),t)
                writer.close(end)
                if name=='cfr':self.assertEqual(require_cfr(path),(12.5,3))
                else:
                    with self.assertRaisesRegex(ValueError,'requires CFR'):require_cfr(path)

    def test_failed_workflow_cannot_leave_successful_old_timing(self):
        with tempfile.TemporaryDirectory() as folder:
            report=Path(folder)/'workflow_benchmark.json';report.write_text('{}')
            with self.assertRaises(ValueError):
                with RunSession(folder,benchmark_name=report.name):raise ValueError('segmentation failed')
            self.assertFalse(report.exists())
            self.assertEqual(json.loads(Path(folder,'run_status.json').read_text())['state'],'failed')
