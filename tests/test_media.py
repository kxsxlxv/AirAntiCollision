import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import cv2
from airavoid.media import SourceClock,TimestampWriter,probe_video,presentation_end


class MediaTests(unittest.TestCase):
    def test_nonzero_stream_start_normalizes_duration_and_decoded_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'zero.mp4';shifted=Path(folder)/'offset.mp4'
            w=TimestampWriter(source,12.5,(160,96))
            for i,t in enumerate([0.,.08,.28]):w.write(np.full((96,160,3),40+i*30,np.uint8),t)
            w.close(.36)
            subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-copyts','-itsoffset','5','-i',str(source),'-c','copy',str(shifted)],check=True)
            meta=probe_video(shifted);end,method=presentation_end(shifted,meta)
            self.assertGreater(float(meta['start_time']),4.9);self.assertAlmostEqual(end,.36,places=4)
            self.assertIn('first_decoded_frame',method)
            cap=cv2.VideoCapture(str(shifted));clock=SourceClock(cap.get(cv2.CAP_PROP_FPS));times=[]
            for i in range(3):
                ok,_=cap.read();self.assertTrue(ok);times.append(clock.update(i,cap.get(cv2.CAP_PROP_POS_MSEC)))
            cap.release();np.testing.assert_allclose(times,[0,.08,.28],atol=1e-6)

    @unittest.skipUnless(os.environ.get('AIRAVOID_TEST_NVENC')=='1','Optional NVIDIA hardware integration test')
    def test_nvenc_preserves_variable_pts_and_duration(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'nvenc.mp4';w=TimestampWriter(path,12.5,(160,96),encoder='h264_nvenc')
            for i,t in enumerate([0.,.08,.28]):w.write(np.full((96,160,3),40+i*30,np.uint8),t)
            w.close(.36)
            data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time:stream=duration,codec_name','-of','json',str(path)],text=True))
            self.assertEqual(data['streams'][0]['codec_name'],'h264')
            np.testing.assert_allclose([float(f['best_effort_timestamp_time']) for f in data['frames']],[0,.08,.28],atol=1/90000)
            self.assertAlmostEqual(float(data['streams'][0]['duration']),.36,places=4)

    def test_variable_source_intervals_are_not_flattened(self):
        c=SourceClock(25.)
        self.assertEqual(c.update(0,5000.),0.)
        self.assertAlmostEqual(c.update(1,5080.),.08)
        self.assertAlmostEqual(c.update(2,5280.),.28)

    def test_missing_pts_fallback_is_explicit(self):
        c=SourceClock(25.);c.update(0,0.)
        self.assertEqual(c.update(1,0.),.04)
        self.assertEqual(c.mode,'frame_rate_fallback')

    def test_nonmonotonic_timestamp_rejected(self):
        c=SourceClock(25.);c.update(0,0.);c.update(1,40.)
        with self.assertRaises(ValueError):c.update(2,20.)

    def test_h264_preserves_variable_pts_and_final_duration(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'vfr.mp4';w=TimestampWriter(path,12.5,(160,96))
            for i,t in enumerate([0.,.08,.28]):w.write(np.full((96,160,3),40+i*30,np.uint8),t)
            w.close(.36)
            cmd=['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time,pkt_duration_time:stream=duration,codec_name','-of','json',str(path)]
            data=json.loads(subprocess.check_output(cmd,text=True))
            self.assertEqual(data['streams'][0]['codec_name'],'h264')
            pts=[float(f['best_effort_timestamp_time']) for f in data['frames']]
            np.testing.assert_allclose(pts,[0,.08,.28],atol=1/90000)
            self.assertAlmostEqual(float(data['streams'][0]['duration']),.36,places=4)
