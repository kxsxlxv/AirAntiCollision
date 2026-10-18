import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
import cv2
from airavoid.geometry import advisory
from airavoid.process_export import ProcessFrameExporter


class ProcessExportTests(unittest.TestCase):
    def test_process_preserves_pts_count_and_runtime(self):
        with tempfile.TemporaryDirectory() as folder:
            exporter=ProcessFrameExporter(folder,12.5,'Тест порядка кадров')
            try:
                for i,t in enumerate([0.,.08,.28]):
                    exporter.submit(np.full((96,160,3),40+i*30,np.uint8),[],t,
                                    {'mode':'unknown','status':'unknown_camera','valid':False,'inliers':0},advisory([],160))
                exporter.finish(.36)
                self.assertEqual(exporter.metadata['cv2_path'],cv2.__file__)
                self.assertEqual(exporter.completed,3);self.assertFalse(exporter.process.is_alive())
                data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time:stream=duration','-of','json',str(Path(folder)/'annotated.mp4')],text=True))
                np.testing.assert_allclose([float(f['best_effort_timestamp_time']) for f in data['frames']],[0,.08,.28],atol=1/90000)
                self.assertAlmostEqual(float(data['streams'][0]['duration']),.36,places=4)
            finally:exporter.abort()

    def test_process_error_propagates_and_owned_child_exits(self):
        with tempfile.TemporaryDirectory() as folder:
            exporter=ProcessFrameExporter(folder,12.5,'Тест ошибки')
            try:
                exporter.submit(None,[],0,{},advisory([]))
                with self.assertRaisesRegex(RuntimeError,'NoneType'):exporter.finish(.1)
            finally:exporter.abort()
            self.assertFalse(exporter.process.is_alive())
