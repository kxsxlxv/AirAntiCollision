import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
import numpy as np
from airavoid.export import FrameExporter
from airavoid.media import TimestampWriter


class ExportTests(unittest.TestCase):
    def test_async_preserves_order_pts_and_metadata_snapshots(self):
        gate=threading.Event();seen=[]
        class Render:
            def render(self,frame,objects,t,camera,advice):
                gate.wait(timeout=2);seen.append((t,len(objects[0]['raw_history'])))
                return frame
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'async.mp4';writer=TimestampWriter(path,12.5,(160,96))
            exporter=FrameExporter(Render(),folder,writer=writer,asynchronous=True)
            objects=[{'raw_history':[(0,2,3)]}]
            exporter.submit(np.full((96,160,3),40,np.uint8),objects,0,{},{});objects[0].pop('raw_history')
            gate.set()
            for i,t in enumerate([.08,.28]):exporter.submit(np.full((96,160,3),70+i*30,np.uint8),[{'raw_history':[(t,2,3)]}],t,{},{})
            exporter.finish(.36)
            self.assertEqual(seen,[(0,1),(.08,1),(.28,1)])
            self.assertEqual(len(exporter.stages['render_ms']),3)
            data=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time:stream=duration','-of','json',str(path)],text=True))
            np.testing.assert_allclose([float(f['best_effort_timestamp_time']) for f in data['frames']],[0,.08,.28],atol=1/90000)
            self.assertAlmostEqual(float(data['streams'][0]['duration']),.36,places=4)

    def test_worker_error_propagates_instead_of_silent_success(self):
        class BadRender:
            def render(self,*args):raise ValueError('intentional render error')
        with tempfile.TemporaryDirectory() as folder:
            exporter=FrameExporter(BadRender(),folder,asynchronous=True)
            exporter.submit(None,[],0,{},{})
            with self.assertRaises(ValueError):exporter.finish(.1)
            exporter.abort()
