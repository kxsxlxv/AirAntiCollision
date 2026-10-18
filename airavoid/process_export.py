"""Bounded render process with explicit acknowledgments and owned-child cleanup."""
from collections import defaultdict
import multiprocessing as mp
from pathlib import Path
import queue
import time
import traceback


def _worker(jobs,results,output,fps,title,calibration,encoder):
    writer=None
    try:
        import av
        import cv2
        from .render import Renderer
        from .media import TimestampWriter
        renderer=Renderer(title,calibration=calibration)
        writer=TimestampWriter(Path(output)/'annotated.mp4',fps,(renderer.width,renderer.height),encoder=encoder)
        results.put({'kind':'ready','metadata':{'cv2':cv2.__version__,'cv2_path':cv2.__file__,
                     'cv2_threads':cv2.getNumThreads(),'av':av.__version__,'encoder':encoder}})
        index=0
        while True:
            job=jobs.get()
            if job['kind']=='close':
                writer.close(job['end_time']);results.put({'kind':'closed','frames':index});break
            if job['index']!=index:raise RuntimeError('Render process received a frame out of order')
            tick=time.perf_counter()
            rendered=renderer.render(job['frame'],job['objects'],job['time'],job['camera'],job['advice'])
            if job['preview']:cv2.imwrite(str(Path(output)/'preview.png'),rendered)
            render_ms=(time.perf_counter()-tick)*1000;tick=time.perf_counter()
            writer.write(rendered,job['time'])
            results.put({'kind':'frame','index':index,'render_ms':render_ms,'encode_ms':(time.perf_counter()-tick)*1000})
            index+=1
    except BaseException as e:
        results.put({'kind':'error','message':type(e).__name__+': '+str(e),'traceback':traceback.format_exc()})
    finally:
        if writer is not None:
            try:writer.abort()
            except Exception:pass


class ProcessFrameExporter:
    def __init__(self,output,fps,title,calibration=None,encoder='libx264'):
        context=mp.get_context('spawn');self.jobs=context.Queue(maxsize=2);self.results=context.Queue()
        self.process=context.Process(target=_worker,args=(self.jobs,self.results,str(output),fps,title,calibration,encoder),
                                     daemon=True,name='airavoid-render')
        self.pending=0;self.submitted=0;self.completed=0;self.max_pending=2;self.closed=False
        self.stages=defaultdict(list);self.metadata={};tick=time.perf_counter()
        try:
            self.process.start();message=self._receive()
            if message['kind']!='ready':raise RuntimeError('Render process did not initialize correctly')
            self.metadata={**message['metadata'],'startup_s':time.perf_counter()-tick}
        except BaseException:
            self.abort();raise

    def _receive(self):
        deadline=time.monotonic()+30
        while True:
            try:
                message=self.results.get(timeout=.2)
                if message['kind']=='error':raise RuntimeError('Render process failed: '+message['message']+'\n'+message['traceback'])
                return message
            except queue.Empty:
                if not self.process.is_alive():raise RuntimeError('Render process exited before acknowledgment, code '+str(self.process.exitcode))
                if time.monotonic()>deadline:raise TimeoutError('Render process did not acknowledge work within30 seconds')

    def _collect(self):
        result=self._receive()
        if result['kind']!='frame' or result['index']!=self.completed:raise RuntimeError('Render acknowledgment order mismatch')
        self.stages['render_ms'].append(result['render_ms']);self.stages['encode_write_ms'].append(result['encode_ms'])
        self.pending-=1;self.completed+=1

    def submit(self,frame,objects,t,camera,advice,preview=False):
        tick=time.perf_counter()
        while self.pending>=self.max_pending:self._collect()
        self.stages['render_queue_wait_ms'].append((time.perf_counter()-tick)*1000)
        self.jobs.put({'kind':'frame','index':self.submitted,'frame':frame,'objects':[o.copy() for o in objects],
                       'time':t,'camera':camera,'advice':advice,'preview':preview})
        self.pending+=1;self.submitted+=1

    def finish(self,end_time):
        while self.pending:self._collect()
        self.jobs.put({'kind':'close','end_time':end_time});result=self._receive()
        if result['kind']!='closed' or result['frames']!=self.submitted:raise RuntimeError('Render process frame-count mismatch')
        self.process.join(timeout=5)
        if self.process.is_alive():raise RuntimeError('Render process did not exit after closing its video')
        if self.process.exitcode:raise RuntimeError('Render process exited with code '+str(self.process.exitcode))
        self._close_queues();self.closed=True

    def _close_queues(self):
        for channel in [self.jobs,self.results]:
            channel.cancel_join_thread();channel.close()

    def abort(self):
        if self.closed:return
        if self.process.pid is not None and self.process.is_alive():
            self.process.terminate();self.process.join(timeout=5)
            if self.process.is_alive():self.process.kill();self.process.join(timeout=5)
        self._close_queues();self.closed=True
