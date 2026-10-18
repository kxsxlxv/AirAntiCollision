"""Bounded, ordered render/export worker; geometry remains sequential and causal."""
from collections import defaultdict,deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import cv2


class FrameExporter:
    def __init__(self,renderer,output,writer=None,process=None,asynchronous=False):
        self.renderer=renderer;self.output=Path(output);self.writer=writer;self.process=process
        self.executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='airavoid-render') if asynchronous else None
        self.pending=deque();self.stages=defaultdict(list);self.closed=False;self.max_pending=2

    def _render(self,frame,objects,t,camera,advice,preview):
        tick=time.perf_counter();rendered=self.renderer.render(frame,objects,t,camera,advice)
        if preview:cv2.imwrite(str(self.output/'preview.png'),rendered)
        render_ms=(time.perf_counter()-tick)*1000;tick=time.perf_counter()
        if self.writer:self.writer.write(rendered,t)
        else:self.process.stdin.write(rendered.tobytes())
        return render_ms,(time.perf_counter()-tick)*1000

    def _collect(self,result):
        self.stages['render_ms'].append(result[0]);self.stages['encode_write_ms'].append(result[1])

    def submit(self,frame,objects,t,camera,advice,preview=False):
        # main() removes raw_history after submission; copy the outer dictionaries.
        # Arrays/history lists are immutable snapshots for the lifetime of this job.
        objects=[o.copy() for o in objects]
        if self.executor:
            tick=time.perf_counter()
            while len(self.pending)>=self.max_pending:self._collect(self.pending.popleft().result())
            self.stages['render_queue_wait_ms'].append((time.perf_counter()-tick)*1000)
            self.pending.append(self.executor.submit(self._render,frame,objects,t,camera,advice,preview))
        else:self._collect(self._render(frame,objects,t,camera,advice,preview))

    def finish(self,end_time):
        while self.pending:self._collect(self.pending.popleft().result())
        if self.writer:
            if self.executor:self.executor.submit(self.writer.close,end_time).result()
            else:self.writer.close(end_time)
        if self.executor:self.executor.shutdown(wait=True)
        self.closed=True

    def abort(self):
        # Registered after writer.abort: wait/cancel render jobs before closing it.
        if not self.closed and self.executor:self.executor.shutdown(wait=True,cancel_futures=True)
        self.closed=True
