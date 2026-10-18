"""Source presentation timestamps and timestamp-preserving offline H.264 export."""
from fractions import Fraction
import json
import subprocess
import numpy as np


def encoder_options(encoder):
    if encoder=='libx264':return {'preset':'veryfast','crf':'20','tune':'zerolatency','bf':'0'}
    if encoder=='h264_nvenc':
        return {'preset':'p3','tune':'ull','rc':'constqp','qp':'20','bf':'0','rc-lookahead':'0','delay':'0'}
    raise ValueError('Supported H.264 encoders: libx264, h264_nvenc')


def probe_video(path):
    result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                           'stream=duration,start_time,avg_frame_rate,r_frame_rate,nb_frames,time_base:format=duration',
                           '-of','json',str(path)],capture_output=True,text=True,check=True)
    data=json.loads(result.stdout)
    return {**data['streams'][0],'format_duration':data.get('format',{}).get('duration')}


def presentation_end(path,metadata):
    """Inspect tail packet PTS/durations; stream.duration can disagree after VFR edits."""
    import av
    with av.open(str(path)) as reader:
        first_stream=reader.streams.video[0]
        stream_start=float(first_stream.start_time*first_stream.time_base) if first_stream.start_time is not None else 0.
        first=next(reader.decode(video=0),None)
        origin=float(first.pts*first.time_base) if first is not None and first.pts is not None else stream_start
        origin_kind='first_decoded_frame' if first is not None and first.pts is not None else 'stream_start_fallback'
    with av.open(str(path)) as container:
        stream=container.streams.video[0]
        start=float(stream.start_time*stream.time_base) if stream.start_time is not None else 0.
        guess=float(metadata.get('duration') or metadata.get('format_duration') or 0.)
        if guess>6:
            try:container.seek(int((start+max(0,guess-6))/float(stream.time_base)),stream=stream)
            except (ValueError,RuntimeError):pass
        ends=[]
        for packet in container.demux(stream):
            if packet.pts is not None and packet.duration and packet.duration>0:
                ends.append(float((packet.pts+packet.duration)*packet.time_base)-origin)
        if ends:return max(ends),'tail_packet_end_relative_'+origin_kind
    try:return max(0.,float(metadata['duration'])-(origin-stream_start)),'stream_duration_fallback'
    except (ValueError,TypeError,KeyError):return None,'unavailable'


class SourceClock:
    def __init__(self,fps):
        self.fps=fps;self.origin=None;self.previous=None;self.mode=None;self.snapped=0

    def update(self,index,milliseconds):
        value=float(milliseconds)/1000.
        if not np.isfinite(value):value=0.
        if self.origin is None:self.origin=value;self.previous=value;return 0.
        if self.mode is None:self.mode='presentation_timestamps' if value>self.previous+1e-9 else 'frame_rate_fallback'
        if self.mode=='frame_rate_fallback':return index/self.fps
        if value<=self.previous+1e-9:raise ValueError('Source presentation timestamps are not strictly increasing')
        self.previous=value;t=value-self.origin
        # Preserve stable numeric behavior on CFR sources, without flattening VFR.
        expected=index/self.fps
        if abs(t-expected)<1e-6:self.snapped+=1;return expected
        return t


class TimestampWriter:
    """One-frame buffer, explicit PTS and packet durations, no temporal resampling."""
    def __init__(self,path,fps,size,encoder='libx264',nvenc_rgb_input=False):
        if nvenc_rgb_input and encoder!='h264_nvenc':raise ValueError('RGB encoder input requires h264_nvenc')
        import av
        self.av=av;self.time_base=Fraction(1,90000);self.nominal_step=1/float(fps)
        self.container=av.open(str(path),'w',options={'movflags':'+faststart'})
        self.stream=self.container.add_stream(encoder,rate=Fraction(float(fps)).limit_denominator(100000))
        self.stream.width,self.stream.height=size;self.stream.pix_fmt='bgr0' if nvenc_rgb_input else 'yuv420p'
        self.stream.time_base=self.time_base;self.stream.codec_context.time_base=self.time_base
        self.stream.codec_context.max_b_frames=0;self.stream.codec_context.thread_count=4 if encoder=='libx264' else 1
        self.stream.options=encoder_options(encoder)
        if nvenc_rgb_input:self.stream.options={**self.stream.options,'rgb_mode':'yuv420'}
        self.pending=None;self.durations={};self.frames=0;self.closed=False

    def write(self,image,t):
        if self.pending is not None:
            old,old_t=self.pending
            if t<=old_t:raise ValueError('Output PTS must increase')
            self._encode(old,old_t,t-old_t)
        self.pending=(image,float(t))

    def _encode(self,image,t,duration):
        frame=self.av.VideoFrame.from_ndarray(image,format='bgr24')
        frame.pts=round(t/self.time_base);frame.time_base=self.time_base
        self.durations[frame.pts]=max(1,round(duration/self.time_base))
        for packet in self.stream.encode(frame):self._mux(packet)
        self.frames+=1

    def _mux(self,packet):
        if packet.pts is None:raise RuntimeError('Encoder did not preserve frame PTS')
        base=packet.time_base or self.time_base
        key=round(Fraction(packet.pts)*base/self.time_base)
        duration=self.durations.pop(key,None)
        if duration is not None:packet.duration=max(1,round(Fraction(duration)*self.time_base/base))
        self.container.mux(packet)

    def close(self,end_time=None):
        if self.closed:return
        if self.pending is not None:
            image,t=self.pending
            duration=self.nominal_step if end_time is None else max(float(self.time_base),end_time-t)
            self._encode(image,t,duration);self.pending=None
        for packet in self.stream.encode():self._mux(packet)
        self.container.close();self.closed=True

    def abort(self):
        if not self.closed:
            try:self.container.close()
            finally:self.closed=True
