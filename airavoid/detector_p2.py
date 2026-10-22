"""Opt-in adapter for the supplied single-class 800x1440 YOLO26m-P2 export.

Independent shape/taxonomy validation; never use the five-class 416x704 parser
configuration implicitly. Runtime allocation is shared with the existing adapter.
"""
from pathlib import Path
import cv2
import numpy as np
from .detector import Detector


def letterbox(image, height=800, width=1440, dtype=np.float32):
    h, w = image.shape[:2]
    ratio = min(width/w, height/h)
    nw, nh = round(w*ratio), round(h*ratio)
    left, top = (width-nw)//2, (height-nh)//2
    padded = np.full((height,width,3),114,np.uint8)
    padded[top:top+nh,left:left+nw] = cv2.resize(image,(nw,nh))
    x = np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None],dtype=dtype)/dtype(255)
    return x, ratio, left, top


class DetectorP2(Detector):
    names = ['fixed-wing UAV']
    input_size = (800,1440)

    def __init__(self, weights, conf=.03, backend='auto'):
        super().__init__(weights,conf=conf,backend=backend,tiled=False,gpu_preprocess=False)
        shape = (tuple(self.buffers[self.input_name].shape) if self.backend=='tensorrt'
                 else tuple(self.session.get_inputs()[0].shape))
        output = (tuple(self.buffers[self.output_name].shape) if self.backend=='tensorrt'
                  else tuple(self.session.get_outputs()[0].shape))
        if shape != (1,3,800,1440) or output != (1,300,6):
            raise ValueError(f'P2 export contract mismatch: input={shape}, output={output}')
        if self.backend != 'tensorrt':
            import ast
            meta=self.session.get_modelmeta().custom_metadata_map
            if ast.literal_eval(meta.get('names','{}')) != {0:'item'}:
                raise ValueError('Unexpected P2 class metadata')

    def _infer(self, image):
        h,w=image.shape[:2]
        x,ratio,left,top=letterbox(image,*self.input_size,dtype=self.input_dtype)
        if self.backend=='tensorrt':
            with self.torch.cuda.stream(self.stream):
                self.buffers[self.input_name].copy_(self.torch.from_numpy(x),non_blocking=True)
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError('P2 TensorRT execution failed')
                y=self.buffers[self.output_name].float().cpu().numpy()[0].copy()
            self.stream.synchronize()
        else:
            y=self.session.run(None,{self.input_name:x})[0][0].astype(float)
        return self._decode(y,h,w,ratio,left,top)

    def _decode(self,y,h,w,ratio,left,top):
        if y.ndim!=2 or y.shape[1]!=6:
            raise ValueError('Expected P2 end-to-end XYXY, score, class rows')
        selected=y[(y[:,4]>=self.conf)&np.isfinite(y).all(axis=1)]
        if len(selected) and (np.any(selected[:,5]!=0) or np.any(selected[:,4]>1)):
            raise ValueError('P2 must emit only class zero and bounded scores')
        return super()._decode(y,h,w,ratio,left,top)

    @property
    def contract(self):
        return {'variant':'B','input_shape':[1,3,800,1440], 'output_shape':[1,300,6],
                'export_label':'item','semantic_taxonomy':self.names,
                'letterbox':'RGB NCHW /255, linear resize, centred 114 pad, round dimensions',
                'preprocess':'CPU OpenCV; detector inference on declared backend',
                'conf':self.conf,'cross_box_nms_iou':.45,'tiled':False,
                'weights_path':str(Path(self.weights).resolve())}
