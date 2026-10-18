from pathlib import Path
import time
import cv2
import numpy as np

NAMES = ['airplane', 'helicopter', 'bird', 'balloon', 'parachute']
_TRT_LOGGER=None


def native_tiles(width,height,crop_width=704,crop_height=416,overlap=.25):
    """Cover any raster with native crops; preserve the established 1280x720 grid."""
    if min(width,height,crop_width,crop_height)<=0 or not 0<=overlap<1:
        raise ValueError('Positive raster/crop dimensions and overlap in [0,1) required')
    def positions(length,crop):
        excess=max(0,length-crop)
        count=int(np.ceil(excess/(crop*(1-overlap))))+1
        return sorted(set(int(round(v)) for v in np.linspace(0,excess,count)))
    return [(x,y) for y in positions(height,crop_height) for x in positions(width,crop_width)]


def interior_boxes(boxes,x,y,cw,ch,width,height,margin):
    """Optional crop-boundary ablation: reject potentially truncated observations."""
    if not margin or not len(boxes):return boxes
    keep=np.ones(len(boxes),bool)
    if x>0:keep &= boxes[:,0]>=margin
    if y>0:keep &= boxes[:,1]>=margin
    if x+cw<width:keep &= boxes[:,2]<=cw-margin
    if y+ch<height:keep &= boxes[:,3]<=ch-margin
    return boxes[keep]


class Detector:
    """Direct supplied YOLO26 end-to-end ONNX/TRT adapter. No Ultralytics dependency."""
    def __init__(self, weights, conf=.18, backend='auto', tiled=False, adaptive=False,gpu_preprocess=False,tile_edge_margin=0.):
        self.conf, self.tiled = conf, tiled
        self.tile_edge_margin=tile_edge_margin
        self.adaptive=adaptive
        self.reset_temporal()
        self.weights = str(weights)
        self.backend = 'tensorrt' if str(weights).endswith('.engine') else 'onnxruntime'
        if self.backend == 'tensorrt':
            import torch
            import tensorrt as trt
            self.torch = torch
            global _TRT_LOGGER
            if _TRT_LOGGER is None:_TRT_LOGGER=trt.Logger(trt.Logger.WARNING)
            self.logger=_TRT_LOGGER
            self.runtime = trt.Runtime(self.logger)
            raw = Path(weights).read_bytes()
            header = int.from_bytes(raw[:4], 'little')
            if 0 < header < 10000:
                raw = raw[4+header:]
            self.engine = self.runtime.deserialize_cuda_engine(raw)
            if self.engine is None:
                raise RuntimeError('Engine incompatible. Build a local engine or use ONNX.')
            self.context = self.engine.create_execution_context()
            self.stream = torch.cuda.Stream()
            self.buffers = {}
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                dtype = torch.from_numpy(np.empty((), trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
                self.buffers[name] = torch.empty(tuple(self.engine.get_tensor_shape(name)), dtype=dtype, device='cuda')
                self.context.set_tensor_address(name, self.buffers[name].data_ptr())
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.input_name = name
                else:
                    self.output_name = name
            self.input_dtype = self.buffers[self.input_name].cpu().numpy().dtype.type
            self.providers = ['TensorRT '+trt.__version__]
            self.gpu_preprocessor=None
            if gpu_preprocess:
                from .gpu import GpuLetterbox
                self.gpu_preprocessor=GpuLetterbox(self.buffers[self.input_name])
        else:
            if gpu_preprocess:raise ValueError('GPU OpenCV preprocessing requires the verified TensorRT backend')
            import torch
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            providers = ['CPUExecutionProvider'] if backend == 'cpu' else [('CUDAExecutionProvider', {'cudnn_conv_algo_search':'HEURISTIC'}), 'CPUExecutionProvider']
            self.session = ort.InferenceSession(str(weights), sess_options=options, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.input_dtype = np.float16 if 'float16' in self.session.get_inputs()[0].type else np.float32
            self.providers = self.session.get_providers()
            self.gpu_preprocessor=None

    def _infer(self, image):
        h, w = image.shape[:2]
        ratio = min(704/w, 416/h)
        nw, nh = round(w*ratio), round(h*ratio)
        left, top = (704-nw)//2, (416-nh)//2
        padded = np.full((416, 704, 3), 114, np.uint8)
        padded[top:top+nh, left:left+nw] = cv2.resize(image, (nw, nh))
        x = np.ascontiguousarray(padded[:,:,::-1].transpose(2,0,1)[None], dtype=self.input_dtype)/self.input_dtype(255)
        if self.backend == 'tensorrt':
            torch = self.torch
            with torch.cuda.stream(self.stream):
                self.buffers[self.input_name].copy_(torch.from_numpy(x), non_blocking=True)
                ok = self.context.execute_async_v3(self.stream.cuda_stream)
                if not ok:
                    raise RuntimeError('TRT execution failed')
                y = self.buffers[self.output_name].float().cpu().numpy()[0].copy()
            self.stream.synchronize()
        else:
            y = self.session.run(None, {self.input_name:x})[0][0].astype(float)
        return self._decode(y,h,w,ratio,left,top)

    def _decode(self,y,h,w,ratio,left,top):
        y = y[(y[:,4] >= self.conf) & np.isfinite(y).all(axis=1)]
        if len(y):
            y[:,[0,2]] = np.clip((y[:,[0,2]]-left)/ratio,0,w)
            y[:,[1,3]] = np.clip((y[:,[1,3]]-top)/ratio,0,h)
            y = y[(y[:,2]-y[:,0]>1) & (y[:,3]-y[:,1]>1)]
        return y

    def _infer_gpu(self,image):
        h,w,ratio,left,top=self.gpu_preprocessor.prepare(image)
        stream=self.gpu_preprocessor.stream
        if not self.context.execute_async_v3(stream.cudaPtr()):raise RuntimeError('GPU TensorRT execution failed')
        stream.waitForCompletion()
        y=self.buffers[self.output_name].float().cpu().numpy()[0].copy()
        return self._decode(y,h,w,ratio,left,top)

    def __call__(self, image):
        if self.gpu_preprocessor:
            self.gpu_preprocessor.upload(image)
            boxes=[self._infer_gpu(self.gpu_preprocessor.frame)]
        else:boxes = [self._infer(image)]
        self.last_network_calls=1
        if self.tiled:
            h,w = image.shape[:2]
            # Overlapping native-resolution crops recover tiny targets without new models.
            tiles=native_tiles(w,h)
            # If the source already fits, the full pass has exactly the same input.
            # Keep native_tiles' coverage contract, but avoid duplicate inference.
            if w<=704 and h<=416:tiles=[]
            selected=tiles
            if self.adaptive and self.calls%5:
                selected=sorted([tile for tile,seen in self.tile_memory.items() if self.calls-seen<=10],
                                key=lambda tile:-self.tile_memory[tile])[:4]
            for x,y in selected:
                if self.gpu_preprocessor:
                    b=self._infer_gpu(self.gpu_preprocessor.crop(x,y,min(704,w-x),min(416,h-y)))
                else:b = self._infer(image[y:y+416,x:x+704])
                b=interior_boxes(b,x,y,min(704,w-x),min(416,h-y),w,h,self.tile_edge_margin)
                self.last_network_calls+=1
                if len(b):
                    b[:,[0,2]] += x
                    b[:,[1,3]] += y
                    boxes.append(b)
        b = np.concatenate(boxes) if boxes else np.empty((0,6))
        if len(b):
            # Cross-tile class-agnostic suppression avoids duplicated physical objects.
            xywh = np.c_[b[:,:2], b[:,2:4]-b[:,:2]].tolist()
            ids = cv2.dnn.NMSBoxes(xywh,b[:,4].tolist(),self.conf,.45)
            b = b[np.asarray(ids,dtype=int).ravel()] if len(ids) else b[:0]
        self.previous=b.tolist()
        if self.adaptive and self.tiled:
            for box in sorted(self.previous,key=lambda row:-row[4]):
                if max(box[2]-box[0],box[3]-box[1])>120:continue
                cx=(box[0]+box[2])/2;cy=(box[1]+box[3])/2
                nearby=sorted(tiles,key=lambda xy:(xy[0]+min(704,w)/2-cx)**2+(xy[1]+min(416,h)/2-cy)**2)
                for tile in nearby[:2]:self.tile_memory[tile]=self.calls
        self.calls+=1
        return self.previous

    def reset_temporal(self):
        self.previous=[];self.calls=0;self.last_network_calls=0;self.tile_memory={}
