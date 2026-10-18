"""Native CUDA OpenCV letterbox into TensorRT's existing FP32 device input.

Memory remains owned by the Torch tensor. GpuMat views neither copy nor free it.
One source upload per detector call, shared by overlapping crop inferences.
"""
import cv2
import numpy as np


class GpuLetterbox:
    def __init__(self,input_tensor):
        if tuple(input_tensor.shape)!=(1,3,416,704):raise ValueError('Expected fixed 416x704 input')
        if input_tensor.element_size()!=4:raise ValueError('Native GPU preprocessing currently requires FP32 TensorRT input')
        self.stream=cv2.cuda_Stream();self.frame=cv2.cuda_GpuMat()
        self.padded=cv2.cuda_GpuMat(416,704,cv2.CV_8UC3)
        self.rgb=cv2.cuda_GpuMat(416,704,cv2.CV_8UC3)
        self.channels=[cv2.cuda_GpuMat(416,704,cv2.CV_8UC1) for _ in range(3)]
        ptr=input_tensor.data_ptr();plane=416*704
        self.planes=[cv2.cuda.createGpuMatFromCudaMemory(416,704,cv2.CV_32F,ptr+i*plane*4,704*4) for i in range(3)]
        self.resized={}

    def upload(self,image):
        self.frame.upload(image,self.stream)

    def crop(self,x,y,w,h):
        return cv2.cuda_GpuMat(self.frame,(int(x),int(y),int(w),int(h)))

    def prepare(self,image):
        w,h=image.size();ratio=min(704/w,416/h)
        nw,nh=round(w*ratio),round(h*ratio)
        left,top=(704-nw)//2,(416-nh)//2
        if (nw,nh) not in self.resized:self.resized[nw,nh]=cv2.cuda_GpuMat(nh,nw,cv2.CV_8UC3)
        resized=self.resized[nw,nh]
        cv2.cuda.resize(src=image,dst=resized,dsize=(nw,nh),interpolation=cv2.INTER_LINEAR,stream=self.stream)
        cv2.cuda.copyMakeBorder(src=resized,top=top,bottom=416-nh-top,left=left,right=704-nw-left,
                                borderType=cv2.BORDER_CONSTANT,dst=self.padded,value=(114,114,114),stream=self.stream)
        cv2.cuda.cvtColor(self.padded,cv2.COLOR_BGR2RGB,self.rgb,0,self.stream)
        cv2.cuda.split(self.rgb,self.channels,self.stream)
        for channel,destination in zip(self.channels,self.planes):
            channel.convertTo(cv2.CV_32F,1/255.,0.,self.stream,destination)
        return h,w,ratio,left,top
