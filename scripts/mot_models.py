"""Research adapters for pinned upstream online MOT implementations.

Run in a fresh process per tracker: upstream projects reuse package names.
Dependencies live in external/mot/python_deps; the production env is untouched.
"""
from pathlib import Path
import sys, types, importlib.util, ast
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'external/mot'
sys.path.insert(0, str(EXT / 'python_deps'))
# Compatibility aliases for original 2022 research implementations on NumPy 2.
for name, value in [('float', float), ('int', int), ('float_', np.float64)]:
    if name not in np.__dict__: setattr(np, name, value)


def package(name, path=None):
    mod = types.ModuleType(name)
    mod.__path__ = [] if path is None else [str(path)]
    sys.modules[name] = mod
    return mod


def mamba_reference():
    """Use upstream Mamba 1.2.0 reference scan with eager PyTorch on Windows.

    Same weights and recurrence; no Triton/CUDA extension or torch.compile/AOT.
    This measures this portable backend, not the paper's optimized kernels.
    """
    import torch
    import torch.nn.functional as F
    from einops import rearrange, repeat
    base = EXT / 'mamba_reference/mamba_ssm'
    for name, path in [('mamba_ssm', base), ('mamba_ssm.modules', base/'modules'),
                       ('mamba_ssm.ops', base/'ops'), ('mamba_ssm.ops.triton', None)]:
        package(name, path)
    scan = types.ModuleType('mamba_ssm.ops.selective_scan_interface')
    scan.__dict__.update(torch=torch, F=F, rearrange=rearrange, repeat=repeat)
    tree = ast.parse((base/'ops/selective_scan_interface.py').read_text())
    node = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'selective_scan_ref')
    exec(compile(ast.Module(body=[node],type_ignores=[]), str(base/'ops/selective_scan_interface.py'), 'exec'), scan.__dict__)
    scan.selective_scan_fn = scan.selective_scan_ref
    def unavailable(*a, **kw): raise RuntimeError('Fused Mamba kernel unavailable in reference backend')
    scan.mamba_inner_fn = unavailable
    sys.modules[scan.__name__] = scan
    norm = types.ModuleType('mamba_ssm.ops.triton.layernorm')
    class RMSNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
            super().__init__(); self.eps=eps
            self.weight=torch.nn.Parameter(torch.ones(hidden_size,device=device,dtype=dtype)); self.register_parameter('bias',None)
        def forward(self,x): return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps)).to(x.dtype)*self.weight
    def rms_norm_fn(x, weight, bias=None, residual=None, eps=1e-5, prenorm=False, residual_in_fp32=False, **kw):
        r=x if residual is None else x+residual
        y=(r.float()*torch.rsqrt(r.float().square().mean(-1,keepdim=True)+eps)).to(weight.dtype)*weight
        if bias is not None:y=y+bias
        return (y,r.float() if residual_in_fp32 else r) if prenorm else y
    def layer_norm_fn(x,weight,bias=None,residual=None,eps=1e-5,prenorm=False,residual_in_fp32=False,**kw):
        r=x if residual is None else x+residual
        y=F.layer_norm(r.to(weight.dtype),(weight.numel(),),weight,bias,eps)
        return (y,r.float() if residual_in_fp32 else r) if prenorm else y
    norm.RMSNorm=RMSNorm;norm.rms_norm_fn=rms_norm_fn;norm.layer_norm_fn=layer_norm_fn
    sys.modules[norm.__name__]=norm


class Appearance:
    """Original FastReID DanceTrack SBS-S50 weights/config, batched RGB crops."""
    def __init__(self):
        import torch
        base=EXT/'TrackTrack/2. FastReID'
        sys.path.insert(0,str(base))
        from fastreid.config import get_cfg
        from fastreid.modeling.meta_arch import build_model
        cfg=get_cfg();cfg.merge_from_file(str(base/'configs/DanceTrack/sbs_S50.yml'))
        cfg.MODEL.BACKBONE.PRETRAIN=False;cfg.MODEL.DEVICE='cuda'
        ck=torch.load(EXT/'weights/dance_sbs_S50.pth',map_location='cpu',weights_only=False)
        sd=ck['model'] if 'model' in ck else ck
        cfg.MODEL.HEADS.NUM_CLASSES=int(sd['heads.weight'].shape[0])
        self.model=build_model(cfg).eval().cuda().half()
        # Classifier weights are training-only. All inference weights must match.
        incompatible=self.model.load_state_dict(sd,strict=False)
        bad=[k for k in incompatible.missing_keys+incompatible.unexpected_keys if not k.startswith('heads.weight')]
        if bad:raise ValueError('FastReID incompatible: '+str(bad))
        self.size=tuple(cfg.INPUT.SIZE_TEST[::-1]);self.dim=2048
    def __call__(self,img,boxes):
        import torch,cv2
        if not len(boxes):return np.empty((0,self.dim),np.float32)
        h,w=img.shape[:2];patches=[]
        for b in boxes:
            x1,y1,x2,y2=np.asarray(b[:4]).astype(int)
            x1=int(np.clip(x1,0,w-1));y1=int(np.clip(y1,0,h-1))
            x2=int(np.clip(x2,x1+1,w));y2=int(np.clip(y2,y1+1,h))
            patches.append(cv2.resize(img[y1:y2,x1:x2,::-1],self.size).transpose(2,0,1).copy())
        result=[]
        with torch.inference_mode():
            for start in range(0,len(patches),16):
                batch=torch.from_numpy(np.stack(patches[start:start+16])).cuda().half()
                result.append(torch.nn.functional.normalize(self.model(batch).float(),dim=1).cpu().numpy())
        return np.concatenate(result)


class Adapter:
    def __init__(self,name,width,height,fps=25,high=.03,appearance=True):
        import torch
        self.name=name;self.w=width;self.h=height;self.frame=0;self.predictions=[]
        self.appearance=Appearance() if appearance and name in ['tracktrack','botsort','diffmot','trackssm'] else None
        if name=='custom':
            from airavoid.tracking import Tracker
            self.tracker=Tracker()
        elif name=='ocsort':
            sys.path.insert(0,str(EXT/'OCSORT'))
            from trackers.ocsort_tracker.ocsort import OCSort
            self.tracker=OCSort(det_thresh=high,max_age=round(fps*.8),min_hits=3,iou_threshold=.1,use_byte=True)
        elif name=='tracktrack':
            sys.path.insert(0,str(EXT/'TrackTrack/3. Tracker'))
            import trackers.tracker as upstream
            class OnlineCMC:
                def __init__(s,vid):s.matrix=np.eye(2,3)
                def get_warp_matrix(s):return s.matrix
            upstream.CMC=OnlineCMC
            args=SimpleNamespace(max_time_lost=round(fps*.8),det_thr=high,init_thr=high,match_thr=.7,
                penalty_p=.2,penalty_q=.4,reduce_step=.05,tai_thr=.55,data_path='Airborne',img_w=width,img_h=height,min_len=3)
            self.tracker=upstream.Tracker(args,'airborne')
            from airavoid.tracking import CameraMotion
            self.camera=CameraMotion('auto');self.previous_transform=np.eye(3)
        elif name=='botsort':
            sys.path.insert(0,str(EXT/'BoTSORT'))
            # Avoid importing the upstream training engine; inject our real inference extractor.
            module=types.ModuleType('fast_reid.fast_reid_interfece')
            owner=self
            class FastReIDInterface:
                def __init__(s,*a,**kw):pass
                def inference(s,img,dets):return owner.appearance(img,dets)
            module.FastReIDInterface=FastReIDInterface;sys.modules[module.__name__]=module
            from tracker.bot_sort import BoTSORT
            args=SimpleNamespace(track_high_thresh=high,track_low_thresh=.029,new_track_thresh=high,
                track_buffer=round(fps*.8*30/fps),proximity_thresh=.5,appearance_thresh=.25,
                with_reid=self.appearance is not None,fast_reid_config='',fast_reid_weights='',device='cuda',
                cmc_method='sparseOptFlow',name='airborne',ablation=False,match_thresh=.8,mot20=False)
            self.tracker=BoTSORT(args,frame_rate=fps)
        elif name in ['diffmot','trackssm']:
            base=EXT/('DiffMOT' if name=='diffmot' else 'TrackSSM')
            sys.path.insert(0,str(base))
            if name=='trackssm':mamba_reference()
            import yaml
            cfg=SimpleNamespace(**yaml.safe_load((base/'configs/dancetrack_test.yaml').read_text()))
            cfg.high_thres=high;cfg.low_thres=.029;cfg.reid_dir=str(EXT/'cache')
            from models.autoencoder import D2MP
            if name=='diffmot':
                from models.condition_embedding import History_motion_embedding
                encoder=History_motion_embedding();weight='DiffMOT_DanceTrack.pt'
            else:
                from models.condition_embedding import Time_info_aggregation
                encoder=Time_info_aggregation();weight='TrackSSM_DanceTrack.pt'
            self.model=D2MP(cfg,encoder=encoder).cuda().eval()
            ck=torch.load(EXT/'weights'/weight,map_location='cpu',weights_only=False)
            self.model.load_state_dict({k.removeprefix('module.'):v for k,v in ck['ddpm'].items()},strict=True)
            # Original TrackSSM decoder predicts absolute normalized boxes; its tracker
            # expects delta samples. Repair this upstream interface mismatch explicitly.
            if name=='trackssm':
                def generate(conds,sample,bestof,flexibility=0.,ret_traj=False,img_w=None,img_h=None):
                    arr=[]
                    for c in conds:
                        x=np.asarray(c).copy();x[:,0::2]/=img_w;x[:,1::2]/=img_h
                        if len(x)!=cfg.interval:x=np.concatenate([x,np.repeat(x[-1:],cfg.interval,axis=0)])[:cfg.interval]
                        arr.append(x)
                    x=torch.as_tensor(np.stack(arr),device='cuda',dtype=torch.float32)
                    y=self.model.ssm_decoder(x[:,-1,:4],self.model.encoder(x))
                    return (y-x[:,-1,:4]).unsqueeze(0).cpu().numpy()
                self.model.generate=generate
            owner=self
            emb=types.ModuleType('tracker.embedding')
            class EmbeddingComputer:
                def __init__(s,*a,**kw):pass
                def compute_embedding(s,img,boxes,tag):
                    return owner.appearance(img,boxes) if owner.appearance else np.ones((len(boxes),1))
                def dump_cache(s):pass
            emb.EmbeddingComputer=EmbeddingComputer;sys.modules[emb.__name__]=emb
            from tracker.DiffMOTtracker import diffmottracker
            self.tracker=diffmottracker(cfg,frame_rate=fps)
        elif name=='putr':
            base=EXT/'PuTR';sys.path.insert(0,str(base))
            import yaml
            from models import build_model
            from models.runtime_tracker import RuntimeTracker
            cfg=yaml.safe_load((base/'configs/eval_dancetrack_putr.yaml').read_text())
            cfg.update(DET_SCORE_THRESH=.029,TRACK_SCORE_THRESH=high,MISS_TOLERANCE=round(fps*.8))
            self.model=build_model(cfg).eval()
            ck=torch.load(EXT/'weights/putr_dancetrack.pth',map_location='cpu',weights_only=False)
            self.model.load_state_dict(ck['model'],strict=True)
            self.tracker=RuntimeTracker(cfg,self.model)
        else:raise ValueError(name)

    def update(self,dets,img,t,warp=None):
        import torch
        self.frame+=1;d=np.asarray(dets,dtype=float).reshape(-1,6)
        if self.name=='custom':
            tr=self.tracker.update(d,t,np.eye(3))
            return [(x.id,*x.box) for x in tr if x.last==t]
        if self.name=='ocsort':
            r=self.tracker.update(d[:,:5],(self.h,self.w),(self.h,self.w))
            return [(int(x[4]),*x[:4]) for x in r]
        if self.name=='botsort':
            r=self.tracker.update(d[:,:5],img)
            return [(x.track_id,*x.tlbr) for x in r]
        if self.name=='tracktrack':
            if warp is None:
                transform,info=self.camera.update(img,t,d)
                warp=(np.linalg.inv(transform)@self.previous_transform)[:2] if info['valid'] and not info['evidence_reset'] else np.eye(2,3)
                self.previous_transform=transform
            self.tracker.cmc.matrix=np.eye(2,3) if warp is None else warp
            f=self.appearance(img,d[:,:4]) if self.appearance else np.ones((len(d),1))
            combined=np.concatenate([d,f],axis=1)
            r=self.tracker.update(combined,combined) if len(d) else self.tracker.update_without_detections()
            return [(x.track_id,*x.x1y1x2y2) for x in r]
        if self.name in ['diffmot','trackssm']:
            d=d[:,:5].copy();d[:,2:4]-=d[:,:2]
            with torch.inference_mode():r=self.tracker.update(d,self.model,self.frame,self.w,self.h,'airborne',img=img)
            return [(x.track_id,*x.tlbr) for x in r]
        if self.name=='putr':
            import cv2
            # Match the original SeqDataset: RGB, short side 800, long side <=1536.
            short=min(800,int(1536*min(self.h,self.w)/max(self.h,self.w)))
            nh=short if self.h<=self.w else round(short*self.h/self.w)
            nw=round(short*self.w/self.h) if self.h<=self.w else short
            image=torch.from_numpy(cv2.resize(img[:,:,::-1],(nw,nh)).copy()).unsqueeze(0).cuda()
            orig=torch.as_tensor(d,device='cuda',dtype=torch.float32)
            detections=orig.clone();detections[:,:4]*=torch.tensor([nw/self.w,nh/self.h,nw/self.w,nh/self.h],device='cuda')
            with torch.inference_mode():r=self.tracker.update(image,detections,orig)
            # Online outputs only: no retrospective addition of tentative history.
            return [(int(x[14]),*x[6:10]) for x in r.cpu().numpy() if int(x[15])==1 and int(x[13])==self.frame-1]
