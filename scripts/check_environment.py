"""Read-only supplied-environment fingerprint and CUDA OpenCV preservation check."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import cv2
import numpy as np
import torch
import tensorrt
import av

initial=json.loads(Path('reports/environment_initial.json').read_text())
source=np.arange(96,dtype=np.uint8).reshape(8,12);gpu=cv2.cuda_GpuMat();gpu.upload(source)
actual=gpu.download();assert np.array_equal(source,actual)
current={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),
         'cv2':cv2.__version__,'cv2_path':cv2.__file__,'cuda_devices':cv2.cuda.getCudaEnabledDeviceCount(),
         'build_sha256':hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),'torch':torch.__version__,
         'tensorrt':tensorrt.__version__,'av':av.__version__,'gpu':torch.cuda.get_device_name(0),
         'cuda_opencv_upload_download_exact':bool(np.array_equal(source,actual)),
         'same_initial_cv2_version':cv2.__version__==initial['cv2'],
         'same_initial_import_path':Path(cv2.__file__).resolve()==Path(initial['cv2_path']).resolve(),
         'same_initial_build':cv2.getBuildInformation()==initial['build'],
         'same_initial_cuda_count':cv2.cuda.getCudaEnabledDeviceCount()==initial['cuda_devices']}
Path('reports/environment_final.json').write_text(json.dumps(current,indent=2))
print(json.dumps(current,indent=2))
assert all(current[key] for key in ['same_initial_cv2_version','same_initial_import_path','same_initial_build','same_initial_cuda_count'])
