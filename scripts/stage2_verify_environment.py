"""Read-only final preservation check; never replaces earlier experiment reports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import torch
# Preserve the established DLL-loading order before the custom OpenCV import.
with torch.inference_mode():
    torch.nn.Conv2d(3, 4, 3).cuda()(torch.zeros(1, 3, 16, 16, device='cuda'))
    torch.cuda.synchronize()
import cv2
import numpy as np
import tensorrt
import av

ROOT = Path(__file__).resolve().parents[1]


def verify():
    original = json.loads((ROOT / 'reports/environment_initial.json').read_text())
    source = np.arange(96, dtype=np.uint8).reshape(8, 12)
    gpu = cv2.cuda_GpuMat()
    gpu.upload(source)
    restored = gpu.download()
    checks = {
        'same_initial_cv2_version': cv2.__version__ == original['cv2'],
        'same_initial_cv2_path': Path(cv2.__file__).resolve() == Path(original['cv2_path']).resolve(),
        'same_initial_build': cv2.getBuildInformation() == original['build'],
        'same_initial_cuda_count': cv2.cuda.getCudaEnabledDeviceCount() == original['cuda_devices'],
        'cuda_upload_download_exact': bool(np.array_equal(source, restored)),
    }
    report = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'cv2': cv2.__version__,
        'cv2_path': cv2.__file__,
        'build_sha256': hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
        'torch': torch.__version__, 'tensorrt': tensorrt.__version__, 'av': av.__version__,
        'gpu': torch.cuda.get_device_name(0), 'checks': checks,
        'all_checks_pass': all(checks.values()),
        'operation': 'read-only imports, one CUDA convolution and exact CUDA upload/download',
    }
    target = ROOT / 'reports/stage2_environment_final.json'
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if not report['all_checks_pass']:
        raise RuntimeError('Original CUDA OpenCV preservation check failed')


if __name__ == '__main__':
    verify()
