"""Build a local TensorRT engine; never alter supplied weights or Python packages."""
import argparse
from pathlib import Path
import time
import torch  # Preload the CUDA DLLs shipped with the existing environment.
import tensorrt as trt

p = argparse.ArgumentParser()
p.add_argument('--onnx', default='../anti_collision_system/best.fp16.onnx')
p.add_argument('--output', default='artifacts/yolo26m_local.engine')
a = p.parse_args()
t = time.perf_counter()
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)
if not parser.parse(Path(a.onnx).read_bytes()):
    raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
engine = builder.build_serialized_network(network, config)
if engine is None:
    raise RuntimeError('TensorRT build failed')
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_bytes(engine)
print(f'Built {a.output} in {time.perf_counter()-t:.2f}s', flush=True)
