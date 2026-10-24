"""Assemble the customer runtime; build/QA tools themselves are not delivered."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'handoff/AirAntiCollision'


def write(name, text):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text.lstrip(), encoding='utf-8-sig' if p.suffix == '.ps1' else 'utf-8')


def copy(src, dest):
    p = OUT / dest
    p.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, p)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import sys
    if OUT.exists() and '--resume' not in sys.argv:
        raise SystemExit('Delivery folder already exists; preserve it and use a new destination.')
    OUT.mkdir(parents=True, exist_ok=True)
    excluded = {'tartan_reference.py', 'appearance.py', 'proposals.py', '__main__.py'}
    for src in (ROOT / 'airavoid').glob('*.py'):
        if src.name not in excluded:
            copy(src, 'app/airavoid/' + src.name)
    p = OUT / 'app/airavoid/assessment_runtime.py'
    text = p.read_text(encoding='utf-8')
    old = "Path(__file__).resolve().parents[1]/'reports/stage2_method_provenance.json'"
    assert text.count(old) == 1
    p.write_text(text.replace(old, "Path(__file__).resolve().parents[2]/'models/registry.json'"), encoding='utf-8')
    assets = {
        'A.engine': ROOT / 'artifacts/yolo26m_local.engine',
        'A.onnx': ROOT.parent / 'anti_collision_system/best.fp16.onnx',
        'B.engine': ROOT / 'artifacts/stage2/yolo26m_p2_local_fp32.engine',
        'B.onnx': Path('X:/MODELS/yolo26m-p2-1440/best.onnx'),
    }
    registry = {'detectors': {}, 'engine_device': 'NVIDIA GeForce RTX 4060',
                'tensorrt_version': '11.4.0.106'}
    for name, src in assets.items():
        copy(src, 'models/' + name)
        variant, kind = name.split('.')
        registry['detectors'][variant + '_' + kind] = {'path': 'models/' + name, 'sha256': digest(src), 'bytes': src.stat().st_size}
    write('models/registry.json', json.dumps(registry, indent=2))
    copy('X:/DATASETS/UAV/TRACKING/RGB/NPS-Drones/Videos/Clip_32.mov', 'examples/NPS_032.mov')
    copy(ROOT / 'data/qut/M2.mp4', 'examples/QUT_M2.mp4')
    for src, dst in [('aviation_nps_field', 'NPS_032'), ('aviation_approach_field', 'QUT_M2')]:
        copy(ROOT / f'demo/{src}/annotated.mp4', f'demo/{dst}.mp4')
        copy(ROOT / f'demo/{src}/encounter_cue.png', f'demo/{dst}.png')
    (OUT / 'results').mkdir(exist_ok=True)
    write('app/run.py', RUN)
    write('app/prepare_models.py', PREPARE)
    write('launch.ps1', LAUNCH)
    write('setup.ps1', SETUP)
    write('requirements-windows.txt', 'numpy==2.5.2\nPillow==12.3.0\nav==18.1.0\ntensorrt==11.4.0.106\nopencv-python==5.0.0.93\n')
    for name, command in {
        'Watch_NPS.cmd': 'start "" "%~dp0demo\\NPS_032.mp4"',
        'Watch_QUT.cmd': 'start "" "%~dp0demo\\QUT_M2.mp4"',
        'Run_NPS.cmd': 'powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" -Example nps',
        'Run_QUT.cmd': 'powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" -Example qut',
        'Setup.cmd': 'powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"',
    }.items():
        write(name, '@echo off\n' + command + '\nif errorlevel 1 pause\n')
    notices = [
        ('NPS_BSD3.txt', 'https://engineering.purdue.edu/~bouman/UAV_Dataset/pubs/LICENSE.txt'),
        ('YOLO_AGPL3.txt', 'https://raw.githubusercontent.com/ultralytics/ultralytics/main/LICENSE'),
        ('QUT_LICENSE', 'https://huggingface.co/datasets/jasminmartin/vision_based_aircraft_detection_dataset/raw/main/LICENSE'),
    ]
    for name, url in notices:
        cached = ROOT / 'artifacts/client_document_sources' / name
        if cached.exists():
            copy(cached, 'licenses/' + name)
            continue
        try:
            text = urllib.request.urlopen(url, timeout=20).read().decode('utf-8')
            write('licenses/' + name, text)
        except Exception as error:
            if name != 'QUT_LICENSE':
                raise
            write('licenses/QUT_SOURCE.txt', 'QUT Aircraft Collision Course Dataset\nBSD-3-Clause as declared by its authors.\n'
                  'https://huggingface.co/datasets/jasminmartin/vision_based_aircraft_detection_dataset\n'
                  'Martin, Riseley, Ford (2022), ICUAS, pp.144-149.\n')
    write('licenses/NOTICE.txt', 'Models are supplied YOLO26m exports; upstream metadata declares AGPL-3.0.\n'
          'Source: https://docs.ultralytics.com/models/yolo26/\n'
          'NPS footage: Li et al., Purdue / Naval Postgraduate School, BSD-3-Clause.\n'
          'QUT footage: Martin, Riseley, Ford, Aircraft Collision Course Dataset, BSD-3-Clause.\n'
          'Python dependencies are downloaded by Setup.cmd and retain their own licenses.\n'
          'No third-party Python packages or FFmpeg binaries are redistributed in this folder.\n')
    print(OUT)


RUN = r'''
"""Customer entry point. Only supported RGB showcase options are exposed."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description='AirAntiCollision: recorded RGB aircraft indicator')
    p.add_argument('--example', choices=['nps','qut'], default='nps')
    p.add_argument('--video', type=Path)
    p.add_argument('--model', choices=['A','B'])
    p.add_argument('--camera', choices=['auto','fixed','unknown'])
    p.add_argument('--nps', action='store_true', help='Explicit NPS-only approximate-range preset')
    p.add_argument('--output', type=Path)
    p.add_argument('--max-frames', type=int)
    a = p.parse_args()
    video = a.video.resolve() if a.video else ROOT / ('examples/NPS_032.mov' if a.example=='nps' else 'examples/QUT_M2.mp4')
    model = a.model or ('A' if not a.video and a.example=='qut' else 'B')
    camera = a.camera or ('fixed' if not a.video and a.example=='qut' else 'auto')
    out = a.output.resolve() if a.output else ROOT / 'results' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    if not video.is_file():p.error('Video file not found: '+str(video))
    os.chdir(ROOT)
    try:
        import torch
        if not torch.cuda.is_available():raise RuntimeError('NVIDIA CUDA device unavailable')
        from airavoid.cli import parse_args, execute
        from airavoid.lifecycle import RunSession
        args = [str(video),'--weights',str(ROOT/'models'/f'{model}.engine'),
                '--detector-variant',model,'--camera',camera,'--demo-showcase',
                '--demo-layout','aviation','--cpu-preprocess','--encoder','h264_nvenc',
                '--async-render','--output',str(out)]
        if a.nps or (not a.video and a.example=='nps'):args += ['--nps-range-demo']
        if a.max_frames:args += ['--max-frames',str(a.max_frames)]
        config = parse_args(args)
        with RunSession(out, {'args':vars(config)}) as session:execute(config, session)
    except (ImportError, RuntimeError) as error:
        print(str(error),file=sys.stderr)
        print('Run Setup.cmd to prepare a separate environment. For another GPU, rebuild using app/prepare_models.py.',file=sys.stderr)
        raise SystemExit(1)


if __name__=='__main__':main()
'''

PREPARE = r'''
"""Build hardware-local TensorRT engines from the unchanged delivered ONNX."""
import argparse,hashlib,json
from pathlib import Path
import torch
import tensorrt as trt

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['A','B','both'],default='both');a=p.parse_args()
    registry_path=ROOT/'models/registry.json';registry=json.loads(registry_path.read_text())
    for model in (['A','B'] if a.model=='both' else [a.model]):
        source=ROOT/'models'/f'{model}.onnx'
        if sha(source)!=registry['detectors'][model+'_onnx']['sha256']:raise ValueError('ONNX checksum differs')
        logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger)
        network=builder.create_network(0);parser=trt.OnnxParser(network,logger)
        if not parser.parse(source.read_bytes()):raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config=builder.create_builder_config();config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<30)
        engine=builder.build_serialized_network(network,config)
        if engine is None:raise RuntimeError('Engine build failed')
        target=ROOT/'models'/f'{model}.engine';temporary=target.with_suffix('.engine.part')
        temporary.write_bytes(engine);temporary.replace(target)
        registry['detectors'][model+'_engine']={'path':'models/'+target.name,'sha256':sha(target),'bytes':target.stat().st_size}
        registry['engine_device']=torch.cuda.get_device_name(0);registry['tensorrt_version']=trt.__version__
        registry_path.write_text(json.dumps(registry,indent=2))
        print('Built',target,flush=True)

if __name__=='__main__':main()
'''

LAUNCH = r'''
param([ValidateSet('nps','qut')][string]$Example='nps',[string]$Video='',[string]$Model='')
$ErrorActionPreference='Stop'
$deliveryPython=if($env:AIRAVOID_PYTHON){$env:AIRAVOID_PYTHON}else{Join-Path $PSScriptRoot '.venv/Scripts/python.exe'}
if(-not(Test-Path -LiteralPath $deliveryPython)){throw 'Run Setup.cmd first, or set AIRAVOID_PYTHON to your prepared Python executable.'}
if(-not(Get-Command ffmpeg -ErrorAction SilentlyContinue)-or-not(Get-Command ffprobe -ErrorAction SilentlyContinue)){throw 'FFmpeg and FFprobe must be on PATH. Install with winget install --id Gyan.FFmpeg --exact, then reopen this window.'}
$deliveryArgs=@((Join-Path $PSScriptRoot 'app/run.py'),'--example',$Example)
if($Video){$deliveryArgs+=@('--video',$Video)}
if($Model){$deliveryArgs+=@('--model',$Model)}
& $deliveryPython @deliveryArgs
if($LASTEXITCODE -ne 0){throw 'Processing failed; see the message above.'}
'''

SETUP = r'''
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
if(-not(Get-Command py -ErrorAction SilentlyContinue)){throw 'Install Python 3.12 x64 with the Python launcher first.'}
if(-not(Test-Path '.venv/Scripts/python.exe')){& py -3.12 -m venv .venv;if($LASTEXITCODE -ne 0){throw 'Python 3.12 is required.'}}
$deliveryPython=Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
& $deliveryPython -m pip install --upgrade pip
if($LASTEXITCODE -ne 0){throw 'pip setup failed'}
& $deliveryPython -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
if($LASTEXITCODE -ne 0){throw 'PyTorch CUDA installation failed'}
& $deliveryPython -m pip install -r requirements-windows.txt
if($LASTEXITCODE -ne 0){throw 'Runtime installation failed'}
& $deliveryPython app/prepare_models.py --model both
if($LASTEXITCODE -ne 0){throw 'TensorRT engine preparation failed'}
Write-Host 'Ready. Install FFmpeg if missing: winget install --id Gyan.FFmpeg --exact'
Write-Host 'Then reopen the window and use Run_NPS.cmd or Run_QUT.cmd.'
'''

if __name__ == '__main__':
    main()
