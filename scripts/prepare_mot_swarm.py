"""Fetch three predeclared real multi-UAV sequences, verified by Git blob SHA.

Selection before running trackers: first test sequence 02, paper illustrated
failure cases 06 and 28. No tracker/detector training or label-based filtering.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import urllib.request,json,hashlib,configparser
ROOT=Path(__file__).resolve().parents[1]
repo='UAVSwarm/UAVSwarm-dataset'
def read(url):
    with urllib.request.urlopen(url,timeout=60) as r:return r.read()
branch=json.loads(read(f'https://api.github.com/repos/{repo}'))['default_branch']
commit=json.loads(read(f'https://api.github.com/repos/{repo}/commits/{branch}'))['sha']
tree=json.loads(read(f'https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1'))['tree']
prefixes=[f'test/UAVSwarm-{n}/' for n in ['02','06','28']]
files=[x for x in tree if x['type']=='blob' and any(x['path'].startswith(p) for p in prefixes) and '/det/' not in x['path']]
out=ROOT/'data/mot_swarm';out.mkdir(parents=True,exist_ok=True)
def fetch(x):
    dest=out/Path(x['path']).relative_to('test');dest.parent.mkdir(parents=True,exist_ok=True)
    b=dest.read_bytes() if dest.exists() else read(f'https://raw.githubusercontent.com/{repo}/{commit}/{x["path"]}')
    if hashlib.sha1(f'blob {len(b)}\0'.encode()+b).hexdigest()!=x['sha']:raise ValueError(x['path'])
    if not dest.exists():dest.write_bytes(b)
    return dict(path=str(dest.relative_to(ROOT)),git_blob=x['sha'],sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
with ThreadPoolExecutor(max_workers=12) as pool:records=list(pool.map(fetch,files))
manifest=dict(repository=f'https://github.com/{repo}',commit=commit,selection=['02','06','28'],files=records,
              selection_reason='First test sequence and two paper-illustrated tracking failure sequences; selected before any model results')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Verified',len(records),'files',sum(x['bytes'] for x in records),'bytes',flush=True)
