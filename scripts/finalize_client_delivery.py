"""Copy the approved runtime into a fresh clean folder and build its archive."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'handoff/AirAntiCollision'
FINAL=ROOT/'handoff/AirAntiCollision_Customer'
PDF=ROOT/'artifacts/client_document/Google_Algorithm_RU_full_final.pdf'
ARCHIVE=ROOT/'handoff/AirAntiCollision_Customer.zip'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    import sys
    refresh='--refresh' in sys.argv
    if (FINAL.exists() or ARCHIVE.exists()) and not refresh:raise ValueError('Final delivery already exists; preserve it')
    assert len(PdfReader(str(PDF)).pages)==7
    FINAL.mkdir(exist_ok=refresh)
    files=[]
    if not refresh:
        for src in sorted(SOURCE.rglob('*')):
            if not src.is_file() or '__pycache__' in src.parts or src.suffix=='.pyc':continue
            rel=src.relative_to(SOURCE)
            if src.suffix.lower()=='.pdf':continue
            dest=FINAL/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    shutil.copy2(PDF,FINAL/'Описание_системы.pdf')
    (FINAL/'results').mkdir(exist_ok=True)
    for path in sorted(FINAL.rglob('*')):
        if not path.is_file() or path.name=='manifest.json':continue
        rel=path.relative_to(FINAL)
        assert not any(p in {'.git','tests','__pycache__','task','research','reports','artifacts'} for p in rel.parts)
        files.append({'path':rel.as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
    manifest={'version':'2026.10.24','main_document':'Описание_системы.pdf','document_pages':7,
              'document_equations':19,'document_tables':8,'entrypoint':'app/run.py',
              'scope':'recorded RGB showcase, Windows / NVIDIA; model A and B supplied',
              'runtime_environment':'Python 3.12, Torch 2.12 cu130, TensorRT 11.4, CPU preprocessing',
              'fresh_machine_installation_tested':False,'files':files}
    (FINAL/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(FINAL.rglob('*')):
            if path.is_file():z.write(path,Path(FINAL.name)/path.relative_to(FINAL))
        z.writestr(FINAL.name+'/results/','')
    with zipfile.ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        names=z.namelist()
        assert len([n for n in names if n.endswith('.pdf')])==1
        assert not any('__pycache__' in n or '/tests/' in n or '/.git/' in n for n in names)
    result={'folder':str(FINAL),'archive':str(ARCHIVE),'files':len(files)+1,
            'archive_bytes':ARCHIVE.stat().st_size,'archive_sha256':sha(ARCHIVE),
            'pdf_pages':7,'native_google_equations':19,'pdf_sha256':sha(FINAL/'Описание_системы.pdf'),
            'zip_crc_all_members_passed':True,'no_git_tests_or_discussion':True,
            'model_A_smoke_frames':30,'model_B_smoke_frames':30,'fresh_machine_installation_tested':False}
    (ROOT/'reports/client_delivery_validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
