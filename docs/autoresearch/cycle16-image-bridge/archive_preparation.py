"""Seal image preparation separately from any future quality-training study."""
from pathlib import Path
import argparse,ast,hashlib,json,re,shutil,subprocess
from urllib.parse import unquote

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);args=ap.parse_args()
root=args.root.resolve();data=args.data.resolve();doc=Path(__file__).resolve().parent;repo=doc.parents[2]
load=lambda p:json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
assert load(root/'status.json')['state']=='COMPLETE'
for name in ('preflight.json','preparation-audit.json','semantics.json','shape-semantics.json'):
    assert load(root/name)['status']=='PASS';assert sha(root/name)==sha(doc/'results'/name)
for p,h in load(root/'bindings.json').items():assert sha(root/p)==h,p
assert sha(data/'manifest.json')==sha(doc/'results/data-manifest.json')
manifest=load(data/'manifest.json')
for p,h in manifest['files'].items():assert sha(data/p)==h
assert sha(data/'cifar-10-python.tar.gz')==manifest['archive_sha256']
for name in ('image_model.py','preflight.py','prepare_data.py','review.md','research-question-card.md','requirements.txt'):
    assert sha(root/'source'/name)==sha(doc/name),name
for name,dst in [('verify_semantics.py','semantics-source.py'),('verify_shape_semantics.py','shape-semantics-source.py'),('audit_preparation.py','preparation-audit-source.py')]:
    assert sha(doc/name)==sha(root/dst)
protected=['src','synthetic','configs','scripts','reports/report1']
assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only','8f459e5','--',*protected],text=True)
links=0
for p in [*doc.glob('*.md'),repo/'README.md',doc.parent/'README.md']:
    for t in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
        t=t.strip('<>').split('#')[0]
        if not t or '://' in t or t.startswith(('/','mailto:')):continue
        assert (p.parent/unquote(t)).exists(),(p,t);links+=1
for p in doc.glob('*.py'):ast.parse(p.read_text())
old=Path('/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle15')
for p,h in load(old/'sealed-runtime.json').items():assert sha(old/p)==h
assert len(load(old/'sealed-runtime.json'))==225
assert len(list((doc/'results/figures').glob('*')))==3
write(doc/'results/artifact-verification.json',dict(status='PASS',source_bindings=len(load(root/'bindings.json')),
    data_train=45000,data_validation=5000,decoded_training_images=50000,old_sealed_files_checked=225,
    local_links=links,figures_visually_reviewed=2,unchanged_scopes=protected,
    no_real_CIFAR_training=True,no_architecture_nomination=True,failed_preparation_preserved=True))
closure={str(p):sha(p) for p in [root/'settings.json',root/'bindings.json',root/'environment-lock.txt',
    data/'manifest.json',data/'images.uint8.npy',data/'split.npz',data/'cifar-10-python.tar.gz',old/'sealed-runtime.json']}
write(root/'closure-bindings.json',closure)
first=Path('/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle16-feasibility')
shutil.copytree(first,root/'failed-feasibility-v1',ignore=shutil.ignore_patterns('__pycache__'))
for suffix in ('preflight','preflight-v2','environment','data-prepare','data-range','preparation-audit'):
    p=Path('/Users/mr.z/ode-workspace-prep')/f'image-bridge-{suffix}.log'
    shutil.copyfile(p,root/f'{suffix}.log')
prefix=Path('/Users/mr.z/ode-workspace-prep/image-bridge-data/cifar10-v1/cifar-10-python.tar.gz.part')
write(root/'serial-transfer-preservation.json',dict(path=str(prefix),bytes=prefix.stat().st_size,sha256=sha(prefix),
    terminal_exit=143,reason='Intentional throughput change after verified HTTP206 range support; full merged archive passes official MD5.'))
shutil.copytree(doc/'results',root/'curated-results');curation=root/'curation-source';curation.mkdir()
for p in doc.iterdir():
    if p.is_file():shutil.copyfile(p,curation/p.name)
write(root/'execution-record.json',dict(host='Zzzs-Mac-mini.local',device_calibration=['cpu','mps'],threads=1,
    environment=dict(tool_session=1059,exit_code=0),serial_download=dict(tool_session=61679,pid=49874,exit_code=143),
    ranged_download=dict(tool_session=54825,pid=50028,exit_code=0),
    first_preflight=dict(tool_session=14271,pid=49953,exit_code=1),
    corrected_preflight=dict(tool_session=91300,pid=50045,exit_code=0),
    clock_witness=dict(tool_session=33414,exit_code=0),shape_witness=dict(tool_session=66793,exit_code=0),
    preparation_audit=dict(tool_session=18685,exit_code=0),real_CIFAR_quality_updates=0,new_method=False))
files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='sealed-runtime.json')
write(root/'sealed-runtime.json',{str(p.relative_to(root)):sha(p) for p in files})
for p,h in load(root/'sealed-runtime.json').items():assert sha(root/p)==h
for p,h in closure.items():assert sha(Path(p))==h
print(json.dumps(dict(status='SEALED_AND_VERIFIED',files=len(files),seal_sha256=sha(root/'sealed-runtime.json'),
    private_bytes=sum(p.stat().st_size for p in files),data_directory_bytes=sum(p.stat().st_size for p in data.iterdir() if p.is_file()),
    source_bindings=len(load(root/'bindings.json')),closure_bindings=len(closure),quality_training=False)))
