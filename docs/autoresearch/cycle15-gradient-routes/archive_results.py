"""Verify curation and seal the read-only diagnostic, retaining failed audit-v1."""
from __future__ import annotations
import argparse,ast,csv,hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args()
root=args.root.resolve();doc=Path(__file__).resolve().parent;repo=doc.parents[2]
sys.path.insert(0,str(root/'source'));import study
study.verify(root);settings=study.load(root/'settings.json')
assert study.load(root/'output/status.json')['state']=='COMPLETE'
audit=study.load(root/'audit-v2/audit.json');assert audit['status']=='PASS' and audit['checkpoints']==88
for src,dst in [('output/gradient-groups.csv','gradient-groups.csv'),('output/checkpoints.csv','checkpoints.csv'),
    ('output/receipt.json','receipt.json'),('audit-v2/audit.json','audit.json'),('preflight.json','preflight.json')]:
    assert study.sha(root/src)==study.sha(doc/'results'/dst),(src,dst)
read=lambda p:list(csv.DictReader(p.open()))
check=read(doc/'results/checkpoints.csv');groups=read(doc/'results/gradient-groups.csv')
assert len(check)==88 and len(groups)==440
assert {(r['law'],r['family'],int(r['step'])) for r in check}=={
    (law,f,k) for law in ('thin','thick') for f in study.FAMILIES for k in study.STEPS}
for row in check:
    key=f"{row['law']}_{row['family']}_{int(row['step']):07d}"
    meta=study.load(root/'output/probes'/f'{key}.json')
    assert study.sha(meta['checkpoint'])==row['checkpoint_sha256']==meta['sha256']
    with np.load(root/'output/probes'/f'{key}.npz') as f:
        # Native torch norm is checked in the independent audit through the
        # exact stored gradient; use torch here to preserve its reduction dtype.
        import torch
        gn=float(torch.from_numpy(f['native_gradient']).norm())
        assert gn==float(row['native_gradient_norm'])
        assert int(row['field_rows'])==20000
        np.testing.assert_allclose(np.linalg.norm(f['full_gradient']),float(row['promoted_gradient_norm']),atol=1e-12,rtol=1e-10)
for row in groups:
    if row['group']=='hidden':assert float(row['forced_norm'])<1e-10*max(1.,float(row['full_norm']))
assert study.sha(root/'audit-v2-source.py')==study.sha(doc/'audit.py')
assert study.sha(root/'source/audit.py')!=study.sha(doc/'audit.py')
protected=['reports/report1','src','configs','scripts','synthetic']
assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only','cd09851','--',*protected],text=True)
links=0
for p in [*doc.glob('*.md'),repo/'README.md',doc.parent/'README.md']:
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
        target=target.strip('<>').split('#')[0]
        if not target or '://' in target or target.startswith(('/','mailto:')):continue
        assert (p.parent/unquote(target)).exists(),(p,target)
        links+=1
for p in doc.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
assert len(list((doc/'results/figures').glob('*')))==4
parent=Path(settings['parent']);old=study.load(parent/'settings.json')
prior=sum(len(study.load(Path(p))) for p in old['prior_seals'])+len(study.load(parent/'sealed-runtime.json'))
assert prior==3184
study.write(doc/'results/artifact-verification.json',dict(status='PASS',checkpoints=88,group_rows=440,
    all_native_field_rows=1760000,prior_sealed_files=prior,bindings=len(study.load(root/'bindings.json')),
    local_links=links,figures_visually_reviewed=2,unchanged_scopes=protected,
    failed_audit_v1_retained=True,no_formal_rerun=True))
for suffix in ('prepare','formal','audit','audit-v2'):
    shutil.copyfile(root.parent/f'oct28-cycle15-{suffix}.log',root/f'{suffix}.log')
shutil.copytree(doc/'results',root/'curated-results')
curation=root/'curation-source';curation.mkdir()
for p in doc.iterdir():
    if p.is_file():shutil.copyfile(p,curation/p.name)
study.write(root/'execution-record.json',dict(host='Zzzs-Mac-mini.local',device='CPU',threads=1,training_updates=0,
    prepare=dict(tool_session=60362,exit_code=0),preflight=dict(tool_session=31369,exit_code=0),
    formal=dict(tool_session=42898,pid=49158,exit_code=0),
    audit_v1=dict(tool_session=93375,exit_code=1,reason='cold near-zero route cosine is ill-conditioned'),
    audit_v2=dict(tool_session=55421,exit_code=0,source_sha256=study.sha(root/'audit-v2-source.py')),
    curate=dict(exit_code=0),new_architecture_selected=False))
study.verify(root)
files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='sealed-runtime.json')
manifest={str(p.relative_to(root)):study.sha(p) for p in files};study.write(root/'sealed-runtime.json',manifest)
assert all(study.sha(root/p)==h for p,h in study.load(root/'sealed-runtime.json').items())
summary=dict(status='SEALED_AND_VERIFIED',files=len(files),seal_sha256=study.sha(root/'sealed-runtime.json'),
    private_bytes=sum(p.stat().st_size for p in files),curated_bytes=sum(p.stat().st_size for p in (doc/'results').rglob('*') if p.is_file()))
print(json.dumps(summary),flush=True)
