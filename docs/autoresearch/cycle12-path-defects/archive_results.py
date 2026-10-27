"""Verify and seal the completed diagnostic without rerunning any model."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--prepare-log', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
doc = Path(__file__).resolve().parent
repo = doc.parents[2]
assert not (root/'sealed-runtime.json').exists(), 'Never modify an existing seal'
sys.path.insert(0, str(root/'source'))
import path_study as formal


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''): h.update(block)
    return h.hexdigest()


def load(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def read(name):
    with (doc/'results'/name).open() as stream: return list(csv.DictReader(stream))


formal.verify_inputs(root)
settings = load(root/'settings.json')
parent = Path(settings['parent'])
assert settings['registration_commit']=='7a6c2dbb1160226f8a2873167eb9a5ae45eb7674'
receipt = load(root/'output/receipt.json')
assert load(root/'output/status.json')['state']=='COMPLETE'
assert receipt['status']=='PASS' and receipt['pid']==44962
audit = load(root/'audit/audit.json')
assert audit['status']=='PASS' and audit['formal_files_unchanged']==21
for name in ('endpoints.csv', 'steps.csv', 'bins.csv', 'telemetry.csv', 'receipt.json',
             'preflight.json', 'independent-checks.csv', 'normal-gains.csv', 'audit.json', 'results.tsv'):
    assert sha(root/'audit'/name)==sha(doc/'results'/name), name
for name in ('path_study.py','defect_core.py','verify_defects.py','protocol.md','theory.md'):
    assert sha(root/'source'/name)==sha(doc/name), name

endpoints = read('endpoints.csv');steps = read('steps.csv');bins = read('bins.csv')
assert (len(endpoints),len(steps),len(bins))==(12,1152,144)
assert {int(r['forced_normal_dimension']) for r in endpoints}=={63}
assert all(float(r['tangent_late_signed_share'])<.5 for r in endpoints)
assert all(float(r['normal_late_signed_share'])>.5 for r in endpoints)
assert sum(float(r['signed_endpoint_mse_contribution'])<0 for r in steps)==12
for row in endpoints:
    for scope in ('full','tangent','normal'):
        group=[r for r in bins if r['law']==row['law'] and r['family']==row['family'] and r['scope']==scope]
        assert len(group)==4
        np.testing.assert_allclose(sum(float(r['signed_share']) for r in group),1,atol=1e-12,rtol=0)

# Independently recalculate all reporting-only mean/SD/ratio precision rows.
basis=np.load(parent/'basis.npy');mc=read('conditional-mc.csv');checked=0
assert len(mc)==48
for record in endpoints:
    law,family=record['law'],record['family']
    with np.load(root/'output/raw'/f'{law}-{family}.npz') as data:
        e=data['bridges'][-1].astype(float)-data['bridges'][0];projector=data['forced_projector']
    t=e@basis;n=e-t@basis.T
    values={scope:np.einsum('nd,nd->n',v,v)/d for scope,v,d in
            [('full',e,128),('tangent',t,2),('normal',n,126)]}
    forced=np.einsum('nd,nd->n',e@projector,e@projector)/126
    for row in [r for r in mc if r['law']==law and r['family']==family]:
        assert int(row['count'])==2048
        if row['quantity']=='paired_endpoint_mse':
            v=values[row['scope']];mean=v.mean();sd=np.std(v,ddof=1);se=sd/np.sqrt(len(v))
            np.testing.assert_allclose(sd,float(row['source_sd']),rtol=1e-12,atol=1e-12)
        else:
            v=values['normal'];mean=forced.mean()/v.mean()
            se=np.std((forced-mean*v)/v.mean(),ddof=1)/np.sqrt(len(v))
            assert row['source_sd']==''
        expected=[mean,se,mean-1.959963984540054*se,mean+1.959963984540054*se]
        actual=[float(row[k]) for k in ('mean','standard_error','lower95','upper95')]
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)
        checked+=1
assert checked==48

# Syntax and local links; no production or submitted-report change is allowed.
for path in doc.glob('*.py'): ast.parse(path.read_text(), filename=str(path))
link_count=0
for path in list(doc.glob('*.md'))+[repo/'README.md',doc.parent/'README.md']:
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text()):
        target=target.strip('<>').split('#')[0]
        if not target or '://' in target or target.startswith(('mailto:','/')): continue
        assert (path.parent/unquote(target)).exists(),(path,target)
        link_count+=1
protected=['reports/report1','synthetic','src','configs','scripts']
unchanged=subprocess.check_output(['git','-C',str(repo),'diff','--name-only',
    settings['registration_commit'],'--',*protected],text=True)
assert unchanged=='',unchanged
figures=list((doc/'results/figures').glob('*'))
assert len(figures)==8 and {p.suffix for p in figures}=={'.png','.pdf'}

curation=root/'curation-source';curation.mkdir(exist_ok=False)
for path in doc.glob('*'):
    if path.is_file(): shutil.copyfile(path,curation/path.name)
archive=root/'curated-results';shutil.copytree(doc/'results',archive)
shutil.copyfile(args.prepare_log,root/'prepare.log')
write(root/'execution-record.json',dict(
    host='Mac mini',device='CPU',threads_per_process=1,new_training=False,
    observation='Terminal tool results consumed before sealing; no completed phase rerun.',
    prepare=dict(tool_session=90776,exit_code=0),
    preflight=dict(tool_session=66937,exit_code=0,elapsed_seconds=14.743716916069388),
    formal=dict(tool_session=40615,pid=44962,exit_code=0,elapsed_seconds=receipt['elapsed_seconds']),
    audit=dict(tool_session=96973,pid=None,pid_note='Not retained in audit receipt; no PID invented.',
               exit_code=0,elapsed_seconds=audit['elapsed_seconds']),
    curation=dict(tool_session=53080,exit_code=0),
    registration=settings['registration_commit'],
    reproduction_commands=[
        'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 '+
        '/Users/mr.z/ode-workspace-prep/.venv-test/bin/python '+cmd for cmd in [
        f'{root}/source/path_study.py preflight --root {root}',
        f'{root}/source/path_study.py run --root {root}',
        f'{doc}/audit_results.py --root {root} --output <fresh-audit-directory>']],
    command_note='Reproduction reference only: existing preflight/run/audit outputs must never be overwritten.'))

formal.verify_inputs(root)
prior_manifests=[Path(p) for p in load(parent/'settings.json')['prior_seals']]+[parent/'sealed-runtime.json']
prior_count=sum(len(load(p)) for p in prior_manifests)
assert prior_count==2629
files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'
             and p.name!='sealed-runtime.json')
manifest={str(p.relative_to(root)):sha(p) for p in files}
write(root/'sealed-runtime.json',manifest)
for name,digest in manifest.items(): assert sha(root/name)==digest,name
verification=dict(status='PASS',runtime_files=len(files),runtime_manifest_sha256=sha(root/'sealed-runtime.json'),
    bindings_verified=len(load(root/'bindings.json')),prior_files_verified=prior_count,
    formal_exit_code=0,audit_exit_code=0,curated_audit_copies_exact=True,
    reporting_precision_rows_verified=checked,source_ast_checks=True,relative_markdown_links_checked=link_count,
    figures_visually_reviewed=4,protected_paths_unchanged=protected,
    new_training=False,new_architecture_nominated=False,new_sampler_nominated=False,
    raw_arrays_and_checkpoints_not_uploaded=True,runtime_bytes=sum(p.stat().st_size for p in files),
    curated_bytes=sum(p.stat().st_size for p in (doc/'results').rglob('*') if p.is_file()))
write(doc/'results/artifact-verification.json',verification)
print(json.dumps(verification,indent=2))
