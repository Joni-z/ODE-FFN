"""Verify source/history/curation and seal one completed runtime exactly once."""
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
import time
from urllib.parse import unquote

import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--prepare-log',type=Path,required=True);args=parser.parse_args()
root=args.root.resolve();doc=Path(__file__).resolve().parent;repo=doc.parents[2]
assert not (root/'sealed-runtime.json').exists(), 'Never rewrite a sealed runtime'
sys.path.insert(0,str(root/'source'));import study as formal
formal.verify_inputs(root);settings,_,_,old,_,native,_=formal.setup(root)
assert settings['registration_commit']=='35cba881b7e7e1954f44eba20951d245e0c160e1'
load=formal.load;write=formal.write
sha=native.sha
receipt=load(root/'output/receipt.json');audit=load(root/'audit/audit.json')
assert receipt['status']==audit['status']=='PASS' and receipt['checkpoints']==24
assert load(root/'output/status.json')['state']=='COMPLETE'
assert audit['formal_files_unchanged']==56
for p in (root/'audit').iterdir():
    if p.is_file(): assert sha(p)==sha(doc/'results'/p.name),p.name
for p in (root/'source').iterdir():
    if p.is_file(): assert sha(p)==sha(doc/p.name),p.name

# Close the complete history-check scope separately; do not repeat the passed dense audit.
started=time.monotonic();history=[]
for entry in old['laws']:
    for family in formal.FAMILIES:
        for step in formal.CHECKPOINTS:
            model,folder=formal.restore(root,entry['law'],family,step)
            actual=native.legacy.visit_diagnostics(model,Path(entry['banks']))
            saved=load(folder/f'clocks_{step:07d}.json');assert len(actual)==len(saved)==6
            for a,b in zip(actual,saved):
                assert (a['layer'],a['visit'])==(b['layer'],b['visit'])
                assert a['activation'].keys()==b['activation'].keys()
                for k,v in a['activation'].items():
                    if isinstance(v,(float,int)): np.testing.assert_allclose(v,b['activation'][k],atol=1e-9,rtol=1e-8)
                    else: assert v==b['activation'][k]
            p_saved=load(folder/f'generator_{step:07d}.json')
            for (name,layer),p in zip(model.ode_layers(),p_saved):
                assert name==p['layer']
                singular=np.linalg.svd(layer.proj.weight.detach().double().numpy(),compute_uv=False)
                np.testing.assert_allclose(singular,p['P_singular_values'],atol=1e-8,rtol=1e-8)
                history.append(dict(law=entry['law'],family=family,step=step,layer=name))
assert len(history)==144
write(doc/'results/history-checks.json',dict(status='PASS',layer_P_and_clock_records=144,
    checkpoints=24,probe_rows=32,elapsed_seconds=time.monotonic()-started,
    scope='supplemental saved native P spectra and complete 32-row activation/clock histories'))

with (doc/'results/intervention-parts.csv').open() as stream: parts=list(csv.DictReader(stream))
assert len(parts)==72
frame=Path(load(Path(settings['parent'])/'settings.json')['parent']);basis=np.load(frame/'basis.npy')
for entry in old['laws']:
    with np.load(Path(entry['banks'])/'field.npz') as bank: truth=bank['oracle']
    for family in formal.FAMILIES:
        for step in formal.CHECKPOINTS:
            with np.load(root/'output/raw'/f'{entry["law"]}-{family}-{step:07d}-predictions.npz') as data:
                a=data['native'].astype(float);b=data['changed'].astype(float)
            e=a-truth;delta=b-a
            # Independent complete-QR normal coordinates avoid cancellation near the cold exact normal field.
            normal_basis=np.linalg.qr(basis,mode='complete')[0][:,2:]
            project=lambda v:(v,v@basis,v@normal_basis)
            ep,dp,ap=project(e),project(delta),project(a)
            old_sum=[float(np.sum(v*v)) for v in ep]
            new_sum=[float(np.sum((v+w)**2)) for v,w in zip(ep,dp)]
            d_sum=[float(np.sum(v*v)) for v in dp]
            cross=[float(np.sum(v*w)) for v,w in zip(ep,dp)]
            base=[float(np.sum(v*v)) for v in ap]
            for i,(scope,d) in enumerate(zip(('full','tangent','normal'),(128,2,126))):
                r=next(r for r in parts if r['law']==entry['law'] and r['family']==family and int(r['step'])==step and r['scope']==scope)
                for key,value in [('original_mse',old_sum[i]/(20000*d)),('changed_mse',new_sum[i]/(20000*d)),
                                  ('displacement_mse',d_sum[i]/(20000*d)),('cross_term',2*cross[i]/(20000*d))]:
                    np.testing.assert_allclose(value,float(r[key]),atol=1e-12,rtol=1e-10)
                np.testing.assert_allclose((new_sum[i]-old_sum[i])/old_sum[i],float(r['relative_risk_change']),atol=1e-10,rtol=1e-8)

link_count=0
for p in doc.glob('*.py'): ast.parse(p.read_text(),filename=str(p))
for p in list(doc.glob('*.md'))+[repo/'README.md',doc.parent/'README.md']:
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
        target=target.strip('<>').split('#')[0]
        if not target or '://' in target or target.startswith(('/','mailto:')): continue
        assert (p.parent/unquote(target)).exists(),(p,target)
        link_count+=1
protected=['reports/report1','synthetic','src','configs','scripts']
assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only',settings['registration_commit'],'--',*protected],text=True)
figures=list((doc/'results/figures').glob('*'));assert len(figures)==6

with (doc/'results/complete.csv').open() as stream: complete=list(csv.DictReader(stream))
with (root/'results.tsv').open('w') as stream:
    stream.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    for r in complete:
        stream.write(f"{settings['registration_commit'][:7]}\t{r['relative_displacement_rms']}\t{receipt['peak_sampled_rss_gib']:.3f}\tkeep\tverified frozen centred-action diagnostic {r['law']} {r['family']} step {r['step']}; no nominee\n")
shutil.copyfile(root/'results.tsv',doc/'results/results.tsv')
source=root/'curation-source';source.mkdir(exist_ok=False)
for p in doc.iterdir():
    if p.is_file(): shutil.copyfile(p,source/p.name)
shutil.copytree(doc/'results',root/'curated-results')
shutil.copyfile(args.prepare_log,root/'prepare.log')
write(root/'execution-record.json',dict(host='Mac mini',device='CPU',threads=1,
    prepare=dict(tool_session=59949,exit_code=0),
    preflight=dict(tool_session=5415,exit_code=0,pid=load(root/'preflight.json')['pid']),
    formal=dict(tool_session=9254,exit_code=0,pid=46900,elapsed_seconds=receipt['elapsed_seconds']),
    audit=dict(tool_session=61540,exit_code=0,pid=None,pid_note='Not retained in audit receipt.',elapsed_seconds=audit['elapsed_seconds']),
    curation=dict(tool_session=84042,exit_code=0),plot=dict(tool_session=12894,exit_code=0),
    new_training=False,new_sampler=False,new_nominee=False,
    reporting_addition='Tangent/normal split uses the same saved intervention predictions; no new selection gate.'))
formal.verify_inputs(root)
manifests=[Path(p) for p in old['prior_seals']]+[frame/'sealed-runtime.json',Path(settings['parent'])/'sealed-runtime.json']
prior_count=sum(len(load(p)) for p in manifests);assert prior_count==2706
files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'
             and p.name!='sealed-runtime.json')
write(root/'sealed-runtime.json',{str(p.relative_to(root)):sha(p) for p in files})
for name,digest in load(root/'sealed-runtime.json').items(): assert sha(root/name)==digest,name
verification=dict(status='PASS',runtime_files=len(files),runtime_manifest_sha256=sha(root/'sealed-runtime.json'),
    bindings_verified=len(load(root/'bindings.json')),prior_files_verified=prior_count,
    formal_exit_code=0,audit_exit_code=0,curated_audit_copies_exact=True,history_records_verified=144,
    reporting_decomposition_rows_verified=72,source_ast_checks=True,relative_markdown_links_checked=link_count,
    figures_visually_reviewed=3,protected_paths_unchanged=protected,new_training=False,new_architecture_nominated=False,
    raw_arrays_and_checkpoints_not_uploaded=True,runtime_bytes=sum(p.stat().st_size for p in files),
    curated_bytes=sum(p.stat().st_size for p in (doc/'results').rglob('*') if p.is_file()))
write(doc/'results/artifact-verification.json',verification);print(json.dumps(verification,indent=2))
