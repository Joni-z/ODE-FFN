"""Verify curated evidence and preserve the complete executed coefficient study."""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import unquote

import numpy as np

parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
args=parser.parse_args();root=args.root.resolve();doc=Path(__file__).resolve().parent;repo=doc.parents[2]
assert not (root/'sealed-runtime.json').exists(), 'Never modify an existing runtime seal'
sys.path.insert(0,str(root/'source'));import jet_study as formal
formal.verify_inputs(root);settings,_,native,_,_=formal.setup(root)
load,write,sha=formal.load,formal.write,native.sha
assert settings['registration_commit']==subprocess.check_output(
    ['git','-C',str(repo),'rev-parse','19af9a8'],text=True).strip()
status=load(root/'output/status.json');audit=load(root/'audit/audit.json');decision=load(root/'audit/decision.json')
assert status['state']=='COMPLETE' and status['completed']==status['total']==10
assert audit['status']=='PASS' and audit['formal_files_unchanged']==293
assert decision['decision']=='stop_this_coefficient_setting' and decision['nominee'] is None
for p in (root/'audit').iterdir():
    if p.is_file():assert sha(p)==sha(doc/'results'/p.name),p.name
for p in (root/'source').iterdir():
    if not p.is_file():continue
    destination=repo/'synthetic/coefficient_jet.py' if p.name=='coefficient_operator.py' else doc/p.name
    if destination.exists():assert sha(p)==sha(destination),p.name

def read(name):
    with (doc/'results'/name).open() as stream:return list(csv.DictReader(stream))
comparison=read('comparison.csv');wide=read('coefficient-singular-values.csv');telemetry=read('training-telemetry.csv')
assert len(comparison)==14 and len(wide)==192 and len(telemetry)==154
reference=Path(settings['geometry_reference']);spectral_checks=telemetry_checks=0
for record in comparison:
    law,family=record['law'],record['family'];fresh=record['new_training']=='True'
    folder=(root if fresh else reference)/'output/runs'/law/(family+'_clean')
    if fresh and family!='swiglu258':
        for step in (0,1000,2500,5000):
            for saved in load(folder/f'generator_{step:07d}.json'):
                row=next(r for r in wide if r['law']==law and r['family']==family and int(r['step'])==step and r['layer']==saved['layer'])
                for prefix,key in [('P','P_singular_values'),('reference_B','reference_B_singular_values'),('C','C_singular_values')]:
                    np.testing.assert_array_equal([float(row[f'{prefix}_sigma_{i:02d}']) for i in range(64)],saved[key])
                spectral_checks+=1
    for line in (folder/'metrics.jsonl').read_text().splitlines():
        saved=json.loads(line);row=next(r for r in telemetry if r['law']==law and r['family']==family and int(r['step'])==saved['step'])
        for key in ('train_loss','grad_norm','clip_fraction','oracle_mse','tangent_mse_per_dimension','normal_mse_per_dimension'):
            if key in saved:assert float(row[key])==saved[key],(law,family,key)
            else:assert row[key]=='',key
        telemetry_checks+=1
assert (spectral_checks,telemetry_checks)==(192,154)
with (Path(settings['parent_cycle11'])/'audit-v2/comparison.csv').open() as stream:
    context={r['law']:r for r in csv.DictReader(stream) if r['family']=='swiglu257'}
for row in read('historical-swi257-context.csv'):
    assert row['registered_cycle14_reference']=='False' and row['new_training_in_cycle14']=='False'
    for key in ('oracle_mse','tangent_mse_per_dimension','jacobian_error','sw2','tangent_sw2','parameters','dense_forward_macs'):
        assert row[key]==context[row['law']][key],key

metric=root.parent/'oct28-cycle14-coordinate-metric.json';assert load(metric)['status']=='PASS'
shutil.copyfile(metric,doc/'results/coordinate-metric.json')
for p in list(doc.glob('*.py'))+[repo/'synthetic/coefficient_jet.py']:ast.parse(p.read_text(),filename=str(p))
links=0
for p in list(doc.glob('*.md'))+[repo/'README.md',doc.parent/'README.md']:
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
        target=target.strip('<>').split('#')[0]
        if not target or '://' in target or target.startswith(('/','mailto:')):continue
        assert (p.parent/unquote(target)).exists(),(p,target)
        links+=1
protected=['reports/report1','src','configs','scripts','synthetic/models.py','synthetic/train.py',
           'synthetic/manifold_benchmark.py','synthetic/conditional_frame.py']
assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only','e608c9c','--',*protected],text=True)
assert len(list((doc/'results/figures').glob('*')))==8

curation=root/'curation-source';curation.mkdir(exist_ok=False)
for p in doc.iterdir():
    if p.is_file():shutil.copyfile(p,curation/p.name)
shutil.copytree(doc/'results',root/'curated-results')
for suffix in ('prepare.log','calibration.log','calibration.json'):
    shutil.copyfile(root.parent/f'oct28-cycle14-{suffix}',root/suffix)
calibration=root.parent/'oct28-cycle14-calibration-source'
shutil.copytree(calibration,root/'calibration-source')
for p in calibration.iterdir():assert sha(p)==sha(root/'calibration-source'/p.name)
launches=[json.loads(line) for line in (root/'output/launches.jsonl').read_text().splitlines()];assert len(launches)==10
for row in launches:
    terminal=load(root/'output/runs'/row['law']/row['arm']/'status.json')
    assert terminal['state']=='COMPLETE' and terminal['step']==5000 and terminal['pid']==row['pid']
write(root/'execution-record.json',dict(host='Mac mini',device='CPU',workers=2,threads_per_worker=1,
    calibration=dict(tool_session=93182,exit_code=0,pre_registration_source_files_preserved=3),
    prepare=dict(tool_session=96049,exit_code=0),preflight=dict(tool_session=33400,exit_code=0),
    formal=dict(tool_session=49051,pid=47790,exit_code=0,elapsed_seconds=status['elapsed_seconds']),
    children=[dict(**row,exit_code=0,exit_evidence='registered parent asserts each wait() code == 0; parent exit0 and terminal cell status verified') for row in launches],
    audit=dict(tool_session=82566,exit_code=0,elapsed_seconds=audit['elapsed_seconds']),
    plot=dict(tool_session=30163,exit_code=0,figures_visually_reviewed=4),
    coordinate_fixture=dict(exit_code=0,post_result=True,no_fitted_data=True),new_nominee=False,
    source_note='Pre-registration scalar detach/reference-label edits preserve model forward; original calibration sources/log are retained.'))
formal.verify_inputs(root)
prior_count=sum(len(load(Path(p))) for p in settings['prior_seals']);assert prior_count==2818
files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'
             and p.name!='sealed-runtime.json')
write(root/'sealed-runtime.json',{str(p.relative_to(root)):sha(p) for p in files})
for name,digest in load(root/'sealed-runtime.json').items():assert sha(root/name)==digest,name
result=dict(status='PASS',runtime_files=len(files),runtime_manifest_sha256=sha(root/'sealed-runtime.json'),
    bindings_verified=len(load(root/'bindings.json')),prior_files_verified=prior_count,formal_exit_code=0,audit_exit_code=0,
    curated_audit_copies_exact=True,derived_spectral_records_verified=spectral_checks,derived_training_rows_verified=telemetry_checks,
    historical_swi257_context_only=True,source_ast_checks=True,relative_markdown_links_checked=links,figures_visually_reviewed=4,
    unchanged_paths=protected,new_production_path='synthetic/coefficient_jet.py',new_architecture_nominated=False,
    raw_arrays_and_checkpoints_not_uploaded=True,runtime_bytes=sum(p.stat().st_size for p in files),
    curated_bytes=sum(p.stat().st_size for p in (doc/'results').rglob('*') if p.is_file()))
write(doc/'results/artifact-verification.json',result);print(json.dumps(result,indent=2))
