"""Validate full ordinary-control evidence and seal one completed runtime."""
from pathlib import Path
import argparse,ast,csv,json,re,shutil,subprocess,sys
from urllib.parse import unquote
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve();assert not (root/'sealed-runtime.json').exists()
doc=Path(__file__).resolve().parent;repo=doc.parents[2];sys.path.insert(0,str(root/'source'));import study
study.verify(root);assert study.load(root/'orchestrator/status.json')['state']=='COMPLETE'
assert study.load(root/'tool-exit.json')==dict(session=37739,exit_code=0)
phases=study.load(root/'orchestrator/launches.json');assert len(phases)==4 and all(r['exit_code']==0 for r in phases)
exits=study.load(root/'output/exits.json');assert len(exits)==3 and all(r['exit_code']==0 for r in exits)
training=study.load(root/'audit-training/result.json');features=study.load(root/'audit-features/result.json')
assert training['status']==features['status']=='PASS'
assert [training[k] for k in ('checkpoints','time_panels','matrix_records','generated_images','stream_batches')]==[33,60,72,3072,5000]
assert len(features['rows'])==9 and features['new_generated_feature_rows']==3072 and features['reference_feature_rows']==2048
for f in (root/'curation').rglob('*'):
    if f.is_file():assert study.sha(f)==study.sha(doc/'results'/f.relative_to(root/'curation'))
read=lambda name:list(csv.DictReader((doc/'results'/name).open()))
rows=read('comparison.csv');assert len(rows)==9
formal=study.load(root/'metric-output/result.json')['rows'];p=study.paths(root);costs={r['family']:r for r in study.load(root/'preflight.json')['arms']}
for row in rows:
    original=next(r for r in formal if r['family']==row['family']);audited=next(r for r in features['rows'] if r['family']==row['family'])
    for key in ('weighted_mse','clean_mse','fid','mean_term','covariance_term','out_of_range_fraction'):assert float(row[key])==float(original[key])
    assert abs(float(row['fid'])-audited['fid'])<1e-8 and int(row['steps'])==128 and int(row['generated'])==1024 and int(row['reference'])==2048
    arm=row['family']
    if arm in study.ARMS:
        path=root/'output/runs'/arm/'generated.npy'
        for key in ('parameters','FFN_forward_MACs','FFN_training_MACs'):assert int(row[key])==costs[arm][key]
    elif arm.endswith('_constant'):path=p['clock_parent']/'output/runs'/arm.removesuffix('_constant')/'generated.npy'
    else:path=p['resolution_parent']/'output/runs'/arm/'generated-128.npy'
    assert float(row['out_of_range_fraction'])==float(np.mean(np.abs(np.load(path))>1))
assert len(read('contrasts.csv'))==18 and len(read('curves.csv'))==77 and len(read('time-panels.csv'))==45 and len(read('per-image-distribution.csv'))==9
matrices=read('matrix-history.csv');assert len(matrices)==72
for arm in study.ARMS:
    for step in study.SNAPSHOTS:
        for record in study.load(root/'output/runs'/arm/f'matrices_{step:07d}.json'):
            public=next(r for r in matrices if r['family']==arm and int(r['step'])==step and r['layer']==record['layer'])
            assert public['shape']=='x'.join(map(str,record['shape']))
            for key,prefix in [('singular_values','sigma'),('eigen_real','eigen_real'),('eigen_imag','eigen_imag')]:
                for index,value in enumerate(record.get(key,[])):assert float(public[f'{prefix}_{index:02d}'])==value
                if key not in record:assert all(not value for k,value in public.items() if k.startswith(prefix+'_'))
            for key in ('frobenius','sigma_max','singular_CV'):assert float(public[key])==record[key]
telemetry=read('training-telemetry.csv');assert len(telemetry)==33
for arm in study.ARMS:
    for line in (root/'output/runs'/arm/'metrics.jsonl').read_text().splitlines():
        native=json.loads(line);public=next(r for r in telemetry if r['family']==arm and int(r['step'])==native['step'])
        for key,value in native.items():assert float(public[key])==value
assert study.load(doc/'results/decision.json')['new_ODE_architecture_nominated'] is False
assert len(list((doc/'results/figures').glob('*')))==8
for name in ('analysis-report.md','stats-appendix.md','figure-catalog.md'):assert (doc/name).exists()
protected=['src','synthetic','configs','scripts','reports/report1']
assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only','98730e0','--',*protected],text=True)
links=0
for f in [*doc.glob('*.md'),repo/'README.md',doc.parent/'README.md']:
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',f.read_text()):
        target=target.strip('<>').split('#')[0]
        if not target or '://' in target or target.startswith(('/','mailto:')):continue
        assert (f.parent/unquote(target)).exists(),(f,target);links+=1
for f in doc.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
study.write(doc/'results/artifact-verification.json',dict(status='PASS',new_cold_models=3,reused_endpoints=6,checkpoints=33,new_time_panels=60,matrix_histories=72,stream_batches=5000,new_generated_images_replayed=3072,
    new_generated_feature_rows=3072,reference_feature_rows=2048,independently_checked_FIDs=9,curated_curve_rows=77,curated_final_time_panels=45,
    figures_visually_reviewed=4,source_bindings=len(study.load(root/'bindings.json')),local_links=links,unchanged_scopes=protected,new_ODE_architecture_nominated=False))
shutil.copytree(doc/'results',root/'curated-results');closure=root/'closure-source';closure.mkdir()
for f in doc.iterdir():
    if f.is_file():shutil.copyfile(f,closure/f.name)
for name,source in [('preflight.log','image-silu-preflight.log'),('controller.log','image-silu-controller.log')]:shutil.copyfile(Path('/Users/mr.z/ode-workspace-prep')/source,root/name)
study.verify(root);files=sorted(f for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name!='sealed-runtime.json')
study.write(root/'sealed-runtime.json',{str(f.relative_to(root)):study.sha(f) for f in files})
for f,digest in study.load(root/'sealed-runtime.json').items():assert study.sha(root/f)==digest
print(json.dumps(dict(status='SEALED_AND_VERIFIED',files=len(files),seal_sha256=study.sha(root/'sealed-runtime.json'),private_bytes=sum(f.stat().st_size for f in files),curated_bytes=sum(f.stat().st_size for f in (doc/'results').rglob('*') if f.is_file()))))
