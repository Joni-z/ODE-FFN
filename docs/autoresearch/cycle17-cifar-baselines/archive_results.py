"""Verify public image evidence and seal this completed baseline study."""
from pathlib import Path
import argparse,ast,csv,hashlib,json,re,shutil,subprocess,sys
from urllib.parse import unquote
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
assert not (root/'sealed-runtime.json').exists(), 'Preserve an already sealed runtime'
doc=Path(__file__).resolve().parent;repo=doc.parents[2];sys.path.insert(0,str(root/'source'));import study
study.verify(root,True)
assert study.load(root/'postprocess-v2/status.json')['state']=='COMPLETE'
train=study.load(root/'audit-training/audit.json');features=study.load(root/'audit-features-v2/audit.json')
assert train['status']==features['status']=='PASS'
assert (train['checkpoints'],train['panels'],train['P_records'],train['clock_rows'],train['sample_batches'],train['stream_batches'])==(44,80,48,240,4,5000)
assert (features['reference_images'],features['generated_images'])==(2048,4096)
for p,h in study.load(root/'postprocess-bindings.json').items():assert study.sha(p)==h
for p,h in study.load(root/'audit-features-v2-bindings.json').items():assert study.sha(p)==h
for p in (root/'curation').rglob('*'):
    if p.is_file():assert study.sha(p)==study.sha(doc/'results'/p.relative_to(root/'curation'))
read=lambda p:list(csv.DictReader(p.open()))
rows=read(doc/'results/comparison.csv');assert len(rows)==4 and {r['family'] for r in rows}==set(study.ARMS)
assert len(read(doc/'results/curves.csv'))==44
assert len(read(doc/'results/P-history.csv'))==48
assert len(read(doc/'results/clocks.csv'))==240
assert len(read(doc/'results/training-telemetry.csv'))==44
assert len(read(doc/'results/per-image-distribution.csv'))==4
assert len(read(doc/'results/FID-accounting.csv'))==4
assert len(read(doc/'results/first-case.csv'))==20
native=read(root/'audit-training/comparison.csv');fids={r['family']:r['fid'] for r in features['results']}
for row in rows:
    original=next(r for r in native if r['family']==row['family'])
    for key in ('weighted_mse','clean_mse','prediction_rms','out_of_range_fraction','cell_seconds'):assert float(row[key])==float(original[key])
    assert float(row['fid'])==fids[row['family']] and int(row['FID_generated'])==1024 and int(row['FID_reference'])==2048
# Independently check every public spectral/clock/telemetry row against the
# already replayed native records rather than trusting only aggregate counts.
P=read(doc/'results/P-history.csv');clocks=read(doc/'results/clocks.csv');telemetry=read(doc/'results/training-telemetry.csv')
for arm in study.ARMS:
    folder=root/'output/runs'/arm
    for step in study.SNAPSHOTS:
        for r in study.load(folder/f'P_{step:07d}.json'):
            got=next(g for g in P if g['family']==arm and int(g['step'])==step and g['layer']==r['layer'])
            for prefix,key in [('sigma','singular_values'),('eigen_real','eigenvalues_real'),('eigen_imag','eigenvalues_imag')]:
                np.testing.assert_array_equal([float(got[f'{prefix}_{i:02d}']) for i in range(64)],r[key])
        for r in study.load(folder/f'clocks_{step:07d}.json'):
            got=next(g for g in clocks if g['family']==arm and int(g['step'])==step and g['layer']==r['layer'] and float(g['time'])==r['time'])
            for key,value in r.items():
                if key not in ('layer','probes'):assert float(got[key])==value
    for line in (folder/'metrics.jsonl').read_text().splitlines():
        r=json.loads(line);got=next(g for g in telemetry if g['family']==arm and int(g['step'])==r['step'])
        for key in ('train_loss','grad_norm','clip_fraction','weighted_mse','clean_mse'):
            if key in r:assert float(got[key])==r[key]
            else:assert got[key]==''
protected=['src','synthetic','configs','scripts','reports/report1']
assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only','98730e0','--',*protected],text=True)
links=0
for p in [*doc.glob('*.md'),repo/'README.md',doc.parent/'README.md']:
    for t in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
        t=t.strip('<>').split('#')[0]
        if not t or '://' in t or t.startswith(('/','mailto:')):continue
        assert (p.parent/unquote(t)).exists(),(p,t);links+=1
for p in doc.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
assert len(list((doc/'results/figures').glob('*')))==10
study.write(doc/'results/artifact-verification.json',dict(status='PASS',checkpoints=44,panels=80,P_records=48,clock_rows=240,
    stream_batches=5000,reference_feature_images=2048,generated_feature_images=4096,
    bindings=len(study.load(root/'bindings.json')),postprocess_bindings=len(study.load(root/'postprocess-bindings.json')),
    figures_visually_reviewed=5,local_links=links,unchanged_scopes=protected,new_architecture_nominated=False))
for suffix,name in [('prepare','image-baseline-v2-prepare.log'),('reference-reuse','image-baseline-v2-reference-reuse.log'),
    ('training','image-baseline-v2-training.log'),('postprocess','image-baseline-v2-postprocess.log'),('feature-audit-v2','image-baseline-v2-feature-audit-v2.log')]:
    shutil.copyfile(Path('/Users/mr.z/ode-workspace-prep')/name,root/f'{suffix}.log')
first=root.parent/'oct28-cycle17';assert not (first/'output').exists()
study.write(root/'initial-preparation-manifest.json',{str(p):study.sha(p) for p in first.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
shutil.copytree(Path('/Users/mr.z/ode-workspace-prep/image-metrics-attempt01'),root/'failed-metrics-preparation')
shutil.copyfile('/Users/mr.z/ode-workspace-prep/image-trainer-doc-pass.json',root/'trainer-doc-pass.json')
shutil.copytree(doc/'results',root/'curated-results');source=root/'curation-source';source.mkdir()
for p in doc.iterdir():
    if p.is_file():shutil.copyfile(p,source/p.name)
study.write(root/'execution-record.json',dict(host='Zzzs-Mac-mini.local',training_backend='cpu',workers=2,threads_per_worker=1,feature_backend='mps',
    metrics_environment_install=dict(first_session=6441,exit_code=0,first_witness_session=32973,witness_exit=1,reason='SciPy1.15.3 Mach-O load error'),
    metrics_environment_v2=dict(install_session=53228,exit_code=0,witness_session=93725,witness_exit=0),
    metric_fixture=dict(session=84046,exit_code=0),initial_prepare=dict(session=12879,exit_code=0),initial_reference=dict(session=64923,exit_code=0),
    active_prepare=dict(session=47157,exit_code=0),reference_reuse=dict(session=92612,exit_code=0),
    preflight=dict(executor='fresh image_trainer_doc_pass agent',exit_code=0,receipt='trainer-doc-pass.json'),
    training=dict(tool_session=39868,pid=51294,exit_code=0,children=study.load(root/'output/launches.json')),
    postprocess=dict(tool_session=9920,exit_code=1,reason='Feature audit path type error before numerical work; preserved',phases=[study.load(p) for p in sorted((root/'postprocess').glob('*-launches.json'))]),
    feature_audit_v2=dict(tool_session=68967,exit_code=0,source='audit-features-v2-source.py',path_fix_only=True),
    no_training_restart=True,new_method=False))
study.verify(root,True)
files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='sealed-runtime.json')
study.write(root/'sealed-runtime.json',{str(p.relative_to(root)):study.sha(p) for p in files})
for p,h in study.load(root/'sealed-runtime.json').items():assert study.sha(root/p)==h
print(json.dumps(dict(status='SEALED_AND_VERIFIED',files=len(files),seal_sha256=study.sha(root/'sealed-runtime.json'),
    private_bytes=sum(p.stat().st_size for p in files),curated_bytes=sum(p.stat().st_size for p in (doc/'results').rglob('*') if p.is_file()))))
