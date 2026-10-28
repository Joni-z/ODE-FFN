"""Verify exact public clock-attribution evidence and seal its runtime."""
from pathlib import Path
import argparse,ast,csv,json,re,shutil,subprocess,sys
from urllib.parse import unquote
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
assert not (root/'sealed-runtime.json').exists()
doc=Path(__file__).resolve().parent;repo=doc.parents[2];sys.path.insert(0,str(root/'source'));import study
study.verify(root,True);assert study.load(root/'orchestrator/status.json')['state']=='COMPLETE'
assert study.load(root/'tool-exit.json')==dict(session=51166,exit_code=0)
phases=study.load(root/'orchestrator/launches.json');assert len(phases)==4 and all(x['exit_code']==0 for x in phases)
for phase,count in [('sampling',2),('features',6)]:
    audit=study.load(root/f'audit-{phase}/result.json');assert audit['status']=='PASS' and len(audit['rows'])==count
for f in (root/'curation').rglob('*'):
    if f.is_file():assert study.sha(f)==study.sha(doc/'results'/f.relative_to(root/'curation'))
read=lambda name:list(csv.DictReader((doc/'results'/name).open()))
rows=read('comparison.csv');assert len(rows)==6
formal=study.load(root/'metric-metadata-v2/result.json')['rows'];independent=study.load(root/'audit-features/result.json')['rows']
for row in rows:
    native=next(r for r in formal if r['family']==row['family']);audit=next(r for r in independent if r['family']==row['family'])
    for key in ('weighted_mse','clean_mse','fid','mean_term','covariance_term','out_of_range_fraction'):assert float(row[key])==float(native[key])
    assert abs(float(row['fid'])-audit['fid'])<1e-8 and int(row['generated'])==1024 and int(row['reference'])==2048 and int(row['steps'])==128
correction=study.load(root/'metric-metadata-v2/receipt.json');assert study.sha(root/'metric-metadata-v2/correct_metadata.py')==correction['source_sha256'];assert correction['status']=='PASS' and correction['unchanged_FIDs'] and correction['unchanged_losses']
for file,digest in correction['unchanged_raw_bindings'].items():assert study.sha(file)==digest
for row in rows:
    path=root/'output/runs'/row['family'].removesuffix('_constant')/'generated.npy' if row['family'].endswith('_constant') else study.paths(root)['resolution_parent']/'output/runs'/row['family']/'generated-128.npy'
    assert float(row['out_of_range_fraction'])==float(np.mean(np.abs(np.load(path))>1))
assert len(read('time-panels.csv'))==30 and len(read('calibration.csv'))==12 and len(read('contrasts.csv'))==2
for row in read('calibration.csv'):
    original=study.load(root/'calibration.json')[row['family']][row['layer']]
    for key in ('value','native_mean','native_SD','native_min','native_max','count'):assert float(row[key])==original[key]
assert study.load(doc/'results/decision.json')['new_architecture_nominated'] is False
assert len(list((doc/'results/figures').glob('*')))==6
for name in ('analysis-report.md','stats-appendix.md','figure-catalog.md'):assert (doc/name).exists()
protected=['src','synthetic','configs','scripts','reports/report1']
assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only','98730e0','--',*protected],text=True)
links=0
for p in [*doc.glob('*.md'),repo/'README.md',doc.parent/'README.md']:
    for t in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
        t=t.strip('<>').split('#')[0]
        if not t or '://' in t or t.startswith(('/','mailto:')):continue
        assert (p.parent/unquote(t)).exists(),(p,t);links+=1
for p in doc.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
failed=root.parent/'oct28-cycle19';assert not (failed/'output').exists()
assert study.load(failed/'failed-preflight.json')['exit_code']==1
for p,h in study.load(failed/'bindings.json').items():assert study.sha(p)==h
study.write(root/'failed-preparation-manifest.json',{str(p):study.sha(p) for p in failed.rglob('*') if p.is_file() and '__pycache__' not in p.parts})
study.write(doc/'results/artifact-verification.json',dict(status='PASS',endpoints=6,new_interventions=2,reused_native_controls=4,source_bindings=len(study.load(root/'bindings.json')),calibration_scalars=12,
    calibration_images=64,calibration_tokens_per_clock=4096,validation_images_per_model=1024,sampled_images_per_model=1024,reference_images=2048,
    new_raw_samples_replayed=2048,new_generated_feature_images_replayed=2048,reference_feature_images_replayed=2048,FID_scores_independently_checked=6,
    time_panels=30,figures_visually_reviewed=3,local_links=links,unchanged_scopes=protected,new_architecture_nominated=False))
shutil.copytree(doc/'results',root/'curated-results');closure=root/'closure-source';closure.mkdir()
for p in doc.iterdir():
    if p.is_file():shutil.copyfile(p,closure/p.name)
for name,source in [('first-preflight.log','image-clocks-preflight.log'),('preflight-v2.log','image-clocks-preflight-v2.log'),('controller.log','image-clocks-controller.log')]:shutil.copyfile(Path('/Users/mr.z/ode-workspace-prep')/source,root/name)
study.verify(root,True);files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='sealed-runtime.json')
study.write(root/'sealed-runtime.json',{str(p.relative_to(root)):study.sha(p) for p in files})
for p,h in study.load(root/'sealed-runtime.json').items():assert study.sha(root/p)==h
print(json.dumps(dict(status='SEALED_AND_VERIFIED',files=len(files),seal_sha256=study.sha(root/'sealed-runtime.json'),private_bytes=sum(p.stat().st_size for p in files),curated_bytes=sum(p.stat().st_size for p in (doc/'results').rglob('*') if p.is_file()))))
