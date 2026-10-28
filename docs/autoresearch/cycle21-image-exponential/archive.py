"""Verify complete frozen-map evidence, public copies and immutable parents."""
from pathlib import Path
import argparse,ast,csv,json,re,shutil,subprocess,sys
from urllib.parse import unquote
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args();root=a.root.resolve()
doc=Path(__file__).resolve().parent;repo=doc.parents[2];sys.path.insert(0,str(root/'source'));import study
assert not (root/'sealed-runtime.json').exists();study.verify(root,True)
assert study.load(root/'orchestrator/status.json')['state']=='COMPLETE'
receipt=study.load(root/'tool-exit.json');assert receipt['exit_code']==0 and isinstance(receipt['session'],int)
phases=study.load(root/'orchestrator/launches.json');children=study.load(root/'output/launches.json')
assert len(phases)==len(children)==4 and all(r['exit_code']==0 for r in [*phases,*children])
sampling=study.load(root/'audit-sampling/result.json');features=study.load(root/'audit-features/result.json')
assert sampling['status']==features['status']=='PASS'
assert sampling['operator_records']==48 and sampling['new_generated_images']==4096 and sampling['time_panels']==20
assert features['new_generated_feature_rows']==4096 and features['reference_feature_rows']==2048 and len(features['rows'])==13
assert study.load(root/'audit-sampling/operator-check.json')['status']=='PASS'
for f in (root/'curation').rglob('*'):
    if f.is_file():assert study.sha(f)==study.sha(doc/'results'/f.relative_to(root/'curation'))
read=lambda name:list(csv.DictReader((doc/'results'/name).open()))
rows=read('comparison.csv');assert len(rows)==13;formal=study.load(root/'metric-output/result.json')['rows'];p=study.paths(root)
for r in rows:
    arm=r['family'];original=next(x for x in formal if x['family']==arm);audited=next(x for x in features['rows'] if x['family']==arm)
    for key in ('weighted_mse','clean_mse','fid','mean_term','covariance_term','out_of_range_fraction'):assert float(r[key])==float(original[key])
    assert abs(float(r['fid'])-audited['fid'])<1e-8
    assert int(r['steps'])==128 and int(r['generated'])==1024 and int(r['reference'])==2048
    if arm in study.ARMS:path=root/'output/runs'/arm/'generated.npy'
    elif arm.startswith('silu'):path=p['ordinary_parent']/'output/runs'/arm/'generated.npy'
    elif arm.endswith('_constant'):path=p['clock_parent']/'output/runs'/arm.removesuffix('_constant')/'generated.npy'
    else:path=p['resolution_parent']/'output/runs'/arm/'generated-128.npy'
    assert float(r['out_of_range_fraction'])==float(np.mean(np.abs(np.load(path))>1))
history=read('operator-history.csv');assert len(history)==48
for actual in study.load(root/'operator-history.json'):
    public=next(r for r in history if r['family']==actual['family'] and int(r['step'])==actual['step'] and r['layer']==actual['layer'])
    for key,value in actual.items():
        if isinstance(value,list):
            prefix={'P_singular_values':'sigma','P_eigen_real':'eigen_real','P_eigen_imag':'eigen_imag'}[key]
            for i,v in enumerate(value):assert float(public[f'{prefix}_{i:02d}'])==v
        elif isinstance(value,(int,float)):assert float(public[key])==value
        else:assert public[key]==value
assert len(read('time-panels.csv'))==65 and len(read('contrasts.csv'))==6
assert study.load(doc/'results/decision.json')['new_ODE_architecture_nominated'] is False
assert len(list((doc/'results/figures').glob('*')))==6
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
study.write(doc/'results/artifact-verification.json',dict(status='PASS',new_training=False,new_interventions=4,reused_endpoints=9,new_generated_images_replayed=4096,independent_operator_records=48,new_time_panels=20,curated_final_time_panels=65,new_generated_features=4096,reference_features=2048,independently_checked_FIDs=13,figures_visually_reviewed=3,source_bindings=len(study.load(root/'bindings.json')),local_links=links,unchanged_scopes=protected,new_ODE_architecture_nominated=False))
shutil.copytree(doc/'results',root/'curated-results');closure=root/'closure-source';closure.mkdir()
for f in doc.iterdir():
    if f.is_file():shutil.copyfile(f,closure/f.name)
for name,source in [('preflight.log','image-exponential-v2-preflight.log'),('controller.log','image-exponential-v2-controller.log')]:shutil.copyfile(Path('/Users/mr.z/ode-workspace-prep')/source,root/name)
study.verify(root,True);files=sorted(f for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name!='sealed-runtime.json')
study.write(root/'sealed-runtime.json',{str(f.relative_to(root)):study.sha(f) for f in files})
for f,digest in study.load(root/'sealed-runtime.json').items():assert study.sha(root/f)==digest
print(json.dumps(dict(status='SEALED_AND_VERIFIED',files=len(files),seal_sha256=study.sha(root/'sealed-runtime.json'),private_bytes=sum(f.stat().st_size for f in files),curated_bytes=sum(f.stat().st_size for f in (doc/'results').rglob('*') if f.is_file()))))
