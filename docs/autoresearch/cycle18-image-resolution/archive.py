"""Verify frozen image-resolution evidence and seal the completed runtime."""
from pathlib import Path
import argparse,ast,csv,hashlib,json,re,shutil,subprocess,sys
from urllib.parse import unquote
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
assert not (root/'sealed-runtime.json').exists(),'Preserve existing seal'
doc=Path(__file__).resolve().parent;repo=doc.parents[2];sys.path.insert(0,str(root/'source'));import study
study.verify(root);p=study.parent(root)
assert study.load(root/'orchestrator/status.json')['state']=='FAILED'
assert study.load(root/'orchestrator-v2/status.json')['state']=='COMPLETE'
assert study.load(root/'tool-exits.json')=={'original_session':99684,'original_exit_code':1,'recovery_session':69697,'recovery_exit_code':0}
assert len(study.load(root/'output/exits.json'))==4 and all(r['exit_code']==0 for r in study.load(root/'output/exits.json'))
for name in ('audit-v2-bindings.json','sampling-bindings.json'):
    for file,digest in study.load(root/name).items():assert study.sha(file)==digest,file
for folder in ('orchestrator','orchestrator-v2'):
    binding=study.load(root/folder/'binding.json')
    assert study.sha(root/folder/'controller.py')==binding['controller_sha256']
assert study.sha(root/'resource-amendment.md')==study.load(root/'orchestrator/binding.json')['resource_amendment_sha256']
for phase in ('sampling','features'):
    result=study.load(root/f'audit-{phase}/result.json');assert result['status']=='PASS'
    assert len(result['rows'])==(8 if phase=='sampling' else 12)
assert all(r['exit_code']==0 for r in study.load(root/'orchestrator-v2/launches.json'))
for f in (root/'curation').rglob('*'):
    if f.is_file():assert study.sha(f)==study.sha(doc/'results'/f.relative_to(root/'curation'))
read=lambda path:list(csv.DictReader(path.open()))
comparison=read(doc/'results/comparison.csv');assert len(comparison)==12
formal=study.load(root/'metric-output/result.json')['rows'];independent=study.load(root/'audit-features/result.json')['rows']
for arm in study.ARMS:
    assert sorted(int(r['N']) for r in comparison if r['family']==arm)==list(study.GRIDS)
    for n in study.GRIDS:
        public=next(r for r in comparison if r['family']==arm and int(r['N'])==n)
        native=next(r for r in formal if r['family']==arm and r['N']==n)
        replay=next(r for r in independent if r['family']==arm and r['N']==n)
        for key in ('fid','mean_term','covariance_term','out_of_range_fraction'):assert float(public[key])==native[key]
        assert abs(replay['fid']-native['fid'])<1e-8
        assert int(public['sample_count'])==1024 and int(public['reference_count'])==2048
    raw=lambda n:np.load(p/'output/runs'/arm/'generated.npy') if n==32 else np.load(root/'output/runs'/arm/f'generated-{n}.npy')
    for left,right in ((32,64),(64,128),(32,128)):
        public=next(r for r in read(doc/'results/sample-differences.csv') if r['family']==arm and int(r['left_N'])==left and int(r['right_N'])==right)
        error=raw(left).astype(np.float64)-raw(right).astype(np.float64)
        assert float(public['pixel_RMS'])==float(np.sqrt(np.mean(error**2)))
        assert float(public['pixel_max'])==float(np.max(np.abs(error)))
assert len(read(doc/'results/contrasts.csv'))==15
assert study.load(doc/'results/decision.json')['new_architecture_nominated'] is False
assert len(list((doc/'results/figures').glob('*')))==6
for name in ('analysis-report.md','stats-appendix.md','figure-catalog.md'):assert (doc/name).is_file()
links=0
for f in [*doc.glob('*.md'),repo/'README.md',doc.parent/'README.md']:
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',f.read_text()):
        target=target.strip('<>').split('#')[0]
        if not target or '://' in target or target.startswith(('/','mailto:')):continue
        assert (f.parent/unquote(target)).exists(),(f,target);links+=1
for f in doc.glob('*.py'):ast.parse(f.read_text(),filename=str(f))
protected=['src','synthetic','configs','scripts','reports/report1']
assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only','98730e0','--',*protected],text=True)
for f in (doc/'interpretation-fixtures').glob('*.json'):assert study.load(f)['status']=='PASS'
coverage=study.load(doc/'primary-source-check.json');assert set(coverage['canonical_ids'])==set(coverage['covered_ids']) and not coverage['uncovered_ids']
study.write(doc/'results/artifact-verification.json',dict(status='PASS',parent_seal_files=392,finer_sample_banks=8,grid_scores=12,contrasts=15,
    generated_feature_images_replayed=12288,reference_feature_images_replayed=2048,figures_visually_reviewed=3,local_links=links,
    source_bindings=len(study.load(root/'bindings.json')),unchanged_scopes=protected,new_architecture_nominated=False))
shutil.copytree(doc/'results',root/'curated-results');closure=root/'closure-source';closure.mkdir()
for f in doc.iterdir():
    if f.is_file():shutil.copyfile(f,closure/f.name)
shutil.copytree(doc/'interpretation-fixtures',closure/'interpretation-fixtures')
for name,source in [('preflight.log','image-resolution-preflight.log'),('original-controller.log','image-resolution-controller.log'),('recovery-controller.log','image-resolution-recovery.log')]:
    shutil.copyfile(Path('/Users/mr.z/ode-workspace-prep')/source,root/name)
study.verify(root)
files=sorted(f for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name!='sealed-runtime.json')
study.write(root/'sealed-runtime.json',{str(f.relative_to(root)):study.sha(f) for f in files})
for name,digest in study.load(root/'sealed-runtime.json').items():assert study.sha(root/name)==digest
print(json.dumps(dict(status='SEALED_AND_VERIFIED',files=len(files),seal_sha256=study.sha(root/'sealed-runtime.json'),private_bytes=sum(f.stat().st_size for f in files),
    curated_bytes=sum(f.stat().st_size for f in (doc/'results').rglob('*') if f.is_file()))))
