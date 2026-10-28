"""Verify proof scope, all numerical artifacts and public copies before sealing."""
from pathlib import Path
import argparse,ast,csv,json,re,shutil,subprocess,sys
from urllib.parse import unquote
import numpy as np
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args();root=a.root.resolve()
doc=Path(__file__).resolve().parent;repo=doc.parents[2];sys.path.insert(0,str(root/'source'));import study
assert not (root/'sealed-runtime.json').exists();study.verify(root)
assert study.load(root/'tool-exit.json')==dict(session=3045,exit_code=0)
assert study.load(root/'orchestrator/status.json')['state']=='COMPLETE'
launches=study.load(root/'orchestrator/launches.json');assert len(launches)==2 and all(r['exit_code']==0 for r in launches)
formal=study.load(root/'output/result.json');audit=study.load(root/'audit/result.json')
assert formal['status']==audit['status']=='PASS' and len(formal['cases'])==120 and audit['point_cases']==120 and audit['finite_difference_cases']==240
assert audit['both_steps_retained'] and len(audit['records'])==240
for file in (root/'curation').rglob('*'):
    if file.is_file():assert study.sha(file)==study.sha(doc/'results'/file.relative_to(root/'curation'))
read=lambda name:list(csv.DictReader((doc/'results'/name).open()))
rows=read('point-cases.csv');assert len(rows)==120 and len(read('finite-differences.csv'))==240
source={}
for name,parent in study.paths(root).items():
    with np.load(parent/name/'validation.npz') as f:clean=f['x1'][:8].astype(np.float64)
    source[name]=(clean,np.random.default_rng(0).standard_normal(clean.shape))
for r in formal['cases']:
    public=next(v for v in rows if v['dataset']==r['dataset'] and float(v['time'])==r['time'] and int(v['index'])==r['index'])
    for key,value in r.items():
        if isinstance(value,(int,float)):assert float(public[key])==value
        else:assert public[key]==value
    name=r['dataset'];label=f"{name}-{r['time']:.1f}-{r['index']}"
    with np.load(root/'output/cases'/f'{label}.npz') as bank:x={k:bank[k] for k in bank.files}
    clean,noise=source[name];assert x['target'].tobytes()==clean[r['index']].tobytes() and x['source'].tobytes()==noise[r['index']].tobytes()
    t=float(x['t']);rr=float(x['r']);np.testing.assert_allclose(x['z'],t*x['target']+(1-t)*x['source'],rtol=0,atol=1e-14)
    np.testing.assert_allclose(x['y'],x['z']/t,rtol=0,atol=1e-14);assert abs(rr-((1-t)/t)**2)<1e-14
    np.testing.assert_allclose(x['covariance'],rr*x['jacobian'],rtol=0,atol=1e-9)
    var_rhs=x['variance_lap_term']+x['variance_transport_term'];fm_rhs=x['fm_lap_term']+x['fm_D_term']+x['fm_z_term']
    np.testing.assert_allclose(x['variance_residual'],x['variance_derivative']-var_rhs,rtol=0,atol=1e-12)
    np.testing.assert_allclose(x['fm_residual'],x['time_derivative']-fm_rhs,rtol=0,atol=1e-12)
    np.testing.assert_allclose(float(r['material_drift_RMS']),np.sqrt(np.mean(x['material_drift']**2)),rtol=1e-12,atol=1e-12)
fds=read('finite-differences.csv')
for r in audit['records']:
    v=next(x for x in fds if x['dataset']==r['dataset'] and float(x['time'])==r['time'] and int(x['index'])==r['index'] and float(x['h'])==r['h'])
    for key,value in r.items():
        if isinstance(value,(int,float)):assert float(v[key])==value
        else:assert v[key]==value
assert {float(r['h']) for r in fds}=={1e-4,5e-5}
proof=study.load(doc/'results/coverage-and-verdict.json');assert proof['acceptance_status']=='provisional' and proof['actual_returned_model'] is None
assert proof['source_unit_coverage']['assigned']==proof['source_unit_coverage']['covered']==9 and proof['microclaims']==35
assert all(proof[k]==0 for k in ('open_fatal','open_critical','open_major','open_minor'))
for receipt in proof['receipts']:
    assert set(receipt['assigned_unit_ids'])==set(receipt['covered_unit_ids'])=={'U'+str(i) for i in range(9)}
    assert len({item for row in receipt['entries'] for item in row['obligation_ids']})==35
raw=(root/'proof-review/raw-round2.md').read_text();fence=chr(96)*3
blocks=[json.loads(v) for v in re.findall(fence+r'json\s*(.*?)'+fence,raw,re.S)]
hashes=next(v for v in blocks if 'derivation.md' in v)
for name,digest in hashes.items():assert study.sha(doc/name)==digest,name
for name in ('study.py','controller.py'):assert study.sha(doc/name)==study.sha(root/'source'/name)
application=study.load(doc/'results/assumption-discharge.json');assert application['status']=='PASS' and len(application['rows'])==3
assert study.load(doc/'results/decision.json')['new_ODE_architecture_nominated'] is False
assert len(list((doc/'results/figures').glob('*')))==4
pdf=(doc/'results/proof-audit-summary.pdf').read_bytes();assert len(re.findall(rb'/Type\s*/Page\b',pdf))==1
for name in ('analysis-report.md','stats-appendix.md','figure-catalog.md','PROOF_AUDIT.md','proof_audit_report.tex'):assert (doc/name).exists()
protected=['src','synthetic','configs','scripts','reports/report1']
assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only','98730e0','--',*protected],text=True)
links=0
for file in [*doc.glob('*.md'),repo/'README.md',doc.parent/'README.md']:
    for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',file.read_text()):
        target=target.strip('<>').split('#')[0]
        if not target or '://' in target or target.startswith(('/','mailto:')):continue
        assert (file.parent/unquote(target)).exists(),(file,target);links+=1
for file in doc.glob('*.py'):ast.parse(file.read_text(),filename=str(file))
study.write(doc/'results/artifact-verification.json',dict(status='PASS',new_training=False,point_cases=120,finite_difference_cases=240,both_steps_retained=True,proof_source_units=9,extracted_microclaims=35,semantic_acceptance='native same-family provisional',actual_reviewer_model_unavailable=True,prior_laws=3,figures_visually_reviewed=2,audit_summary_pages=1,audit_summary_visual_review=True,TeX_compiled=False,local_links=links,unchanged_scopes=protected,new_ODE_architecture_nominated=False))
shutil.copytree(doc/'results',root/'curated-results');closure=root/'closure-source';closure.mkdir()
for file in doc.iterdir():
    if file.is_file():shutil.copyfile(file,closure/file.name)
shutil.copyfile(Path('/Users/mr.z/ode-workspace-prep/denoiser-dynamics-controller.log'),root/'controller.log')
study.verify(root);files=sorted(f for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.name!='sealed-runtime.json')
study.write(root/'sealed-runtime.json',{str(f.relative_to(root)):study.sha(f) for f in files})
for file,digest in study.load(root/'sealed-runtime.json').items():assert study.sha(root/file)==digest
print(json.dumps(dict(status='SEALED_AND_VERIFIED',files=len(files),seal_sha256=study.sha(root/'sealed-runtime.json'),private_bytes=sum(f.stat().st_size for f in files),curated_bytes=sum(f.stat().st_size for f in (doc/'results').rglob('*') if f.is_file()))))
