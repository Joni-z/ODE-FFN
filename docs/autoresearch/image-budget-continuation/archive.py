"""Close the audited seven-arm comparison without changing any model result."""
from pathlib import Path
import argparse, ast, csv, importlib.util, json, re, shutil, subprocess, sys
from urllib.parse import unquote
import numpy as np


def main(root):
    doc=Path(__file__).resolve().parent;repo=doc.parents[2]
    sys.path.insert(0,str(root/'source'))
    import study
    assert not (root/'sealed-runtime.json').exists()
    study.verify(root)
    train=study.load(root/'output/summary.json')
    assert train['state']=='COMPLETE' and len(train['exits'])==7
    assert all(r['exit_code']==0 for r in train['exits'])
    done=study.load(root/'finalization/status.json')
    assert done['phase']=='AUDITS_COMPLETE' and len(done['exits'])==2
    assert all(r['exit_code']==0 for r in done['exits'])
    exits=study.load(root/'tool-exits.json')
    assert {r['phase'] for r in exits}=={'training','evaluation','finalization','curation'}
    assert all(r['exit_code']==0 and isinstance(r['session'],int) for r in exits)
    training=study.load(root/'audit-training/result.json')
    features=study.load(root/'audit-features/result.json')
    assert training['status']==features['status']=='PASS'
    assert len(training['rows'])==112 and len(training['sampling'])==14
    assert training['independent_stream_batches']==15000
    assert len(features['rows'])==14 and features['generated_features_bitwise']==14336
    assert features['reference_features_bitwise']==2048
    for f in (root/'curation').rglob('*'):
        if f.is_file():assert study.sha(f)==study.sha(doc/'results'/f.relative_to(root/'curation'))
    read=lambda name:list(csv.DictReader((doc/'results'/name).open()))
    rows=read('comparison.csv');assert len(rows)==21
    assert {(r['arm'],int(r['step'])) for r in rows}=={(a,s) for a in study.ARMS for s in (5000,10000,20000)}
    prior={r['family']:r for r in csv.DictReader((study.PARENTS['ordinary']/'curation/comparison.csv').open())}
    formal=study.load(root/'metric-output/result.json')['rows']
    for r in rows:
        arm,step=r['arm'],int(r['step'])
        for key in ('parameters','FFN_forward_MACs'):assert int(r[key])==int(prior[arm][key])
        if step==5000:
            for key in ('weighted_mse','clean_mse','fid'):assert float(r[key])==float(prior[arm][key])
            continue
        actual=next(x for x in formal if x['arm']==arm and x['step']==step)
        checked=next(x for x in features['rows'] if x['arm']==arm and x['step']==step)
        fit=next(x for x in training['rows'] if x['arm']==arm and x['step']==step)
        for key in ('weighted_mse','clean_mse','fid'):assert float(r[key])==actual[key]
        assert abs(float(r['fid'])-checked['fid'])<1e-8
        for key in ('weighted_mse','clean_mse'):np.testing.assert_allclose(float(r[key]),fit[key],rtol=1e-8,atol=1e-10)
        assert abs(float(r['FID_mean_term'])+float(r['FID_covariance_term'])-float(r['fid']))<1e-10
    curves=read('curves.csv');assert len(curves)==112
    for r in curves:
        actual=next(x for x in training['rows'] if x['arm']==r['arm'] and x['step']==int(r['step']))
        for key in ('weighted_mse','clean_mse','prediction_rms'):np.testing.assert_allclose(float(r[key]),actual[key],rtol=1e-8,atol=1e-10)
    panels=read('time-panels.csv');assert len(panels)==70
    for r in panels:
        time=float(r['time']);step=int(r['step']);arm=r['arm']
        with np.load(study.PARENTS['native']/'banks'/f'panel_{time:.1f}.npz') as b:
            target=b['target'].astype(np.float64);delta=b['delta'].astype(np.float64)[:,None,None,None]
        with np.load(root/'output/runs'/arm/f'panel_{step:07d}_{time:.1f}.npz') as b:pred=b['prediction'].astype(np.float64)
        error=pred-target
        np.testing.assert_allclose([float(r['weighted_mse']),float(r['clean_mse'])],[np.mean((error/delta)**2),np.mean(error**2)],rtol=1e-12,atol=1e-12)
    history=read('P-history.csv');assert len(history)==48
    for r in history:
        actual=next(x for x in study.load(root/'output/runs'/r['arm']/f"P_{int(r['step']):07d}.json") if x['layer']==r['layer'])
        sv=np.asarray(actual['singular_values'])
        for key,value in dict(sigma_max=sv[0],singular_CV=sv.std()/sv.mean(),frobenius=actual['frobenius'],max_real_eigenvalue=max(actual['eigenvalues_real'])).items():
            np.testing.assert_allclose(float(r[key]),value,rtol=1e-12,atol=1e-12)
    source=root/'curation-source-v3/curate.py'
    spec=importlib.util.spec_from_file_location('image_budget_curator',source)
    curator=importlib.util.module_from_spec(spec);spec.loader.exec_module(curator)
    decision=study.load(doc/'results/decision.json')
    assert decision['primary_step']==20000 and decision['new_method_established'] is False
    assert decision['criteria']==curator.priority(formal)
    assert curator.check_rule()['status']=='PASS'
    labels={'ODE K1':'ode_k1','ODE K2':'ode_k2','Swi86':'swiglu86','Swi129':'swiglu129',
            'SiLU64':'silu64','SiLU131':'silu131','SiLU193':'silu193'}
    report_rows=[]
    for line in (doc/'analysis-report.md').read_text().splitlines():
        cells=[x.strip().replace('**','') for x in line.strip().strip('|').split('|')]
        if len(cells)==5 and cells[0] in labels:report_rows.append(cells)
    assert len(report_rows)==7
    for label,*values in report_rows:
        arm=labels[label]
        expected=[format(float(next(r['fid'] for r in rows if r['arm']==arm and int(r['step'])==s)),'.2f') for s in (5000,10000,20000)]
        expected.append(format(float(next(r['weighted_mse'] for r in rows if r['arm']==arm and int(r['step'])==20000)),'.6f'))
        assert values==expected,(label,values,expected)
    ledger=list(csv.DictReader((root/'results.tsv').open(),delimiter='\t'))
    assert len(ledger)==7 and {r['description'].split(':')[0] for r in ledger}==set(study.ARMS)
    for r in ledger:
        arm=r['description'].split(':')[0]
        assert abs(float(r['metric'])-next(x['fid'] for x in formal if x['arm']==arm and x['step']==20000))<1e-8
    assert study.sha(root/'results.tsv')==study.sha(doc/'results/results.tsv')
    review=study.load(root/'visual-review.json')
    pngs={f.name for f in (root/'curation').glob('*.png')};assert len(pngs)==7
    assert review['status']=='PASS' and set(review['files'])==pngs
    for name,digest in review['files'].items():assert study.sha(root/'curation'/name)==digest
    for name in ('analysis-report.md','stats-appendix.md','figure-catalog.md','README.md'):assert (doc/name).exists()
    protected=['src','synthetic','configs','scripts','reports/report1']
    assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only','98730e0','--',*protected],text=True)
    links=0
    for p in [*doc.glob('*.md'),repo/'README.md',doc.parent/'README.md']:
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            link=link.strip('<>').split('#')[0]
            if not link or '://' in link or link.startswith(('/','mailto:')):continue
            assert (p.parent/unquote(link)).exists(),(p,link);links+=1
    for p in doc.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    study.verify(root)
    for name,expected in [('oct28-cycle19-v2','c8ecd01c4c24ad6be6f3a8d1bf3aa7b5b3222f03712e4879e2cd5ecf15759d96'),('oct28-cycle21-v2','fa13abe370318077c521bb4202bf8f37b8ccfee6daf4cb5c0ef7ea92daf9b3cf')]:
        parent=root.parent/name;assert study.sha(parent/'sealed-runtime.json')==expected
        for file,digest in study.load(parent/'sealed-runtime.json').items():assert study.sha(parent/file)==digest
    logs=root/'phase-logs';logs.mkdir()
    for name in ('controller','evaluation','finalization'):
        source=Path('/Users/mr.z/ode-workspace-prep')/f'image-budget-{name}.log'
        shutil.copyfile(source,logs/f'{name}.log')
    study.write(doc/'results/artifact-verification.json',dict(status='PASS',continued_native_arms=7,new_architecture=False,
        primary_step=20000,checkpoint_replays=112,sampling_banks=14,independent_stream_batches=15000,
        generated_features=14336,reference_features=2048,independent_FIDs=14,P_records=48,time_panels=70,
        figures_visually_reviewed=7,report_rows_verified=7,analysis_reports_verified=3,
        local_links=links,protected_scopes=protected,new_method_established=False))
    shutil.copytree(doc/'results',root/'curated-results')
    closure=root/'closure-source';closure.mkdir()
    for p in doc.iterdir():
        if p.is_file():shutil.copyfile(p,closure/p.name)
    files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='sealed-runtime.json')
    study.write(root/'sealed-runtime.json',{str(p.relative_to(root)):study.sha(p) for p in files})
    for file,digest in study.load(root/'sealed-runtime.json').items():assert study.sha(root/file)==digest
    print(json.dumps(dict(status='SEALED_AND_VERIFIED',files=len(files),seal_sha256=study.sha(root/'sealed-runtime.json'),
        private_bytes=sum(p.stat().st_size for p in files),curated_bytes=sum(p.stat().st_size for p in (doc/'results').rglob('*') if p.is_file()))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root.resolve())
