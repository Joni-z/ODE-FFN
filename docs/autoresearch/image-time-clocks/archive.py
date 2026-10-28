"""Verify all public intervention evidence, then preserve one immutable closure."""
import argparse,ast,csv,importlib.util,json,re,shutil,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
import numpy as np

def main(root):
    doc=Path(__file__).resolve().parent;repo=doc.parents[2];sys.path.insert(0,str(root/'source'));import study
    assert not (root/'sealed-runtime.json').exists();study.verify(root,True)
    assert study.load(root/'orchestrator/status.json')['state']=='COMPLETE'
    phases=study.load(root/'orchestrator/launches.json');children=study.load(root/'output/launches.json')
    assert len(phases)==4 and len(children)==2 and all(r['exit_code']==0 for r in [*phases,*children])
    assert study.load(root/'controller-tool-exit.json')['exit_code']==study.load(root/'curation-tool-exit.json')['exit_code']==0
    receipt=study.load(root/'documented-pass.json');assert receipt['status']=='PASS' and not receipt['doc_vs_reality_mismatches']
    assert receipt['preflight_sha256']==study.sha(root/'preflight.json')
    assert receipt['metric_preflight_sha256']==study.sha(root/'metric-preflight.json')
    preflight=study.load(root/'preflight.json');assert preflight['failed_preflight_calibration_recomputed_bitwise']
    assert len(preflight['promoted_witnesses'])==10 and all(r['FFN_forward_MACs']==8192 for r in preflight['costs'])
    assert all((r['stored_parameters'],r['active_parameters'],r['fixed_position_parameters'])==(211094,185744,4096) for r in preflight['costs'])
    for name in ('study.py','audit.py','controller.py','curate.py'):assert study.sha(doc/name)==study.sha(root/'source'/name)
    for p in (root/'curation').rglob('*'):
        if p.is_file():assert study.sha(p)==study.sha(doc/'results'/p.relative_to(root/'curation'))
    read=lambda name:list(csv.DictReader((doc/'results'/name).open()))
    rows=read('comparison.csv');formal=study.load(root/'metric-output/result.json')['rows'];assert len(rows)==len(formal)==11
    checked=study.load(root/'audit-features/result.json');sampling=study.load(root/'audit-sampling/result.json')
    assert checked['status']==sampling['status']=='PASS' and len(checked['rows'])==11
    assert checked['new_generated_features_bitwise']==checked['reference_features_bitwise']==sampling['new_generated_images_bitwise']==2048
    for r in rows:
        actual=next(v for v in formal if v['family']==r['family'])
        for key in ('weighted_mse','clean_mse','fid','mean_term','covariance_term','out_of_range_fraction','parameters','FFN_forward_MACs'):
            assert float(r[key])==actual[key],(r['family'],key)
        assert (int(r['step']),int(r['generated']),int(r['reference']),int(r['Heun_steps']))==(20000,1024,2048,128)
        assert abs(float(r['fid'])-next(v['fid'] for v in checked['rows'] if v['family']==r['family']))<1e-8
        assert float(r['out_of_range_fraction'])==float(np.mean(np.abs(np.load(study.sample_bank(root,r['family'])))>1))
    profile=read('calibration-profile.csv');pooled=read('calibration-pooled.csv');assert len(profile)==774 and len(pooled)==6
    with np.load(root/'native-calibration.npz') as f:
        for r in profile:
            value=f[r['layer']][int(r['time_index'])].astype(np.float64)
            for key,v in dict(value=np.float32(value.mean()),native_mean=value.mean(),native_SD=value.std(),native_min=value.min(),native_max=value.max()).items():assert float(r[key])==float(v)
        for r in pooled:assert float(r['pooled'])==float(np.float32(f[r['layer']].astype(np.float64).mean()))
    panels=read('time-panels.csv');assert len(panels)==55
    for r in panels:
        t=float(r['time']);name=r['family']
        with np.load(study.PARENTS['native']/'banks'/f'panel_{t:.1f}.npz') as f:target=f['target'].astype(np.float64);delta=f['delta'].astype(np.float64)[:,None,None,None]
        if name in study.ARMS:path=root/'output/runs'/name/f'panel-{t:.1f}.npz'
        elif name.endswith('_constant'):path=study.PARENTS['clocks20k']/'output/runs'/name.removesuffix('_constant')/f'panel-{t:.1f}.npz'
        else:path=study.PARENTS['budget']/'output/runs'/name/f'panel_0020000_{t:.1f}.npz'
        with np.load(path) as f:error=f['prediction'].astype(np.float64)-target
        np.testing.assert_allclose([float(r['weighted_mse']),float(r['clean_mse'])],[np.mean((error/delta)**2),np.mean(error**2)],rtol=1e-12,atol=1e-12)
    decision=study.load(doc/'results/decision.json');assert decision==study.decision(formal)
    by={r['family']:r for r in formal};x=by['ode_k1_time'];p=by['ode_k1_pooled'];refs=[by[n] for n in ('swiglu86','silu131')]
    native_margin=all(x['fid']<=.95*r['fid'] for r in refs) and all(x[k]<=1.05*min(r[k] for r in refs) for k in ('weighted_mse','clean_mse'))
    specificity=x['fid']<=.95*p['fid'] and all(x[k]<=1.05*p[k] for k in ('weighted_mse','clean_mse'))
    assert decision['time_specific_explanation_supported']==(native_margin and x['fid']<by['silu64']['fid'] and specificity)
    assert not decision['new_trained_method'] and not decision['pooled_control_nominated'] and not decision['old_constant_decision_changed']
    spec=importlib.util.spec_from_file_location('time_profile_curator',root/'source/curate.py');curator=importlib.util.module_from_spec(spec);spec.loader.exec_module(curator)
    labels={v:k for k,v in curator.LABELS.items()};reported=[]
    for line in (doc/'analysis-report.md').read_text().splitlines():
        cells=[v.strip().replace('**','') for v in line.strip().strip('|').split('|')]
        if len(cells)==4 and cells[0] in labels:reported.append(cells)
    assert len(reported)==11
    for label,*values in reported:
        r=by[labels[label]];assert values==[format(r[k],p) for k,p in [('weighted_mse','.6f'),('clean_mse','.6f'),('fid','.2f')]]
    review=study.load(root/'visual-review.json');pngs={p.name for p in (root/'curation/figures').glob('*.png')}
    assert review['status']=='PASS' and len(pngs)==4 and set(review['files'])==pngs
    for name,digest in review['files'].items():assert study.sha(root/'curation/figures'/name)==digest
    assert len(list((doc/'results/figures').glob('*')))==8
    assert len(list(csv.DictReader((doc/'results/results.tsv').open(),delimiter='\t')))==11
    for name in ('analysis-report.md','stats-appendix.md','figure-catalog.md'):assert (doc/name).exists()
    protected=['src','synthetic','configs','scripts','reports/report1']
    assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only','98730e0','--',*protected],text=True)
    links=0
    for p in [*doc.glob('*.md'),repo/'README.md',doc.parent/'README.md']:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            target=target.strip('<>').split('#')[0]
            if not target or '://' in target or target.startswith(('/','mailto:')):continue
            path=(p.parent/unquote(target)).resolve();assert path.exists(),(p,target)
            assert subprocess.check_output(['git','-C',str(repo),'ls-files','--',str(path.relative_to(repo))],text=True).strip(),(p,'untracked',target)
            links+=1
    for p in doc.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    study.verify(root,True)
    study.write(doc/'results/artifact-verification.json',dict(status='PASS',endpoints=11,new_generated_images_replayed=2048,
        new_features_replayed=2048,reference_features_replayed=2048,FIDs_independently_recomputed=11,calendar_scalars=774,
        pooled_scalars=6,time_panel_scores=55,report_rows_verified=11,figures_visually_reviewed=4,local_links=links,
        protected_scopes=protected,failed_pre_quality_v1_preserved=True,new_training=False,new_trained_method=False))
    shutil.copytree(doc/'results',root/'curated-results');closure=root/'closure-source';closure.mkdir()
    for p in doc.iterdir():
        if p.is_file():shutil.copyfile(p,closure/p.name)
    files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='sealed-runtime.json')
    study.write(root/'sealed-runtime.json',{str(p.relative_to(root)):study.sha(p) for p in files})
    for name,digest in study.load(root/'sealed-runtime.json').items():assert study.sha(root/name)==digest
    print(json.dumps(dict(status='SEALED_AND_VERIFIED',files=len(files),seal_sha256=study.sha(root/'sealed-runtime.json'),
        private_bytes=sum(p.stat().st_size for p in files),curated_bytes=sum(p.stat().st_size for p in (doc/'results').rglob('*') if p.is_file()))),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root.resolve())
