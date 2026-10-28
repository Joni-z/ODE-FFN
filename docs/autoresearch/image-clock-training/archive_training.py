"""Verify reported cold-control evidence and preserve its complete private closure."""
import argparse,ast,csv,importlib.util,json,re,shutil,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
import numpy as np

def main(root):
    doc=Path(__file__).resolve().parent;repo=doc.parents[2]
    sys.path.insert(0,str(root/'source'));import training_controls as study
    spec=importlib.util.spec_from_file_location('registered_control_curator',root/'curation-source/curate_training.py')
    curator=importlib.util.module_from_spec(spec);spec.loader.exec_module(curator)
    assert not (root/'sealed-runtime.json').exists();study.verify(root)
    external=curator.verify_external_data(study,root)
    registration=study.load(root/'curation-source/registration.json')
    assert registration['registered_before_new_quality_scores'] and not registration['changes_formal_source_or_acceptance']
    for name,digest in registration['files'].items():
        assert study.sha(root/'curation-source'/name)==study.sha(doc/name)==digest
        registered=subprocess.check_output(['git','-C',str(repo),'show',registration['source_commit']+':'+str((doc/name).relative_to(repo))])
        assert registered==(doc/name).read_bytes()
    assert study.load(root/'orchestrator/status.json')['state']=='COMPLETE'
    phases=study.load(root/'orchestrator/launches.json');workers=study.load(root/'output/launches.json')
    assert len(phases)==6 and len(workers)==2 and all(v['exit_code']==0 for v in [*phases,*workers])
    assert study.load(root/'controller-tool-exit.json')['exit_code']==study.load(root/'curation-tool-exit.json')['exit_code']==0
    receipt=study.load(root/'documented-pass.json');assert receipt['status']=='PASS' and receipt['doc_vs_reality_mismatches']==[]
    assert receipt['preflight_sha256']==study.sha(root/'preflight.json')
    assert receipt['metric_preflight_sha256']==study.sha(root/'metric-preflight.json')
    for name in ('training_controls.py','clock_operators.py','audit_training.py','controller.py'):
        assert study.sha(doc/name)==study.sha(root/'source'/name)
    for p in (root/'curation').rglob('*'):
        if p.is_file():assert study.sha(p)==study.sha(doc/'results'/p.relative_to(root/'curation'))
    assert study.load(doc/'results/external-data-verification.json')==external
    audits={name:study.load(root/f'audit-{name}/result.json') for name in ('training','sampling','features')}
    assert all(v['status']=='PASS' for v in audits.values())
    assert audits['training']['all_native_stream_digests']==42 and audits['training']['physical_boundary_updates']==12
    assert not audits['training']['full_optimization_trajectory_replayed']
    assert audits['sampling']['new_generated_images_bitwise']==audits['features']['new_generated_features_bitwise']==7168
    assert audits['features']['reference_features_bitwise']==2048
    read=lambda name:list(csv.DictReader((doc/'results'/name).open()))
    formal=study.load(root/'metric-output/result.json')['rows'];by={(v['arm'],v['step']):v for v in formal}
    rows=read('comparison.csv');assert len(rows)==len(by)==32
    for r in rows:
        arm,step=r['arm'],int(r['step']);actual=by[arm,step]
        for key in ('weighted_mse','clean_mse','fid','FID_mean_term','FID_covariance_term','FFN_forward_MACs'):
            assert float(r[key])==actual[key],(arm,step,key)
        assert int(r['stored_learned_parameters'])==actual['parameters']
        frozen=arm in ('native_posthoc06','ode_k1_time','ode_k1_pooled','ode_k1_constant','ode_k2_constant')
        active=185744 if frozen or arm in study.ARMS else actual['parameters']
        assert int(r['active_learned_parameters'])==active
        assert int(r['training_active_learned_parameters'])==(211094 if frozen else active)
        assert (int(r['generated']),int(r['reference']),int(r['Heun_steps']))==(1024,2048,128)
        raw=np.load(curator.sample_path(study,root,arm,step));assert raw.shape==(1024,3,32,32) and np.isfinite(raw).all()
        assert float(r['raw_tail_fraction'])==float(np.mean(np.abs(raw)>1))
        checked=next(v for v in audits['features']['rows'] if (v['arm'],v['step'])==(arm,step))
        assert abs(checked['fid']-actual['fid'])<1e-8
    primary=read('primary-20k.csv');assert len(primary)==14 and all(int(r['step'])==20000 for r in primary)
    assert primary==[r for r in rows if int(r['step'])==20000]
    ratios=read('primary-ratios.csv');assert len(ratios)==26
    for r in ratios:
        a=by[r['arm'],20000];b=by[r['reference'],20000]
        assert r['arm'] in study.ARMS and r['arm']!=r['reference']
        for key in ('fid','weighted_mse','clean_mse'):assert float(r[key+'_ratio'])==a[key]/b[key]
        assert float(r['FID_difference'])==a['fid']-b['fid']
    panels=read('time-panels.csv');assert len(panels)==75
    for r in panels:
        arm,step,t=r['arm'],int(r['step']),float(r['time'])
        p=root/'output/runs'/arm/f'panel_{step:07d}_{t:.1f}.npz' if arm in study.ARMS else root/'sampling'/f'posthoc-panel-{t:.1f}.npz'
        with np.load(p) as f:prediction=f['prediction'].astype(np.float64)
        b=study.bank(t);error=prediction-b['target'].astype(np.float64);delta=b['delta'].astype(np.float64)[:,None,None,None]
        assert int(r['examples'])==64
        assert [float(r['weighted_mse']),float(r['clean_mse'])]==[float(np.mean((error/delta)**2)),float(np.mean(error**2))]
    matrices=read('matrix-history.csv');assert len(matrices)==210
    states={step:study.native_state(step) for step in study.STEPS}
    for r in matrices:
        arm,step=r['arm'],int(r['step'])
        if arm in study.ARMS:
            with np.load(root/'output/runs'/arm/f'matrices-{step:07d}.npz') as f:a=f[r['layer']].astype(np.float64)
        else:
            match=re.fullmatch(r'block([0-2])\.P([12])',r['layer']);assert arm=='ode_k1' and match
            j,k=match.groups();a=states[step]['model'][f'blocks.{j}.mlp.ode{k}.proj.weight'].numpy().astype(np.float64)
        singular=np.linalg.svd(a,compute_uv=False);eigen=np.linalg.eigvals(a)
        wanted=[singular[0],singular[-1],singular.std()/singular.mean(),np.linalg.norm(a),eigen.real.max(),np.abs(eigen).max()]
        observed=[float(r[k]) for k in ('sigma_max','sigma_min','singular_CV','frobenius','eigen_max_real','eigen_max_abs')]
        np.testing.assert_allclose(observed,wanted,rtol=1e-12,atol=1e-12)
    curves=read('training-curves.csv');assert {r['arm'] for r in curves}=={*study.ARMS,'ode_k1','swiglu86','silu64','silu131'}
    for arm in {r['arm'] for r in curves}:
        if arm in study.ARMS:source=curator.jsonlines(root/'output/runs'/arm/'metrics.jsonl')
        else:
            parent=study.PARENTS['ordinary'] if arm.startswith('silu') else study.PARENTS['native']
            source=curator.jsonlines(parent/'output/runs'/arm/'metrics.jsonl')+curator.jsonlines(study.PARENTS['budget']/'output/runs'/arm/'metrics.jsonl')
        values={v['step']:v for v in source};selected=[r for r in curves if r['arm']==arm];assert len(selected)==len(values)
        for r in selected:
            v=values[int(r['step'])]
            for k in ('weighted_mse','clean_mse','train_loss','grad_norm','clip_fraction'):
                assert (None if r[k]=='' else float(r[k]))==v.get(k)
    decision=study.load(doc/'results/decision.json');assert decision==study.decisions(formal)
    assert not decision['new_ODE_method'] and not decision['posthoc_reference_nominated'] and decision['ordinary_controls_only']
    assert 'FINAL QUALITY PENDING' not in (doc/'stats-appendix.md').read_text()
    labels={v:k for k,v in curator.LABELS.items()};reported=[]
    for line in (doc/'analysis-report.md').read_text().splitlines():
        cells=[v.strip().replace('**','') for v in line.strip().strip('|').split('|')]
        if len(cells)==4 and cells[0] in labels:reported.append(cells)
    assert len(reported)==14
    for label,*values in reported:
        v=by[labels[label],20000]
        assert values==[format(v[k],p) for k,p in [('weighted_mse','.6f'),('clean_mse','.6f'),('fid','.2f')]]
    review=study.load(root/'visual-review.json');pngs={p.name for p in (root/'curation/figures').glob('*.png')}
    assert review['status']=='PASS' and len(pngs)==5 and set(review['files'])==pngs
    for name,digest in review['files'].items():assert study.sha(root/'curation/figures'/name)==digest
    assert len(list((doc/'results/figures').glob('*')))==10
    assert len(list(csv.DictReader((doc/'results/results.tsv').open(),delimiter='\t')))==32
    assert study.load(doc/'results/cost-witness.json')['rows']==study.load(root/'preflight.json')['costs']
    resources=curator.jsonlines(root/'output/resources.jsonl')
    for r in read('resource-summary.csv'):
        assert float(r['training_seconds'])==study.load(root/'output/runs'/r['arm']/'result.json')['seconds']
        assert int(r['peak_CPU_RSS_bytes'])==max(v['RSS'] for record in resources for v in record['resources'] if v['arm']==r['arm'])
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
    study.verify(root);assert curator.verify_external_data(study,root)==external
    study.write(doc/'results/artifact-verification.json',dict(status='PASS',endpoints=32,primary20k=14,ratios=26,time_panels=75,
        matrix_summaries=210,native_stream_checks=42,boundary_updates=12,full_optimization_trajectory_replayed=False,
        new_samples_replayed=7168,new_features_replayed=7168,reference_features_replayed=2048,figures_visually_reviewed=5,
        report_rows_verified=14,external_input_files_verified=3,local_links=links,protected_scopes=protected,new_ODE_method=False))
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
