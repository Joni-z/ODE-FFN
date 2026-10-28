"""Check complete reference evidence and independently verify reporting arrays."""
import argparse,ast,csv,importlib.util,json,re,shutil,subprocess,sys
from pathlib import Path
from urllib.parse import unquote
import numpy as np


def main(root):
    doc=Path(__file__).resolve().parent;repo=doc.parents[2]
    sys.path.insert(0,str(root/'source'));import gaussian_reference as g
    assert Path(g.__file__).resolve()==root/'source/gaussian_reference.py'
    spec=importlib.util.spec_from_file_location('closure_gaussian_curator',root/'curation-source/curate_gaussian.py')
    curator=importlib.util.module_from_spec(spec);spec.loader.exec_module(curator)
    assert not (root/'sealed-runtime.json').exists();g.verify(root)
    preserved={str(p):g.sha(p) for directory in ('fit','sampling','metric-output','curation','reporting-subspaces')
               for p in (root/directory).rglob('*') if p.is_file()}
    for filename,names in [('registration.json',('curate_gaussian.py',)),('subspace-registration.json',('analyze_subspaces.py',))]:
        registration=g.load(root/'curation-source'/filename)
        for name in names:
            assert g.sha(root/'curation-source'/name)==g.sha(doc/name)==registration['files'][name]
            original=subprocess.check_output(['git','-C',str(repo),'show',registration['source_commit']+':'+str((doc/name).relative_to(repo))])
            assert original==(doc/name).read_bytes()
    assert g.load(root/'curation-source/subspace-registration.json')['reporting_only_after_outcomes']
    for name in ('gaussian_math.py','gaussian_reference.py','protocol.md'):
        assert g.sha(doc/name)==g.sha(root/'source'/name)
    phases=g.load(root/'orchestrator/launches.json')
    assert g.load(root/'orchestrator/status.json')['state']=='COMPLETE'
    assert [v['phase'] for v in phases]==['fit','audit-fit','sample','audit-sampling','metrics','audit-features']
    assert all(v['exit_code']==0 for v in phases)
    assert g.load(root/'controller-tool-exit.json')['exit_code']==g.load(root/'curation-tool-exit.json')['exit_code']==0
    readiness=g.load(root/'documented-pass.json')
    assert readiness['status']=='PASS' and readiness['doc_vs_reality_mismatches']==[]
    assert readiness['preflight_sha256']==g.sha(root/'preflight.json')
    assert all(v['exit_code']==0 for v in readiness['records'])
    preflight=g.load(root/'preflight.json')
    assert preflight['status']=='PASS' and not preflight['real_statistics_fitted']
    assert preflight['image_shape_native_Heun_samples']==64 and preflight['physical_prior_and_native_sampler_bitwise']
    assert preflight['PPCA_matrix_MACs_per_image']==362496
    assert preflight['historical_features_bitwise']==preflight['historical_FIDs_recomputed']==32
    fit=g.load(root/'fit/result.json');audits={k:g.load(root/f'audit-{k}/result.json') for k in ('fit','sampling','features')}
    assert fit['status']=='PASS' and fit['train_images']==fit['MLE_denominator']==45000 and not fit['validation_used_for_fit']
    assert all(v['status']=='PASS' for v in audits.values())
    assert audits['fit']['train_images']==45000 and audits['fit']['independent_raw_moment_batch']==257
    assert len(audits['fit']['rows'])==2 and audits['fit']['formal_files_unchanged']==5
    assert audits['sampling']['generated_images_checked']==2048
    for r in audits['sampling']['rows']:
        assert r['generated_images']==1024
        assert r['sample_difference']['maximum']<3e-5 and r['sample_difference']['RMS']<3e-6
        assert all(v['maximum']<3e-6 for v in r['prediction_errors'])
    assert audits['features']['new_features_bitwise']==audits['features']['reference_features_bitwise']==2048
    assert len(audits['features']['rows'])==34
    assert max(abs(v['difference']) for v in audits['features']['rows'])<1e-8
    for p in (root/'curation').rglob('*'):
        if p.is_file():
            copied=doc/'results'/p.relative_to(root/'curation')
            if p.name=='full-documented-pass.json':assert g.load(p)==g.load(copied)
            else:assert g.sha(p)==g.sha(copied)
    for p in (root/'reporting-subspaces').iterdir():
        if p.is_file():assert g.sha(p)==g.sha(doc/'results/subspace-analysis'/p.name)
    read=lambda name:list(csv.DictReader((doc/'results'/name).open()))
    formal=g.load(root/'metric-output/result.json')['rows'];by={(r['arm'],r['step']):r for r in formal}
    rows=read('comparison.csv');assert len(rows)==len(by)==34
    old={(v['arm'],int(v['step'])):v for v in csv.DictReader((g.COLD/'curation/comparison.csv').open())}
    feature_audit={(v['arm'],v['step']):v for v in audits['features']['rows']}
    for r in rows:
        arm,step=r['arm'],int(r['training_updates']);actual=by[arm,step];statistical=arm in g.ARMS
        assert r['fit_kind']==actual['fit_kind']
        for k in ('weighted_mse','clean_mse','fid','FID_mean_term','FID_covariance_term'):assert float(r[k])==actual[k]
        assert int(r['stored_fitted_scalars'])==actual['parameters']
        assert int(r['inference_active_values'])==(actual['parameters'] if statistical else int(old[arm,step]['active_learned_parameters']))
        assert int(r['counted_matrix_MACs_per_image'])==(actual['matrix_MACs_per_image'] if statistical else actual['FFN_forward_MACs']*64*3)
        assert ('complete Gaussian' in r['matrix_cost_scope'])==statistical
        assert r['value_precision']==('float64 statistics; float32 output' if statistical else 'float32 neural')
        assert r['budget_label']==('train-only statistical fit' if statistical else str(step)+' updates')
        assert (int(r['generated']),int(r['reference']),int(r['Heun_steps']))==(1024,2048,128)
        raw=np.load(actual['sample_path']);assert raw.shape==(1024,3,32,32) and raw.dtype==np.float32 and np.isfinite(raw).all()
        assert float(r['raw_tail_fraction'])==float(np.mean(np.abs(raw)>1))
        assert abs(feature_audit[arm,step]['fid']-actual['fid'])<1e-8
    primary=read('primary-comparison.csv')
    expected=sorted([r for r in rows if int(r['training_updates'])==20000 or r['arm'] in g.ARMS],key=lambda r:curator.ORDER.index(r['arm']))
    assert primary==expected and len(primary)==16
    ratios=read('primary-ratios.csv');assert len(ratios)==28
    for r in ratios:
        a=by[r['arm'],0];b=by[r['reference'],20000];assert r['arm'] in g.ARMS
        for k in ('fid','weighted_mse','clean_mse'):assert float(r[k+'_ratio'])==a[k]/b[k]
        assert float(r['FID_difference'])==a['fid']-b['fid']
    panels=read('time-panels.csv');assert len(panels)==35
    for r in panels:
        arm,t=r['arm'],float(r['time'])
        if arm in g.ARMS:path=root/'sampling'/f'{arm}-panel-{t:.1f}.npz'
        else:path=(g.COLD if arm in ('fixed06','matched_silu64') else g.cold().PARENTS['budget'])/'output/runs'/arm/f'panel_0020000_{t:.1f}.npz'
        with np.load(path) as f:prediction=f['prediction']
        bank=g.bank(t);error=prediction.astype(np.float64)-bank['target'].astype(np.float64)
        wanted=[float(np.mean((error/bank['delta'].astype(np.float64)[:,None,None,None])**2)),float(np.mean(error**2))]
        assert int(r['examples'])==64 and [float(r['weighted_mse']),float(r['clean_mse'])]==wanted
    p=g.gm.load_parameters(root/'fit/ppca59.npz');d=3072;u=p['basis'];mu=p['mean']
    np.testing.assert_allclose(u.T@u,np.eye(59),atol=1e-12,rtol=1e-12)
    covariance=np.load(root/'fit/training-covariance.npy');trace=float(np.trace(covariance))
    np.testing.assert_allclose(covariance@u,u*p['eigenvalues'],atol=1e-9,rtol=1e-9)
    np.testing.assert_allclose(p['tail_variance'],(trace-p['eigenvalues'].sum())/(d-59),atol=1e-12,rtol=1e-12)
    iso=g.gm.load_parameters(root/'fit/isotropic.npz')
    assert iso['basis'].shape==(d,0) and np.array_equal(iso['mean'],mu)
    np.testing.assert_allclose(iso['tail_variance'],trace/d,rtol=1e-12,atol=1e-12)
    assert g.gm.stored_scalars(p)==184380 and g.gm.stored_scalars(iso)==3073
    spectrum=read('retained-spectrum.csv');assert len(spectrum)==59
    for i,r in enumerate(spectrum):assert int(r['index'])==i+1 and float(r['eigenvalue'])==p['eigenvalues'][i]
    summary=g.load(doc/'results/summary.json')
    assert summary['rank']==59 and summary['stored_float64_bytes']==1475040
    assert summary['retained_variance_fraction']==float(p['eigenvalues'].sum()/trace)
    decision=g.load(doc/'results/decision.json');assert decision==g.decisions(formal)
    assert not decision['new_ODE_method'] and not decision['neural_mechanism_identified'] and not decision['rank_or_noise_rescue_selected']
    assert all(not v[k] for v in decision['rows'] for k in ('fitting_guard','allocation_margin','strict_Swi86_SiLU131','strict_default_SiLU64'))
    # Independent algebra: complement energy is total minus projected energy,
    # rather than the producer's explicit residual reconstruction.
    subspaces=read('subspace-analysis/subspaces.csv');assert len(subspaces)==17
    reference=np.load(root/'sampling/ppca59-generated.npy').reshape(1024,d).astype(np.float64)-mu
    ref_coeff=reference@u;maximum=0.
    def check(observed,wanted):
        nonlocal maximum
        maximum=max(maximum,abs(float(observed)-wanted));np.testing.assert_allclose(float(observed),wanted,rtol=1e-12,atol=1e-12)
    for row in subspaces:
        arm=row['arm'];real=arm=='real_reference'
        images=g.gm.normalize_pixels(np.load(g.NATIVE/'banks/fid-reference.uint8.npy')) if real else np.load(by[arm,0 if arm in g.ARMS else 20000]['sample_path'])
        x=images.reshape(len(images),d).astype(np.float64)-mu;coeff=x@u;n=len(x)
        retained=float(np.einsum('ij,ij->',coeff,coeff)/(n*d));total=float(np.einsum('ij,ij->',x,x)/(n*d));discarded=total-retained
        assert int(row['images'])==n
        for name,value in [('retained_energy_per_coordinate',retained),('discarded_energy_per_coordinate',discarded),('total_energy_per_coordinate',total),('retained_RMS',np.sqrt(retained)),('discarded_RMS',np.sqrt(discarded))]:check(row[name],float(value))
        if real:assert all(row[k]=='' for k in ('coupled_difference_total','discarded_fraction_of_coupled_difference','retained_centered_correlation_to_PPCA'));continue
        delta=x-reference;dc=coeff-ref_coeff
        all_diff=float(np.einsum('ij,ij->',delta,delta)/(n*d));ret_diff=float(np.einsum('ij,ij->',dc,dc)/(n*d));comp_diff=all_diff-ret_diff
        for k,v in [('coupled_difference_total',all_diff),('coupled_difference_retained',ret_diff),('coupled_difference_discarded',comp_diff)]:check(row[k],v)
        if arm=='ppca59':assert all_diff==0 and row['discarded_fraction_of_coupled_difference']==''
        else:check(row['discarded_fraction_of_coupled_difference'],comp_diff/all_diff)
        a=coeff-coeff.mean(0);b=ref_coeff-ref_coeff.mean(0)
        correlation=float(np.einsum('ij,ij->',a,b)/np.sqrt(np.einsum('ij,ij->',a,a)*np.einsum('ij,ij->',b,b)))
        check(row['retained_centered_correlation_to_PPCA'],correlation)
    observed=g.load(root/'reporting-subspaces/producer-observation.json')
    assert observed['original_exit_code'] is None and not observed['producer_relaunched']
    reporting=g.load(root/'reporting-subspaces/summary.json')
    assert reporting['status']=='PASS' and reporting['reporting_added_after_outcomes']
    assert not reporting['fitted_or_generated_or_scored_new_samples'] and not reporting['neural_mechanism_identified']
    # The original reporting exit is unavailable. This complete independent
    # check validates its numbers without inventing a producer exit receipt.
    g.write(doc/'results/subspace-verification.json',dict(status='PASS',rows=17,independent_algebra='total minus projected energy',maximum_absolute_difference=maximum,
        original_reporting_tool_exit_observed=False,producer_relaunched=False,new_fitting_or_sampling_or_FID=False))
    resources=[json.loads(v) for v in (root/'orchestrator/resources.jsonl').read_text().splitlines()]
    for r,phase in zip(read('resource-summary.csv'),phases):
        assert r['phase']==phase['phase'] and float(r['seconds'])==phase['seconds']
        assert int(r['peak_CPU_RSS_bytes'])==max(v['RSS'] for q in resources if q['phase']==r['phase'] for v in q['resources'])
    labels={v:k for k,v in curator.LABELS.items()};reported=[]
    for line in (doc/'analysis-report.md').read_text().splitlines():
        cells=[v.strip().replace('**','') for v in line.strip().strip('|').split('|')]
        if len(cells)==4 and cells[0] in labels:reported.append(cells)
    assert len(reported)==16
    for label,*values in reported:
        arm=labels[label];actual=by[arm,0 if arm in g.ARMS else 20000]
        assert values==[format(actual[k],fmt) for k,fmt in [('weighted_mse','.6f'),('clean_mse','.6f'),('fid','.2f')]]
    review=g.load(root/'visual-review.json');assert review['status']=='PASS' and len(review['files'])==5 and review['fixed_examples']==128
    for name,digest in review['files'].items():assert g.sha(root/name)==digest
    assert len(list((doc/'results/figures').glob('*')))==8
    assert len(list(csv.DictReader((doc/'results/results.tsv').open(),delimiter='\t')))==34
    protected=['src','synthetic','configs','scripts','reports/report1']
    assert not subprocess.check_output(['git','-C',str(repo),'diff','--name-only','98730e0','--',*protected],text=True)
    links=0
    for path in [*doc.glob('*.md'),repo/'README.md',doc.parent/'README.md']:
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',path.read_text()):
            target=target.strip('<>').split('#')[0]
            if not target or '://' in target or target.startswith(('/','mailto:')):continue
            resolved=(path.parent/unquote(target)).resolve();assert resolved.exists(),(path,target)
            assert subprocess.check_output(['git','-C',str(repo),'ls-files','--',str(resolved.relative_to(repo))],text=True).strip(),(path,'untracked',target)
            links+=1
    for path in doc.glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    for path,digest in preserved.items():assert g.sha(path)==digest
    g.verify(root)
    g.write(doc/'results/artifact-verification.json',dict(status='PASS',formal_phases=6,endpoints=34,primary_rows=16,ratios=28,time_panels=35,
        generated_endpoints_independently_checked=2048,sampling_replay='registered numerical tolerances, not all bitwise',new_features_bitwise=2048,reference_features_bitwise=2048,
        original_array_subspace_rows=17,reporting_original_tool_exit_observed=False,figures_visually_reviewed=5,fixed_examples=128,report_rows_verified=16,
        local_links=links,protected_scopes=protected,new_ODE_method=False,neural_mechanism_identified=False))
    shutil.copytree(doc/'results',root/'curated-results');closure=root/'closure-source';closure.mkdir()
    for path in doc.iterdir():
        if path.is_file():shutil.copyfile(path,closure/path.name)
    files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='sealed-runtime.json')
    g.write(root/'sealed-runtime.json',{str(p.relative_to(root)):g.sha(p) for p in files})
    for name,digest in g.load(root/'sealed-runtime.json').items():assert g.sha(root/name)==digest
    print(json.dumps(dict(status='SEALED_AND_VERIFIED',files=len(files),seal_sha256=g.sha(root/'sealed-runtime.json'),private_bytes=sum(p.stat().st_size for p in files),
        curated_bytes=sum(p.stat().st_size for p in (doc/'results').rglob('*') if p.is_file()),reporting_original_tool_exit_observed=False)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);main(parser.parse_args().root.resolve())
