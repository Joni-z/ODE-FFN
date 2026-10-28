"""Source-bound classical Gaussian controls on the existing image screen."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,hashlib,importlib.util,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import psutil
import scipy.linalg
import torch
import gaussian_math as gm
if __name__=='__main__':sys.modules['gaussian_reference']=sys.modules[__name__]

BASE=Path('/Users/mr.z/ode-workspace-prep/autoresearch')
COLD=BASE/'oct28-image-clock-training-v2'
NATIVE=BASE/'oct28-cycle17-v2'
ELIGIBILITY=BASE/'oct28-gaussian-eligibility'
COLD_SEAL='aad29b214ba537a2ee2034c60987c98f220eef00ca97b0105fb2e3bbadffff9c'
PYTHON='/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python'
ARMS=('ppca59','isotropic')
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)
def append(p,value):
    with Path(p).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
def module(name,path):
    path=Path(path)
    if name not in sys.modules:
        sys.path.insert(0,str(path.parent));spec=importlib.util.spec_from_file_location(name,path)
        m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    assert Path(sys.modules[name].__file__).resolve()==path.resolve()
    return sys.modules[name]
def cold():return module('gaussian_parent_cold',COLD/'source/training_controls.py')
def native():return module('gaussian_parent_native',NATIVE/'source/study.py')
def resolution():return module('gaussian_parent_resolution',BASE/'oct28-cycle18/source/study.py')
def metric_modules():
    sys.path.insert(0,str(NATIVE/'source'))
    import image_metrics,metric_core
    assert Path(image_metrics.__file__).resolve()==NATIVE/'source/image_metrics.py'
    assert Path(metric_core.__file__).resolve()==NATIVE/'source/metric_core.py'
    return image_metrics,metric_core
def verify(root):
    for path,digest in load(root/'bindings.json').items():assert sha(path)==digest,path
    assert sha(COLD/'sealed-runtime.json')==COLD_SEAL
    for name,digest in load(COLD/'sealed-runtime.json').items():assert sha(COLD/name)==digest,name
    cold().verify(COLD);native().verify(NATIVE,True)
def prepare(root):
    repo=Path(__file__).resolve().parents[3]
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    assert not root.exists();root.mkdir();(root/'source').mkdir();(root/'doc-validation').mkdir()
    for p in Path(__file__).resolve().parent.iterdir():
        if p.is_file():shutil.copyfile(p,root/'source'/p.name)
    assert load(ELIGIBILITY/'fixtures.json')['status']=='PASS'
    receipt=load(ELIGIBILITY/'doc-validation/receipt.json');assert receipt['execution']['terminal_exit_code']==0
    references=load(COLD/'metric-output/result.json')['rows'];assert len(references)==32
    curator=module('gaussian_parent_curator',COLD/'curation-source/curate_training.py')
    references=[dict(r,feature_path=str(cold().feature_path(COLD,r['arm'],r['step'])),
                     sample_path=str(curator.sample_path(cold(),COLD,r['arm'],r['step'])),fit_kind='neural reference') for r in references]
    data=Path(load(NATIVE/'settings.json')['data'])
    settings=dict(source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        data=str(data),rank=gm.RANK,scalar_budget=gm.SCALAR_BUDGET,dimension=gm.DIMENSION,
        fit_batch=512,audit_batch=257,arms=ARMS,seed=0,references=references,threads=1,
        real_fit_backend='cpu',feature_backend='mps',fitted_statistics_dtype='float64',prediction_and_sampler_dtype='float32',
        new_ODE_method=False,neural_training_updates=0,reference_competitiveness_only=True)
    write(root/'settings.json',settings)
    files=[*(root/'source').iterdir(),root/'settings.json',COLD/'sealed-runtime.json',
           data/'images.uint8.npy',data/'split.npz',data/'manifest.json']
    files += [p for p in ELIGIBILITY.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    write(root/'bindings.json',{str(p.resolve()):sha(p) for p in files if p.is_file()})
    verify(root);print(json.dumps(dict(status='PREPARED',rank=gm.RANK,reference_rows=len(references))),flush=True)
def bank(t=None):return native().bank(NATIVE/'banks'/('validation.npz' if t is None else f'panel_{t:.1f}.npz'))
def predict(model,b):return native().predict(model,b)
def difference(a,b):
    d=a.astype(np.float64)-b.astype(np.float64)
    return dict(maximum=float(np.max(np.abs(d))),RMS=float(np.sqrt(np.mean(d*d))))
class IndependentPredictor(torch.nn.Module):
    def __init__(self,p):super().__init__();self.p=p
    def forward(self,x,t,labels=None):return torch.from_numpy(gm.independent_mean(self.p,x.numpy(),t.numpy()).astype(np.float32))
def manual(model,source,progress=None):
    answers=[]
    with torch.no_grad():
        for begin in range(0,len(source),64):
            x=torch.from_numpy(np.array(source[begin:begin+64]));labels=torch.zeros(len(x),dtype=torch.long)
            for j in range(128):
                t=torch.full((len(x),),j/128);nt=torch.full_like(t,(j+1)/128)
                first=(model(x,t,labels)-x)/(1-t[:,None,None,None]).clamp_min(.05)
                trial=x+first/128;second=(model(trial,nt,labels)-trial)/(1-nt[:,None,None,None]).clamp_min(.05)
                x=x+(first+second)/256
            assert torch.isfinite(x).all();answers.append(x.numpy())
            if progress:progress(begin+len(x))
    return np.concatenate(answers)
def preflight(root):
    torch.set_num_threads(1);verify(root);tick=time.monotonic();out=root/'preflight-work';out.mkdir()
    primitive=gm.fixtures();rng=np.random.default_rng(0);d,r=gm.DIMENSION,gm.RANK
    p=gm.validate(dict(mean=np.linspace(-.1,.1,d),basis=np.eye(d,r),eigenvalues=np.linspace(1.7,.8,r),tail_variance=.3))
    gm.save_parameters(out/'fixture-prior.npz',p)
    source=np.load(NATIVE/'banks/sampling-source.npy')[:64]
    observed=resolution().integrate(gm.GaussianPredictor(p).eval(),source,128)
    restored=resolution().integrate(gm.GaussianPredictor(gm.load_parameters(out/'fixture-prior.npz')).eval(),source,128)
    assert observed.tobytes()==restored.tobytes()
    independent=manual(IndependentPredictor(p).eval(),source);error=difference(observed,independent)
    assert error['maximum']<3e-5 and error['RMS']<3e-6,error
    from torch.utils._python_dispatch import TorchDispatchMode
    class Count(TorchDispatchMode):
        def __init__(self):self.macs=0
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            answer=func(*args,**(kwargs or {}))
            if func==torch.ops.aten.mm.default:self.macs+=args[0].shape[0]*args[0].shape[1]*args[1].shape[1]
            return answer
    x=torch.randn(2,3,32,32,generator=torch.Generator().manual_seed(0));t=torch.tensor([.3,.7])
    with Count() as count:gm.GaussianPredictor(p)(x,t)
    assert count.macs//2==362496
    pixels=rng.integers(0,256,(83,3,4,4),dtype=np.uint8)
    mean,cov=gm.stream_statistics(pixels,np.arange(83),17)
    all_values=gm.normalize_pixels(pixels)
    np.testing.assert_allclose(mean,all_values.mean(0),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(cov,np.cov(all_values,rowvar=False,ddof=0),rtol=1e-12,atol=1e-12)
    fake=rng.integers(0,256,(128,3,32,32),dtype=np.uint8);start=time.monotonic()
    fm,fc=gm.stream_statistics(fake,np.arange(128),64);fp=gm.from_covariance(fm,fc,r)
    assert gm.stored_scalars(fp)==184380;kernel_seconds=time.monotonic()-start
    m,metric=metric_modules();ex=m.extractor(NATIVE)
    reference=np.load(NATIVE/'metric-inputs/reference-features.npy')
    rows=load(root/'settings.json')['references'];row=next(v for v in rows if v['arm']=='ode_k1' and v['step']==20000)
    features=m.features(ex,metric.uint8_images(np.load(row['sample_path'])[:32]))
    assert features.tobytes()==np.load(row['feature_path'])[:32].tobytes()
    for row in rows:assert abs(metric.fid(np.load(row['feature_path']),reference)-row['fid'])<1e-8
    verify(root);write(root/'preflight.json',dict(status='PASS',primitive=primitive,image_shape_native_Heun_samples=64,
        physical_prior_and_native_sampler_bitwise=True,independent_Heun_error=error,PPCA_matrix_MACs_per_image=362496,
        covariance_normalization_fixture=True,actual_shape_fake_fit_seconds=kernel_seconds,
        historical_features_bitwise=32,historical_FIDs_recomputed=32,real_statistics_fitted=False,seconds=time.monotonic()-tick))
    print(json.dumps(load(root/'preflight.json')),flush=True)
def fit(root):
    verify(root);assert load(root/'documented-pass.json')['status']=='PASS';out=root/'fit';out.mkdir();tick=time.monotonic()
    s=load(root/'settings.json');data=Path(s['data']);images=np.load(data/'images.uint8.npy',mmap_mode='r')
    with np.load(data/'split.npz') as f:ids=f['train'].copy();validation=f['validation'].copy()
    assert len(ids)==45000 and len(validation)==5000 and len(np.intersect1d(ids,validation))==0
    def progress(phase,n,total):write(out/'progress.json',dict(phase=phase,processed=n,total=total,seconds=time.monotonic()-tick))
    mean,cov=gm.stream_statistics(images,ids,512,progress);np.save(out/'training-covariance.npy',cov)
    rows=[]
    for arm,rank in (('ppca59',59),('isotropic',0)):
        parameters=gm.from_covariance(mean,cov,rank);gm.save_parameters(out/f'{arm}.npz',parameters)
        rows.append(dict(arm=arm,rank=rank,stored_scalars=gm.stored_scalars(parameters),stored_float64_bytes=gm.stored_scalars(parameters)*8,
            tail_variance=parameters['tail_variance'],trace=float(np.trace(cov)),matrix_MACs_per_image=2*gm.DIMENSION*rank))
    verify(root);write(out/'result.json',dict(status='PASS',rows=rows,train_images=45000,validation_used_for_fit=False,
        MLE_denominator=45000,train_indices_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),seconds=time.monotonic()-tick))
def audit_fit(root):
    verify(root);out=root/'audit-fit';out.mkdir();tick=time.monotonic();s=load(root/'settings.json');data=Path(s['data'])
    before={str(p):sha(p) for p in (root/'fit').rglob('*') if p.is_file()}
    images=np.load(data/'images.uint8.npy',mmap_mode='r')
    with np.load(data/'split.npz') as f:ids=f['train'].copy()
    first=np.zeros(gm.DIMENSION,np.float64);second=np.zeros((gm.DIMENSION,gm.DIMENSION),np.float64)
    for begin in range(0,len(ids),257):
        pixels=np.array(images[ids[begin:begin+257]])
        x=(torch.from_numpy(pixels).float()/127.5-1).numpy().reshape(len(pixels),-1).astype(np.float64)
        first+=x.sum(0);second+=x.T@x
        write(out/'progress.json',dict(processed=min(begin+257,len(ids)),total=len(ids)))
    mean=first/len(ids);cov=second/len(ids)-np.outer(mean,mean);cov=(cov+cov.T)*.5
    np.save(out/'independent-covariance.npy',cov)
    stored=np.load(root/'fit/training-covariance.npy');np.testing.assert_allclose(cov,stored,rtol=1e-10,atol=1e-10)
    all_eigen=np.linalg.eigvalsh(cov)[::-1];rows=[]
    for arm,rank in (('ppca59',59),('isotropic',0)):
        p=gm.load_parameters(root/f'fit/{arm}.npz');np.testing.assert_allclose(p['mean'],mean,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(p['eigenvalues'],all_eigen[:rank],rtol=1e-9,atol=1e-10)
        np.testing.assert_allclose(p['tail_variance'],all_eigen[rank:].mean(),rtol=1e-9,atol=1e-10)
        eigen_error=float(np.max(np.abs(cov@p['basis']-p['basis']*p['eigenvalues']))) if rank else 0.
        assert eigen_error<1e-9
        rows.append(dict(arm=arm,rank=rank,mean_max_error=float(np.max(np.abs(p['mean']-mean))),eigen_residual_max=eigen_error,
            tail_variance=p['tail_variance'],full_covariance_max_error=float(np.max(np.abs(cov-stored)))))
    for p,digest in before.items():assert sha(p)==digest
    verify(root);write(out/'result.json',dict(status='PASS',rows=rows,independent_raw_moment_batch=257,train_images=45000,formal_files_unchanged=len(before),seconds=time.monotonic()-tick))
def sample(root):
    torch.set_num_threads(1);verify(root);assert load(root/'audit-fit/result.json')['status']=='PASS'
    out=root/'sampling';out.mkdir();tick=time.monotonic();rows=[]
    source=np.load(NATIVE/'banks/sampling-source.npy');validation=bank()
    for arm in ARMS:
        model=gm.GaussianPredictor(gm.load_parameters(root/f'fit/{arm}.npz')).eval()
        prediction=predict(model,validation);np.savez_compressed(out/f'{arm}-validation.npz',prediction=prediction)
        metrics=native().metrics(prediction,validation)
        for t in (.1,.3,.5,.7,.9):np.savez_compressed(out/f'{arm}-panel-{t:.1f}.npz',prediction=predict(model,bank(t)))
        def progress(n):write(out/'progress.json',dict(completed=len(rows),total=2,arm=arm,images=n,seconds=time.monotonic()-tick))
        start=time.monotonic();generated=resolution().integrate(model,source,128,progress)
        np.save(out/f'{arm}-generated.npy',generated)
        rows.append(dict(arm=arm,**metrics,raw_tail_fraction=float(np.mean(np.abs(generated)>1)),sampling_seconds=time.monotonic()-start))
    verify(root);write(out/'result.json',dict(status='PASS',rows=rows,seconds=time.monotonic()-tick))
def audit_sampling(root):
    torch.set_num_threads(1);verify(root);out=root/'audit-sampling';out.mkdir();tick=time.monotonic();rows=[]
    source=np.load(NATIVE/'banks/sampling-source.npy')
    before={str(p):sha(p) for folder in ('fit','sampling') for p in (root/folder).rglob('*') if p.is_file()}
    for arm in ARMS:
        model=IndependentPredictor(gm.load_parameters(root/f'fit/{arm}.npz')).eval();errors=[]
        for t in (None,.1,.3,.5,.7,.9):
            b=bank(t);prediction=predict(model,b)
            path=root/'sampling'/(f'{arm}-validation.npz' if t is None else f'{arm}-panel-{t:.1f}.npz')
            with np.load(path) as f:stored=f['prediction'].copy()
            np.testing.assert_allclose(prediction,stored,rtol=3e-6,atol=3e-6);errors.append(difference(prediction,stored))
        def progress(n):write(out/'progress.json',dict(completed=len(rows),total=2,arm=arm,images=n))
        generated=manual(model,source,progress);np.save(out/f'{arm}-independent-generated.npy',generated)
        error=difference(generated,np.load(root/f'sampling/{arm}-generated.npy'))
        assert error['maximum']<3e-5 and error['RMS']<3e-6,error
        rows.append(dict(arm=arm,generated_images=1024,sample_difference=error,prediction_errors=errors))
    for p,digest in before.items():assert sha(p)==digest
    verify(root);write(out/'result.json',dict(status='PASS',rows=rows,generated_images_checked=2048,
        sample_check='independent posterior and manual Heun; registered numeric tolerances, not bitwise',seconds=time.monotonic()-tick))
def decisions(rows):
    by={(r['arm'],r['step']):r for r in rows};refs=[by[(a,20000)] for a in ('swiglu86','silu131')];out=[]
    for arm in ARMS:
        r=by[(arm,0)]
        fit=all(r[k]<=1.05*min(v[k] for v in refs) for k in ('weighted_mse','clean_mse'))
        margin=all(r['fid']<=.95*v['fid'] for v in refs)
        out.append(dict(arm=arm,fitting_guard=fit,allocation_margin=fit and margin,
            strict_Swi86_SiLU131=all(r['fid']<v['fid'] for v in refs),strict_default_SiLU64=r['fid']<by[('silu64',20000)]['fid'],
            reference_competitiveness_only=True))
    return dict(rows=out,new_ODE_method=False,neural_mechanism_identified=False,rank_or_noise_rescue_selected=False)
def metrics(root):
    torch.set_num_threads(1);verify(root);assert load(root/'audit-sampling/result.json')['status']=='PASS'
    out=root/'metric-output';out.mkdir();tick=time.monotonic();s=load(root/'settings.json')
    m,metric=metric_modules();reference=np.load(NATIVE/'metric-inputs/reference-features.npy');ex=m.extractor(NATIVE)
    rows=[]
    for r in s['references']:
        value=metric.fid(np.load(r['feature_path']),reference);assert abs(value-r['fid'])<1e-8
        rows.append(dict(r))
    fitted={r['arm']:r for r in load(root/'fit/result.json')['rows']}
    for r in load(root/'sampling/result.json')['rows']:
        arm=r['arm'];raw=np.load(root/f'sampling/{arm}-generated.npy');feature=m.features(ex,metric.uint8_images(raw),progress=arm)
        path=out/f'{arm}-features.npy';np.save(path,feature)
        rows.append(dict(r,step=0,fit_kind='train-only Gaussian MLE; no SGD budget',parameters=fitted[arm]['stored_scalars'],
            stored_float64_bytes=fitted[arm]['stored_float64_bytes'],rank=fitted[arm]['rank'],
            matrix_MACs_per_image=fitted[arm]['matrix_MACs_per_image'],feature_path=str(path),
            sample_path=str(root/f'sampling/{arm}-generated.npy'),fid=metric.fid(feature,reference),neural_training_updates=0))
    assert len(rows)==34
    for r in rows:
        a=np.load(r['feature_path']).astype(np.float64);ref=reference.astype(np.float64)
        r['FID_mean_term']=float(np.sum((a.mean(0)-ref.mean(0))**2));r['FID_covariance_term']=r['fid']-r['FID_mean_term']
    verify(root);write(out/'result.json',dict(status='PASS',rows=rows,seconds=time.monotonic()-tick));write(out/'decision.json',decisions(rows))
def audit_features(root):
    torch.set_num_threads(1);verify(root);out=root/'audit-features';out.mkdir();tick=time.monotonic()
    before={str(p):sha(p) for p in (root/'metric-output').rglob('*') if p.is_file()};m,metric=metric_modules();ex=m.extractor(NATIVE)
    ref=m.features(ex,np.load(NATIVE/'banks/fid-reference.uint8.npy'))
    assert ref.tobytes()==np.load(NATIVE/'metric-inputs/reference-features.npy').tobytes();rows=[]
    for r in load(root/'metric-output/result.json')['rows']:
        stored=np.load(r['feature_path'])
        if r['arm'] in ARMS:
            actual=m.features(ex,metric.uint8_images(np.load(r['sample_path'])));assert actual.tobytes()==stored.tobytes()
        a,b=stored.astype(np.float64),ref.astype(np.float64);x,y=a-a.mean(0),b-b.mean(0)
        singular=scipy.linalg.svd(x@y.T,compute_uv=False,lapack_driver='gesvd')
        value=float(np.sum((a.mean(0)-b.mean(0))**2)+np.sum(x*x)/(len(a)-1)+np.sum(y*y)/(len(b)-1)-2*singular.sum()/np.sqrt((len(a)-1)*(len(b)-1)))
        assert abs(value-r['fid'])<1e-8
        rows.append(dict(arm=r['arm'],step=r['step'],fid=value,difference=value-r['fid']));print(json.dumps(rows[-1]),flush=True)
    for p,digest in before.items():assert sha(p)==digest
    verify(root);write(out/'result.json',dict(status='PASS',rows=rows,new_features_bitwise=2048,reference_features_bitwise=2048,seconds=time.monotonic()-tick))
def run(root):
    verify(root);receipt=load(root/'documented-pass.json')
    assert receipt['status']=='PASS' and receipt['doc_vs_reality_mismatches']==[]
    assert receipt['preflight_sha256']==sha(root/'preflight.json') and load(root/'preflight.json')['status']=='PASS'
    out=root/'orchestrator';out.mkdir();me=psutil.Process();write(out/'identity.json',dict(pid=me.pid,create_time=me.create_time()))
    phases=(('fit',600),('audit-fit',600),('sample',300),('audit-sampling',300),('metrics',600),('audit-features',600))
    records=[];process=None
    try:
        for name,ceiling in phases:
            command=[PYTHON,'-B',str(root/'source/gaussian_reference.py'),name,'--root',str(root)]
            with (out/f'{name}.log').open('w') as f:process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
            monitor=psutil.Process(process.pid);monitor.cpu_percent();tracked={};tick=time.monotonic()
            record=dict(phase=name,command=command,pid=process.pid,create_time=monitor.create_time());records.append(record);write(out/'launches.json',records)
            while process.poll() is None:
                assert time.monotonic()-tick<ceiling;resources=[]
                try:children=[monitor,*monitor.children(recursive=True)]
                except psutil.NoSuchProcess:children=[]
                for p in children:
                    try:
                        key=(p.pid,p.create_time())
                        if key not in tracked:p.cpu_percent();tracked[key]=p;continue
                        p=tracked[key];resources.append(dict(pid=p.pid,create_time=key[1],CPU_percent=p.cpu_percent(),RSS=p.memory_info().rss))
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                append(out/'resources.jsonl',dict(phase=name,seconds=time.monotonic()-tick,resources=resources))
                write(out/'status.json',dict(state='RUNNING',phase=name,completed=len(records)-1,total=6,pid=process.pid,seconds=time.monotonic()-tick));time.sleep(5)
            record.update(exit_code=process.wait(),seconds=time.monotonic()-tick);write(out/'launches.json',records)
            assert record['exit_code']==0,record;print(json.dumps(record),flush=True)
        verify(root);write(out/'status.json',dict(state='COMPLETE',phases=6,new_ODE_method=False))
    except BaseException as error:
        if process and process.poll() is None:
            process.terminate();process.wait(timeout=30)
        write(out/'status.json',dict(state='FAILED',error=repr(error),records=records));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','preflight','fit','audit-fit','sample','audit-sampling','metrics','audit-features','run'))
    parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();torch.set_num_threads(1)
    globals()[args.action.replace('-','_')](args.root.resolve())
