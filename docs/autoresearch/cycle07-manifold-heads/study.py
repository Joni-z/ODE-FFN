"""Fixed paired-geometry/head compatibility screen using unchanged native FFNs."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import torch

FAMILIES=('ode_k2','swiglu128','swiglu192')
HEADS=('clean','shifted','direct')
ARMS=tuple(f'{family}_{head}' for family in FAMILIES for head in HEADS)
STEPS=(0,1000,2500,5000)
TIMES=(.1,.3,.5,.7,.9)


def load(path: Path) -> dict:return json.loads(path.read_text())


def write(path: Path,value: object) -> None:
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)


def append(path: Path,value: object) -> None:
    with path.open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')


def setup(root: Path):
    settings=load(root/'settings.json');sys.path.insert(0,settings['joint_source'])
    import joint_study as native
    native.setup(root);torch.set_num_threads(1);sys.path.insert(0,str(root/'source'))
    import benchmark_core as core
    assert Path(core.__file__).resolve()==root/'source/benchmark_core.py'
    return settings,native,core


def entry_for(root: Path,law: str) -> dict:
    return next(row for row in load(root/'settings.json')['laws'] if row['law']==law)


def build(root: Path,law: str,arm: str):
    _,native,core=setup(root);family,head=arm.rsplit('_',1)
    return core.CappedHead(native.old_build(core.DIM,family,'direct').raw,head)


def training_bank(entry: dict,native) -> torch.Tensor:
    path=Path(entry['train'])
    return torch.from_numpy(np.load(path) if path.suffix=='.npy' else native.legacy.load_targets(path,entry['dim']))


def prepare(root: Path,repo: Path) -> None:
    root.mkdir(parents=True,exist_ok=False);source=root/'source';source.mkdir()
    for name in ('study.py','verify_core.py','protocol.md','theory.md'):shutil.copyfile(Path(__file__).parent/name,source/name)
    shutil.copyfile(repo/'synthetic/manifold_benchmark.py',source/'benchmark_core.py')
    previous=root.parent;settings=load(previous/'oct27-cycle05/settings.json')
    settings.update(repo=str(repo),laws=[],prior_seals=[str(previous/f'oct27-cycle{number:02d}/sealed-runtime.json') for number in range(1,7)],
        registration_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip())
    write(root/'settings.json',settings);_,native,core=setup(root);basis=core.make_basis();np.save(root/'basis.npy',basis)
    def rng(role):return np.random.default_rng(np.random.SeedSequence([0,20261028,role]))
    training=core.paired_targets(rng(702),65536,basis)
    validation=core.paired_targets(rng(703),20000,basis)
    probes=core.paired_targets(rng(704),32,basis)
    samples=core.paired_targets(rng(705),2048,basis);samples2=core.paired_targets(rng(706),2048,basis)
    r=rng(707);field_t=r.random((20000,1)).astype(np.float32);field_source=r.standard_normal((20000,128)).astype(np.float32)
    r=rng(708);probe_t=r.uniform(.1,.9,(32,1)).astype(np.float32);probe_source=r.standard_normal((32,128)).astype(np.float32)
    r=rng(709);sampling_source=r.standard_normal((2048,128)).astype(np.float32)
    directions=r.standard_normal((128,128));directions=(directions/np.linalg.norm(directions,axis=1,keepdims=True)).astype(np.float32)
    for index,(law,variance) in enumerate((('thin',0.),('thick',.04))):
        banks=root/'banks'/law;banks.mkdir(parents=True);train=root/f'{law}-train.npy';np.save(train,training[index])
        entry=dict(law=law,dim=128,normal_variance=variance,banks=str(banks),train=str(train));settings['laws'].append(entry)
        target=validation[index];x=(1-field_t)*field_source+field_t*target
        truth=core.oracle(x,field_t,basis,variance)
        pair=(1-field_t)/np.maximum(1-field_t,.05)*(target-field_source)
        np.savez_compressed(banks/'field.npz',x=x,t=field_t,oracle=truth['oracle'],clean=truth['clean_mean'],pair=pair)
        px=(1-probe_t)*probe_source+probe_t*probes[index]
        np.savez_compressed(banks/'probe.npz',x=px,t=probe_t)
        np.savez_compressed(banks/'sampling.npz',source=sampling_source,target=samples[index],target2=samples2[index],directions=directions)
        for clock in TIMES:
            t=np.full((64,1),clock,dtype=np.float32);z=rng(710+int(clock*10)).standard_normal((64,128)).astype(np.float32)
            x=(1-t)*z+t*target[:64];truth=core.oracle(x,t,basis,variance,jacobian=True)
            np.savez_compressed(banks/f'panel_{clock:.1f}.npz',x=x,t=t,oracle=truth['oracle'],oracle_jacobian=truth['oracle_jacobian'])
        class Oracle:
            def __call__(self,x,t):return torch.from_numpy(core.oracle(x.numpy(),t.numpy(),basis,variance)['oracle']).to(x.dtype)
        source_tensor=torch.from_numpy(sampling_source);generated=native.legacy.generated_samples(Oracle(),source_tensor,32)
        np.savez_compressed(banks/'oracle-samples.npz',generated=generated.numpy())
        write(banks/'oracle-sampling.json',dict(sw2=native.legacy.sliced_w2(generated,torch.from_numpy(samples[index]),torch.from_numpy(directions)),
            normal_sample_energy_per_dimension=float(np.mean((generated.numpy()-(generated.numpy()@basis)@basis.T)**2)),
            capped_field_not_exact_probability_flow=True,continuous_thin_normal_std=.05/np.e))
    write(root/'settings.json',settings)
    paths=list(source.rglob('*'))+[root/'settings.json',root/'basis.npy',repo/'docs/autoresearch/program.md']
    paths+=list(Path(settings['native_root']).rglob('*.py'))+list(Path(settings['joint_source']).glob('*.py'))
    paths+=list((root/'banks').rglob('*'))+list(root.glob('*-train.npy'))+[Path(p) for p in settings['prior_seals']]
    paths=sorted({p for p in paths if p.is_file() and p.suffix!='.pyc'})
    write(root/'bindings.json',{str(path):native.sha(path) for path in paths})
    print(dict(prepared=str(root),bindings=len(paths),cells=18),flush=True)


def verify_inputs(root: Path,native) -> None:
    for path,digest in load(root/'bindings.json').items():assert native.sha(path)==digest,path
    for seal in load(root/'settings.json')['prior_seals']:
        seal=Path(seal)
        for name,digest in load(seal).items():assert native.sha(seal.parent/name)==digest,name


def update(model,opt,x,t,target,source,core):
    opt.zero_grad(set_to_none=True);label=core.regression_target(target,source,t)
    loss=(model(x,t)-label).square().mean();assert torch.isfinite(loss)
    loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(gradient)
    opt.step();return float(loss.detach()),float(gradient)


def preflight(root: Path) -> None:
    settings,native,core=setup(root);verify_inputs(root,native)
    from verify_core import verify
    mathematics=verify(core);basis=np.load(root/'basis.npy');rows=[];geometry=[]
    train=[training_bank(entry,native).numpy().astype(np.float64) for entry in settings['laws']]
    np.testing.assert_allclose(train[0]@basis,train[1]@basis,rtol=1e-5,atol=5e-7)
    normal=train[0]-(train[0]@basis)@basis.T
    assert np.max(np.abs(normal))<3e-7
    for entry,data in zip(settings['laws'],train):
        covariance=np.cov(data,rowvar=False);eigenvalues=np.linalg.eigvalsh(covariance)
        geometry.append(dict(law=entry['law'],numerical_covariance_rank=int(np.sum(eigenvalues>1e-9)),
            normal_energy_per_coordinate=float(np.mean((data-(data@basis)@basis.T)**2)),
            thin_float32_max_normal_residual=float(np.max(np.abs(normal)))))
        bank=training_bank(entry,native)
        for arm in ARMS:
            family,head=arm.rsplit('_',1);model=build(root,entry['law'],arm);base=build(root,entry['law'],family+'_direct')
            for name,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[name]),name
            for name,value in base.named_buffers():assert torch.equal(value,dict(model.named_buffers())[name]),name
            ends=[]
            for resume in (False,True):
                model=build(root,entry['law'],arm);opt=native.optimizer(model);rng=torch.Generator().manual_seed(100000);digest=native.EMPTY
                for step in range(1,4):
                    x,t,a,b=native.draw(bank,rng);digest=native.legacy.fingerprint_batch(digest,a,b,t);update(model,opt,x,t,a,b,core)
                    if resume and step==2:
                        state=native.checkpoint(model,opt,rng,step,digest);model=build(root,entry['law'],arm);opt=native.optimizer(model)
                        digest=native.restore(model,opt,rng,state)
                ends.append(native.checkpoint(model,opt,rng,3,digest))
            native.equal(*ends);cost=native.compute_budget(model)
            expected={'ode_k2':(98310,121280,353536),'swiglu128':(98046,96128,278080),'swiglu192':(123072,120896,352384)}[family]
            assert tuple(cost[k] for k in ('parameters','dense_forward_macs','dense_training_forward_backward_macs'))==expected,(arm,cost)
            rows.append(dict(law=entry['law'],arm=arm,raw_pairing=True,exact_resume=True,**cost))
    verify_inputs(root,native);write(root/'preflight.json',dict(status='PASS',mathematics=mathematics,cells=rows,geometry=geometry))
    write(root/'ready.json',dict(bindings_sha256=native.sha(root/'bindings.json'),preflight_sha256=native.sha(root/'preflight.json')))
    print(dict(status='PASS',cells=len(rows),mathematics=mathematics,geometry=geometry),flush=True)


def field_parts(root: Path,banks: Path,prediction: Path) -> dict:
    basis=np.load(root/'basis.npy')
    with np.load(prediction) as data:pred=data['prediction'].astype(np.float64)
    with np.load(banks/'field.npz') as data:truth=data['oracle'];t=data['t'];x=data['x'];clean=data['clean']
    error=pred-truth;parallel=error@basis;normal=error-parallel@basis.T
    denoised=x+np.maximum(1-t,.05)*pred
    return dict(tangent_mse_per_dimension=float(np.mean(parallel**2)),normal_mse_per_dimension=float(np.sum(normal**2)/(len(normal)*126)),
        clean_oracle_mse=float(np.mean((denoised-clean)**2)))


def generator_visits(model,banks: Path) -> list[dict]:
    visits=[]
    for name,layer in model.ode_layers():
        p=layer.proj.weight.detach().double().numpy();eigen=np.linalg.eigvals(p);singular=np.linalg.svd(p,compute_uv=False)
        visits.append(dict(layer=name,P_singular_values=singular.tolist(),P_eigenvalues_real=eigen.real.tolist(),
            P_eigenvalues_imag=eigen.imag.tolist(),P_frobenius=float(np.linalg.norm(p))))
    return visits


def cell(root: Path,law: str,arm: str) -> None:
    _,native,core=setup(root);entry=entry_for(root,law);folder=root/'output/runs'/law/arm;folder.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def status(state='RUNNING',step=0,**extra):write(folder/'status.json',dict(state=state,step=step,pid=os.getpid(),elapsed_seconds=time.monotonic()-started,**extra))
    status()
    try:
        model=build(root,law,arm);opt=native.optimizer(model);rng=torch.Generator().manual_seed(100000);digest=native.EMPTY
        bank=training_bank(entry,native);banks=Path(entry['banks']);cost=native.compute_budget(model)
        write(folder/'provenance.json',dict(law=law,arm=arm,model_seed=0,training_stream=100000,steps=5000,batch=512,**cost))
        def save(step):
            torch.save(native.checkpoint(model,opt,rng,step,digest),folder/f'step_{step:07d}.pt')
            write(folder/f'generator_{step:07d}.json',generator_visits(model,banks))
            metrics=native.legacy.field_evaluation(model,banks,save=folder/f'field_{step:07d}.npz')
            metrics.update(field_parts(root,banks,folder/f'field_{step:07d}.npz'))
            write(folder/f'clocks_{step:07d}.json',native.legacy.visit_diagnostics(model,banks))
            return metrics
        append(folder/'metrics.jsonl',dict(step=0,**save(0)))
        ls=gs=0.;n=clipped=0
        for step in range(1,5001):
            x,t,a,b=native.draw(bank,rng);digest=native.legacy.fingerprint_batch(digest,a,b,t)
            loss,gradient=update(model,opt,x,t,a,b,core);ls+=loss;gs+=gradient;n+=1;clipped+=int(gradient>1)
            if step%100==0:status(step=step);assert time.monotonic()-started<600,'registered cell timeout'
            if step%500==0:
                row=dict(step=step,train_loss=ls/n,grad_norm=gs/n,clip_fraction=clipped/n)
                if step in STEPS:row.update(save(step))
                append(folder/'metrics.jsonl',row);ls=gs=0.;n=clipped=0
        result=native.legacy.diagnostics(model,banks,folder/'diagnostics',rollout=False)
        result.update(field_parts(root,banks,folder/'diagnostics/field.npz'))
        basis=np.load(root/'basis.npy')
        with np.load(folder/'diagnostics/samples.npz') as data:generated=data['generated']
        with np.load(banks/'sampling.npz') as data:target=data['target']
        projected=torch.tensor(generated@basis,dtype=torch.float32);truth=torch.tensor(target@basis,dtype=torch.float32)
        angles=torch.arange(128)*2*torch.pi/128;directions=torch.stack((torch.cos(angles),torch.sin(angles)),-1)
        result.update(tangent_sw2=native.legacy.sliced_w2(projected,truth,directions),
            normal_sample_energy_per_dimension=float(np.mean((generated-(generated@basis)@basis.T)**2)),
            capped_field_not_exact_probability_flow=True)
        write(folder/'result.json',dict(law=law,arm=arm,input_stream_digest=digest,**cost,**result));status('COMPLETE',5000)
    except BaseException:status('FAILED',error=traceback.format_exc());raise


def run(root: Path) -> None:
    settings,native,core=setup(root);ready=load(root/'ready.json')
    assert native.sha(root/'bindings.json')==ready['bindings_sha256'] and native.sha(root/'preflight.json')==ready['preflight_sha256']
    assert load(root/'preflight.json')['status']=='PASS';verify_inputs(root,native)
    out=root/'output';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    started=time.monotonic();stop=threading.Event();abort=threading.Event();lock=threading.Lock();children=set()
    status=dict(state='RUNNING',completed=0,total=18,pid=os.getpid())
    def progress(**extra):status.update(extra,elapsed_seconds=time.monotonic()-started);write(out/'status.json',status);print(status,flush=True)
    def monitor():
        import psutil
        while not stop.is_set():
            parent=psutil.Process();rows=[]
            for process in [parent]+parent.children(recursive=True):
                try:rows.append(dict(pid=process.pid,rss_bytes=process.memory_info().rss,cpu_seconds=sum(process.cpu_times()[:2])))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            append(out/'monitor.jsonl',dict(elapsed=time.monotonic()-started,processes=rows));stop.wait(5)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();progress()
    environment=os.environ.copy();environment.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    def launch(law,arm):
        command=[sys.executable,str(Path(__file__)),'cell','--root',str(root),'--law',law,'--arm',arm]
        with (out/f'{law}-{arm}.log').open('w') as log:
            with lock:
                if abort.is_set():raise RuntimeError('study aborted')
                process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT);children.add(process)
                append(out/'launches.jsonl',dict(law=law,arm=arm,pid=process.pid,command=command))
            try:code=process.wait(timeout=600)
            except subprocess.TimeoutExpired:process.terminate();process.wait();raise
            finally:
                with lock:children.remove(process)
        assert code==0,(law,arm,code)
        return law+'/'+arm
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(launch,entry['law'],arm) for entry in settings['laws'] for arm in ARMS]
            try:
                for future in as_completed(futures):progress(completed=status['completed']+1,latest=future.result())
            except BaseException:
                abort.set()
                for future in futures:future.cancel()
                with lock:
                    for process in children:process.terminate()
                raise
        verify_inputs(root,native);progress(state='COMPLETE')
    except BaseException:progress(state='FAILED',error=traceback.format_exc());raise
    finally:stop.set();watcher.join(timeout=6)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','preflight','run','cell'));p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo',type=Path);p.add_argument('--law');p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    if a.action=='prepare':prepare(a.root,a.repo)
    elif a.action=='preflight':preflight(a.root)
    elif a.action=='run':run(a.root)
    else:cell(a.root,a.law,a.arm)
