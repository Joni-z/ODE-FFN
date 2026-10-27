"""One registered data-law control with unchanged native training semantics."""
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

from mixture import FullSupportMixture

FAMILIES=('ode_k2','free_isru','swiglu64','swiglu128','swiglu160')
SNAPSHOTS=(0,1000,2500,5000)
TIMES=(.1,.3,.5,.7,.9)


def load(p):return json.loads(Path(p).read_text())


def write(p,value):
    p=Path(p);temporary=p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temporary,p)


def append(p,value):
    with Path(p).open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')


def setup(root):
    settings=load(root/'settings.json');sys.path.insert(0,settings['joint_source'])
    import joint_study as reference
    reference.setup(root);torch.set_num_threads(1)
    assert Path(reference.__file__).resolve()==Path(settings['joint_source'])/'joint_study.py'
    return settings,reference


def target(root):
    spec=load(root/'data/distribution.json')
    return FullSupportMixture(spec['locations'],spec['rotation'],spec['signs'])


def prepare(root,repo,base):
    root.mkdir(parents=True,exist_ok=False);(root/'source').mkdir()
    for name in ('study.py','mixture.py','protocol.md'):
        shutil.copyfile(Path(__file__).parent/name,root/'source'/name)
    joint=base/'joint_dependence_20261026';settings=load(joint/'settings.json')
    settings.update(joint_source=str(joint/'source'),old_joint_root=str(joint),repo=str(repo),
        registration_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        old_cycle_runtime='/Users/mr.z/ode-workspace-prep/autoresearch/oct27-cycle01')
    write(root/'settings.json',settings);_,ref=setup(root)
    old=ref.distribution(joint);law=FullSupportMixture(old.locations,old.rotation,old.signs)
    (root/'data').mkdir();(root/'banks').mkdir();write(root/'data/distribution.json',law.specification())
    rng=np.random.default_rng(0);train=law.sample(rng,100000);validation=law.sample(rng,20000)
    np.save(root/'data/train.npy',train);np.save(root/'data/validation.npy',validation)
    from synthetic.data import fm_batch
    probe_x,probe_t,_=fm_batch(law.sample(rng,2048),rng)
    np.savez_compressed(root/'banks/probe.npz',x=probe_x[:256],t=probe_t[:256])
    x,t,pair=fm_batch(validation,np.random.default_rng(7761))
    np.savez_compressed(root/'banks/field.npz',x=x,t=t,pair=pair,oracle=law.oracle(x,t).astype(np.float32))
    gen=np.random.default_rng(14427);source=gen.standard_normal((2048,16)).astype(np.float32)
    directions=gen.standard_normal((128,16)).astype(np.float32)
    directions/=np.linalg.norm(directions,axis=1,keepdims=True).clip(min=1e-12)
    np.savez_compressed(root/'banks/sampling.npz',source=source,target=validation[:2048],target2=validation[2048:4096],directions=directions)
    gen=np.random.default_rng(4427)
    for clock in TIMES:
        tt=np.full((512,1),clock,dtype=np.float32)
        xx=tt*validation[:512]+(1-tt)*gen.standard_normal((512,16)).astype(np.float32)
        velocity,jacobian,weight=law.oracle(xx,tt,True)
        np.savez_compressed(root/'banks'/f'panel_{clock:.1f}.npz',x=xx,t=tt,oracle=velocity,oracle_jacobian=jacobian,posterior_code_weight=weight)
    write(root/'data-manifest.json',dict(specification=law.specification(),train_count=100000,validation_count=20000,
        data_initialization=0,model_seed=0,field_count=len(x),training_stream=100000,evaluation_streams=[7761,14427,4427]))
    # Bind new inputs, native libraries, cold states, and all explicitly reused outcomes.
    paths=list((root/'source').iterdir())+list((root/'data').iterdir())+list((root/'banks').iterdir())+[root/'settings.json',root/'data-manifest.json']
    paths+=list(Path(settings['native_root']).rglob('*.py'))+list(Path(settings['joint_source']).glob('*.py'))
    for family in FAMILIES:
        folder=joint/'output/runs'/family
        paths+=list(folder.rglob('*'))
    references=ref.reference_rows(root)
    for entry in references:paths+=list(Path(entry['folder']).rglob('*'))
    paths+=[repo/'docs/autoresearch/program.md',repo/'docs/autoresearch/cycle01-complete-field/results/decision.json']
    paths+=[Path(settings['old_cycle_runtime'])/'sealed-runtime.json']
    paths=sorted({p for p in paths if p.is_file()})
    write(root/'bindings.json',{str(p):ref.sha(p) for p in paths})
    print(dict(prepared=str(root),bindings=len(paths)),flush=True)


def verify(root,ref):
    for path,digest in load(root/'bindings.json').items():assert ref.sha(path)==digest,path
    previous=Path(load(root/'settings.json')['old_cycle_runtime'])
    for path,digest in load(previous/'sealed-runtime.json').items():assert ref.sha(previous/path)==digest,path


def preflight(root):
    settings,ref=setup(root);verify(root,ref);law=target(root)
    # Independent enumeration: all65536 means with explicit categorical probabilities.
    indices=np.arange(2**16,dtype=np.uint32)
    signs=2*((indices[:,None]>>np.arange(16))&1).astype(np.float64)-1
    means=signs*law.locations
    code={tuple(s) for s in law.signs}
    membership=np.array([tuple(s) in code for s in signs]);assert membership.sum()==32
    prior=np.full(len(means),.5/len(means));prior[membership]+=.5/32
    assert np.all(prior>0);np.testing.assert_allclose(prior.sum(),1,atol=1e-15)
    xx=torch.tensor(np.sin(np.arange(11*16).reshape(11,16))*.7,dtype=torch.float64,requires_grad=True)
    tt=torch.tensor(np.linspace(.05,.95,11)[:,None]);rotation=torch.tensor(law.rotation);m=torch.tensor(means)
    q=xx@rotation;d=(1-tt)**2+.04*tt**2
    logits=(tt/d*q)@m.T+torch.tensor(np.log(prior))[None]
    probability=torch.softmax(logits,dim=-1);average=probability@m
    native=(((.04*tt-(1-tt))/d)*q+(1-tt)/d*average)@rotation.T
    jj=torch.stack([torch.autograd.grad(native[:,i].sum(),xx,retain_graph=True)[0] for i in range(16)],1)
    expected,jac,_=law.oracle(xx.detach().numpy(),tt.numpy(),True)
    np.testing.assert_allclose(native.detach().numpy(),expected,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(jj.detach().numpy(),jac,rtol=1e-9,atol=1e-10)
    # The exact common origin covariance and one coupled four-sign moment.
    np.testing.assert_allclose(prior@signs,np.zeros(16),atol=1e-14)
    np.testing.assert_allclose((signs.T*prior)@signs,np.eye(16),atol=1e-12)
    fourth=float(prior@np.prod(signs[:,:4],axis=1));np.testing.assert_allclose(fourth,.5,atol=1e-12)
    bank=torch.from_numpy(np.load(root/'data/train.npy'));checks=[]
    for family in FAMILIES:
        model=ref.build(family);old=torch.load(Path(settings['old_joint_root'])/'output/runs'/family/'step_0000000.pt',weights_only=False)
        ref.equal(model.state_dict(),old['model']);ref.equal(dict(model.named_buffers()),old['runtime_buffers'])
        ref.equal(ref.optimizer(model).state_dict(),old['optimizer'])
        endpoints=[]
        for resume in (False,True):
            model=ref.build(family);opt=ref.optimizer(model);rng=torch.Generator().manual_seed(100000);digest=ref.EMPTY
            for step in range(1,4):
                x,t,a,b=ref.draw(bank,rng);digest=ref.legacy.fingerprint_batch(digest,a,b,t);ref.update(model,opt,x,t,a,b)
                if resume and step==2:
                    checkpoint=ref.checkpoint(model,opt,rng,step,digest);model=ref.build(family);opt=ref.optimizer(model)
                    digest=ref.restore(model,opt,rng,checkpoint)
            endpoints.append(ref.checkpoint(model,opt,rng,3,digest))
        ref.equal(*endpoints);checks.append(dict(family=family,identical_cold_state=True,exact_resume=True,costs=ref.compute_budget(model)))
    verify(root,ref)
    write(root/'preflight.json',dict(status='PASS',enumerated_components=65536,positive_probabilities=True,
        velocity_max_error=float(np.max(np.abs(native.detach().numpy()-expected))),
        jacobian_max_error=float(np.max(np.abs(jj.detach().numpy()-jac))),four_sign_moment=fourth,checks=checks))
    print(dict(status='PASS',components=65536,cold_and_resume=len(checks)),flush=True)


def matrices(model):
    rows=[]
    for name,p in model.named_parameters():
        if name.endswith('.proj.weight') or name.endswith(('.matrix1','.matrix2')):
            p=p.detach().double()
            rows.append(dict(layer=name,kind='original_P' if name.endswith('.proj.weight') else 'ordinary_ISRU_map',
                singular_values=torch.linalg.svdvals(p).tolist(),eigenvalues_real=torch.linalg.eigvals(p).real.tolist(),
                eigenvalues_imag=torch.linalg.eigvals(p).imag.tolist()))
    return rows


def cell(root,family):
    _,ref=setup(root);folder=root/'output/runs'/family;folder.mkdir(exist_ok=False);started=time.monotonic()
    def status(state='RUNNING',step=0,**extra):write(folder/'status.json',dict(state=state,step=step,pid=os.getpid(),elapsed_seconds=time.monotonic()-started,**extra))
    status()
    try:
        model=ref.build(family);opt=ref.optimizer(model);rng=torch.Generator().manual_seed(100000);digest=ref.EMPTY
        bank=torch.from_numpy(np.load(root/'data/train.npy'));banks=root/'banks'
        write(folder/'provenance.json',dict(family=family,model_seed=0,training_stream=100000,steps=5000,batch=512,costs=ref.compute_budget(model)))
        def save(step):
            torch.save(ref.checkpoint(model,opt,rng,step,digest),folder/f'step_{step:07d}.pt')
            write(folder/f'matrices_{step:07d}.json',matrices(model));write(folder/f'visits_{step:07d}.json',ref.legacy.visit_diagnostics(model,banks))
            return ref.legacy.field_evaluation(model,banks,save=folder/f'field_{step:07d}.npz')
        append(folder/'metrics.jsonl',dict(step=0,**save(0)))
        loss_sum=grad_sum=0.;count=clipped=0
        for step in range(1,5001):
            x,t,a,b=ref.draw(bank,rng);digest=ref.legacy.fingerprint_batch(digest,a,b,t)
            loss,gradient=ref.update(model,opt,x,t,a,b);loss_sum+=loss;grad_sum+=gradient;count+=1;clipped+=int(gradient>1)
            if step%100==0:
                status(step=step);assert time.monotonic()-started<600,'registered cell timeout'
            if step%500==0:
                row=dict(step=step,train_loss=loss_sum/count,grad_norm=grad_sum/count,clip_fraction=clipped/count)
                if step in SNAPSHOTS:row.update(save(step))
                append(folder/'metrics.jsonl',row);loss_sum=grad_sum=0.;count=clipped=0
        result=ref.legacy.diagnostics(model,banks,folder/'diagnostics',rollout=False)
        write(folder/'result.json',dict(family=family,law='full_component_support_half_mixture',input_stream_digest=digest,**ref.compute_budget(model),**result))
        status('COMPLETE',5000)
    except BaseException:status('FAILED',error=traceback.format_exc());raise


def run(root):
    _,ref=setup(root);verify(root,ref);assert load(root/'preflight.json')['status']=='PASS'
    output=root/'output';output.mkdir(exist_ok=False);(output/'runs').mkdir()
    stop=threading.Event();started=time.monotonic();status=dict(state='RUNNING',completed=0,total=5,pid=os.getpid())
    def progress(**extra):status.update(extra,elapsed_seconds=time.monotonic()-started);write(output/'status.json',status);print(status,flush=True)
    def monitor():
        import psutil
        while not stop.is_set():
            parent=psutil.Process();rows=[]
            for process in [parent]+parent.children(recursive=True):
                try:rows.append(dict(pid=process.pid,rss_bytes=process.memory_info().rss,cpu_seconds=sum(process.cpu_times()[:2])))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            append(output/'monitor.jsonl',dict(elapsed=time.monotonic()-started,processes=rows));stop.wait(5)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();progress()
    environment=os.environ.copy();environment.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    def launch(family):
        command=[sys.executable,str(Path(__file__)),'cell','--root',str(root),'--family',family]
        with (output/(family+'.log')).open('w') as log:
            process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
            append(output/'launches.jsonl',dict(family=family,pid=process.pid,command=command))
            try:code=process.wait(timeout=600)
            except subprocess.TimeoutExpired:process.terminate();process.wait();raise
        assert code==0,(family,code)
        return family
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(launch,family) for family in FAMILIES]
            for future in as_completed(futures):progress(completed=status['completed']+1,latest=future.result())
        verify(root,ref);progress(state='COMPLETE')
    except BaseException:progress(state='FAILED',error=traceback.format_exc());raise
    finally:stop.set();watcher.join(timeout=6)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','preflight','run','cell'));p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo',type=Path);p.add_argument('--base',type=Path);p.add_argument('--family',choices=FAMILIES);a=p.parse_args()
    if a.action=='prepare':prepare(a.root,a.repo,a.base)
    elif a.action=='preflight':preflight(a.root)
    elif a.action=='run':run(a.root)
    else:cell(a.root,a.family)
