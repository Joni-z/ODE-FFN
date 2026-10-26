"""Twelve new placement fits with twelve explicitly historical controls."""
import argparse
import copy
import os
import subprocess
import sys
import threading
import time as timer
import traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import torch

BASE=Path('/Users/mr.z/ode-workspace-prep/architecture-search')
MAP_ROOT=BASE/'effective_map_20261026';DATA=MAP_ROOT/'output/cache'
sys.path.insert(0,str(MAP_ROOT/'source'))
import map_study as maps
from map_core import direct_copy
from teacher_support import TIMES,SNAPSHOTS,load,write,append,sha,signature,assert_equal
from bias_core import BiasCore,input_jacobian

NEW=('hidden_bias','output_bias');REFERENCES=('w_map','swiglu128')


def cold(family: str,position: int) -> torch.nn.Module:
    base=maps.cold('w_map' if family in NEW else family,position)
    if family not in NEW:return base
    return BiasCore(base.matrix1,base.matrix2,family)


def reference_folder(name: str,family: str,position: int) -> Path:
    rows=load(MAP_ROOT/'audit-cohort.json')
    return Path(next(r['folder'] for r in rows if r['teacher']==name and r['student']==family and r['position']==position))


def budget(module: torch.nn.Module) -> dict:
    from synthetic.conservative_readout import MatrixCounter
    counts={}
    for n in (1,512):
        x=torch.linspace(-.4,.7,n*64).reshape(n,64)
        with MatrixCounter() as inference:
            with torch.no_grad():module(x)
        with MatrixCounter() as training:
            y=module(x);torch.autograd.grad(y.square().mean(),tuple(module.parameters()))
        counts[f'batch{n}_forward_macs']=int(inference.macs)
        counts[f'batch{n}_forward_backward_macs']=int(training.macs)
    return dict(parameters=sum(p.numel() for p in module.parameters()),**counts,
                scope='actual dense matrix MACs; offset/activation pointwise work excluded')


def statistics(module: BiasCore) -> dict:
    offset=module.offset.detach().double()
    values={}
    for name,p in [('matrix1',module.matrix1),('matrix2',module.matrix2)]:
        values[name]=dict(singular_values=torch.linalg.svdvals(p.detach().double()).tolist())
    with torch.no_grad():jac=input_jacobian(copy.deepcopy(module).double(),torch.zeros(1,64,dtype=torch.float64))[0]
    sign,logdet=torch.linalg.slogdet(jac)
    return dict(offset=offset.tolist(),offset_norm=float(offset.norm()),offset_mean=float(offset.mean()),
                offset_minimum=float(offset.min()),offset_maximum=float(offset.max()),matrices=values,
                origin_jacobian_sign=float(sign),origin_jacobian_logabsdet=float(logdet))


def validation(module: torch.nn.Module,name: str,position: int,path: Path|None=None) -> dict:
    return maps.metrics(module,DATA,name,position,path)


def preflight(root: Path) -> None:
    torch.set_num_threads(1);pairing=[];derivatives=[];references=[]
    for j in range(3):
        z,_=maps.cache_arrays(DATA,'validation',j,'flow_map');x=torch.from_numpy(np.array(z[:13],copy=True))
        base=cold('w_map',j);base_value,base_jac=maps.old.native_jacobian(base,x)
        for family in NEW:
            module=cold(family,j);value,jac=maps.old.native_jacobian(module,x)
            assert torch.equal(value,base_value) and torch.equal(jac,base_jac)
            assert torch.equal(module.matrix1,base.matrix1) and torch.equal(module.matrix2,base.matrix2)
            pairing.append(dict(position=j,placement=family,bitwise_values=True,bitwise_input_jacobians=True))
            double=copy.deepcopy(module).double()
            with torch.no_grad():double.offset.copy_(torch.linspace(-.6,.7,64,dtype=torch.float64))
            xx=x.double();value,native=maps.old.native_jacobian(double,xx)
            expected=input_jacobian(double,xx);torch.testing.assert_close(native,expected,rtol=1e-10,atol=1e-11)
            y=torch.cos(torch.arange(xx.numel(),dtype=torch.float64)).reshape_as(xx)
            loss=(double(xx)-y).square().mean();actual=torch.autograd.grad(loss,double.offset)[0]
            upstream=2*(double(xx)-y)/xx.numel()
            if family=='output_bias':manual=upstream.sum(0)
            else:
                pre=xx@double.matrix1.T+double.offset;s=torch.sigmoid(pre)
                manual=((upstream@double.matrix2)*(s+pre*s*(1-s))).sum(0)
            torch.testing.assert_close(actual,manual,rtol=1e-10,atol=1e-11)
            derivatives.append(dict(position=j,placement=family,jacobian_max_abs=float((native-expected).detach().abs().max()),
                                    offset_gradient_max_abs=float((actual-manual).detach().abs().max())))
            exact=direct_copy(maps.target('flow_map',j));representer=BiasCore(exact.matrix1,exact.matrix2,family)
            zz,yy=maps.cache_arrays(DATA,'validation',j,'flow_map')
            assert np.array_equal(maps.old.predict(representer,zz),yy)
        for name in maps.TEACHERS:
            for family in NEW:
                history=[]
                for resume in (False,True):
                    module=cold(family,j);opt=maps.old.optimizer(module);rng=torch.Generator().manual_seed(0)
                    digests=dict.fromkeys(('indices','input','label'),maps.old.EMPTY)
                    zz,yy=maps.cache_arrays(DATA,'train',j,name)
                    for step in range(1,4):
                        xx2,yy2=maps.old.consumed(zz,yy,rng,digests);maps.old.train_step(module,opt,xx2,yy2)
                        if resume and step==2:
                            state=maps.old.checkpoint(module,opt,rng,step,digests)
                            module=cold(family,j);opt=maps.old.optimizer(module);digests=maps.old.restore(module,opt,rng,state)
                    history.append(maps.old.checkpoint(module,opt,rng,3,digests))
                assert_equal(*history)
            for family in REFERENCES:
                folder=reference_folder(name,family,j);module=cold(family,j)
                state=torch.load(folder/'step_0000000.pt',map_location='cpu',weights_only=False)
                assert_equal(module.state_dict(),state['model']);assert_equal(maps.old.optimizer(module).state_dict(),state['optimizer'])
                assert load(folder/'status.json')['state']=='COMPLETE'
                references.append(dict(teacher=name,student=family,position=j,initial_signature=signature(module),folder=str(folder)))
    # Algebraic fixture at nonzero offsets: output offsets keep the linear odd part.
    base=cold('w_map',0).double();x=torch.linspace(-1,1,64,dtype=torch.float64).reshape(1,64)
    output=BiasCore(base.matrix1,base.matrix2,'output_bias').double()
    hidden=BiasCore(base.matrix1,base.matrix2,'hidden_bias').double()
    with torch.no_grad():output.offset.fill_(1.);hidden.offset.fill_(1.)
    linear=.5*(x@base.matrix1.T)@base.matrix2.T
    torch.testing.assert_close((output(x)-output(-x))/2,linear,rtol=1e-10,atol=1e-11)
    assert float((((hidden(x)-hidden(-x))/2)-linear).detach().abs().max())>1e-6
    write(root/'preflight.json',dict(status='PASS',paired_initial_checks=pairing,nonzero_offset_derivatives=derivatives,
          exact_flow_representers=6,exact_resume_cells=12,historical_references=references,
          costs={family:budget(cold(family,0)) for family in (*NEW,*REFERENCES)}))


def cell(root: Path,name: str,family: str,position: int) -> None:
    torch.set_num_threads(1);folder=root/'output/runs'/f'{name}--{family}--{position}';folder.mkdir(exist_ok=False)
    started=timer.monotonic()
    def status(state='RUNNING',step=0,**extra):write(folder/'status.json',dict(state=state,step=step,pid=os.getpid(),elapsed_seconds=timer.monotonic()-started,**extra))
    status()
    try:
        module=cold(family,position);opt=maps.old.optimizer(module);rng=torch.Generator().manual_seed(0)
        initial=signature(module);digests=dict.fromkeys(('indices','input','label'),maps.old.EMPTY)
        z,y=maps.cache_arrays(DATA,'train',position,name)
        write(folder/'provenance.json',dict(teacher=name,student=family,position=position,seed=0,steps=5000,batch_size=512,
              initial_signature=initial,origin='new cold placement training',device='cpu',threads=torch.get_num_threads(),costs=budget(module)))
        def save(step):
            torch.save(maps.old.checkpoint(module,opt,rng,step,digests),folder/f'step_{step:07d}.pt')
            write(folder/f'statistics_{step:07d}.json',statistics(module))
            return validation(module,name,position,folder/f'validation_{step:07d}.npy')
        append(folder/'metrics.jsonl',dict(step=0,**save(0)))
        loss_sum=grad_sum=0.;count=clipped=0
        for step in range(1,5001):
            xx,yy=maps.old.consumed(z,y,rng,digests);loss,grad=maps.old.train_step(module,opt,xx,yy)
            loss_sum+=loss;grad_sum+=grad;count+=1;clipped+=int(grad>1)
            if step%100==0:status(step=step)
            if step%500==0:
                row=dict(step=step,training_loss=loss_sum/count,mean_gradient_norm=grad_sum/count,clip_fraction=clipped/count)
                if step in SNAPSHOTS:row.update(save(step))
                append(folder/'metrics.jsonl',row);print(row,flush=True);loss_sum=grad_sum=0.;count=clipped=0
        panels=[]
        for clock in TIMES:
            zz,yy=maps.cache_arrays(DATA,f'panel-{clock:.1f}',position,name);value=maps.old.predict(module,zz)
            np.save(folder/f'panel_{clock:.1f}.npy',value);panels.append(dict(time=clock,**maps.old.error(value,yy)))
        zv,_=maps.cache_arrays(DATA,'validation',position,name)
        _,jac=maps.old.native_jacobian(module,torch.from_numpy(np.array(zv[:128],copy=True)));np.save(folder/'jacobian.npy',jac.numpy())
        truth=maps.jacobian_target(DATA,position,name).astype(np.float64)
        je=float(np.mean(np.sum((jac.numpy().astype(np.float64)-truth)**2,axis=(1,2))))
        result=dict(teacher=name,student=family,position=position,step=5000,origin='new cold placement training',
                    **validation(module,name,position),panels=panels,jacobian_error=je,
                    relative_jacobian_error=je/float(np.mean(np.sum(truth**2,axis=(1,2)))),
                    consumed_digests=digests,final_signature=signature(module),elapsed_seconds=timer.monotonic()-started)
        assert result['final_signature']!=initial;write(folder/'result.json',result);status(state='COMPLETE',step=5000)
    except BaseException:status(state='FAILED',error=traceback.format_exc());raise


def run(root: Path) -> None:
    torch.set_num_threads(1);ready=load(root/'ready.json');assert load(root/'preflight.json')['status']=='PASS'
    for path,digest in ready['bindings'].items():assert sha(path)==digest,path
    out=root/'output';out.mkdir(exist_ok=False);(out/'runs').mkdir();(out/'reused').mkdir();started=timer.monotonic()
    for row in load(root/'preflight.json')['historical_references']:
        folder=Path(row['folder']);entry={**row,'files_sha256':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}}
        write(out/'reused'/f"{row['teacher']}--{row['student']}--{row['position']}.json",entry)
    stop,abort,lock=threading.Event(),threading.Event(),threading.Lock();children=set()
    progress=dict(state='RUNNING',completed=0,total_new=12,reused=12,pid=os.getpid())
    def status(**extra):progress.update(extra,elapsed_seconds=timer.monotonic()-started);write(out/'status.json',progress);print(progress,flush=True)
    def monitor():
        while not stop.is_set():
            row=maps.old.observe();append(out/'monitor.jsonl',row);write(out/'monitor.json',row);stop.wait(5)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();status()
    try:
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        def launch(job):
            name,family,j=job;label=f'{name}--{family}--{j}'
            command=[sys.executable,str(Path(__file__)),'cell','--root',str(root),'--teacher',name,'--student',family,'--position',str(j)]
            with (out/(label+'.log')).open('w') as log:
                with lock:
                    if abort.is_set():raise RuntimeError('study aborted')
                    p=subprocess.Popen(command,cwd=Path(__file__).parent,env=env,stdout=log,stderr=subprocess.STDOUT);children.add(p);append(out/'launches.jsonl',dict(name=label,pid=p.pid,command=command))
                code=p.wait()
                with lock:children.remove(p)
            if code:raise RuntimeError(f'{label} exited {code}')
            return label
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(launch,(t,f,j)) for t in maps.TEACHERS for j in range(3) for f in NEW]
            try:
                for future in as_completed(futures):status(completed=progress['completed']+1,latest=future.result())
            except BaseException:
                abort.set()
                for future in futures:future.cancel()
                with lock:
                    for p in children:p.terminate()
                raise
        for path,digest in ready['bindings'].items():assert sha(path)==digest,path
        status(state='COMPLETE')
    except BaseException:status(state='FAILED',error=traceback.format_exc());raise
    finally:stop.set();watcher.join(timeout=6)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','run','cell']);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--teacher',choices=maps.TEACHERS);parser.add_argument('--student',choices=NEW);parser.add_argument('--position',type=int,choices=range(3));args=parser.parse_args()
    if args.action=='preflight':preflight(args.root)
    elif args.action=='run':run(args.root)
    else:cell(args.root,args.teacher,args.student,args.position)
