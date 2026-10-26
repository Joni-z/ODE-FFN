"""Fifteen new paired-core fits, three verified historical controls."""
import argparse
import copy
import os
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

import numpy as np
import torch

BASE=Path('/Users/mr.z/ode-workspace-prep/architecture-search')
PARENT=BASE/'local_ffn_teacher_20261026'
sys.path.insert(0,str(PARENT/'source'))
import local_study as old
from teacher_support import TIMES,SNAPSHOTS,load,write,append,sha,signature,teacher,assert_equal
from map_core import MapCore,effective,direct_copy,sequential,input_jacobian

TEACHERS=('flow_map','swiglu128')
FAMILIES=('p_map','w_map','swiglu128')
SPLITS=('train','validation',*[f'panel-{t:.1f}' for t in TIMES])


def cold(family: str,position: int) -> torch.nn.Module:
    if family=='swiglu128':return old.cold(family,position)
    reference=old.cold('ode_k2',position)
    p=MapCore(reference.ode1.proj.weight,reference.ode2.proj.weight,'p_map')
    return p if family=='p_map' else direct_copy(p)


def target(name: str,position: int) -> torch.nn.Module:
    if name=='swiglu128':return teacher(name)[0].raw.blocks[position].ffn
    reference=teacher('ode_k2')[0].raw.blocks[position].ffn
    module=MapCore(reference.ode1.proj.weight,reference.ode2.proj.weight,'p_map')
    for p in module.parameters():p.requires_grad_(False)
    return module.eval()


def cache_arrays(cache: Path,split: str,position: int,name: str) -> tuple[np.ndarray,np.ndarray]:
    count=load(cache/'metadata.json')['counts'][split]
    z=np.load(PARENT/'output/cache'/old.file_name(split,position),mmap_mode='r')[:count]
    path=cache/f'{split}-flow-y{position}.npy' if name=='flow_map' else PARENT/'output/cache'/old.file_name(split,position,'swiglu128')
    y=np.load(path,mmap_mode='r')[:count]
    assert len(z)==len(y)==count
    return z,y


def jacobian_target(cache: Path,position: int,name: str) -> np.ndarray:
    path=cache/f'jac-flow-{position}.npy' if name=='flow_map' else PARENT/'output/cache'/f'jac-swiglu128-{position}.npy'
    return np.load(path)


def prepare(cache: Path,small: bool=False) -> None:
    cache.mkdir(exist_ok=False)
    counts={split:(2048 if split in ('train','validation') else 128) if small else
            (100000 if split=='train' else 20000 if split=='validation' else 512) for split in SPLITS}
    original=load(PARENT/'output/cache/metadata.json')
    # Every shared array was sealed before the current experiment.
    references={}
    for split in SPLITS:
        for j in range(3):
            for family in (None,'swiglu128'):
                name=old.file_name(split,j,family)
                path=PARENT/'output/cache'/name
                assert sha(path)==original['files_sha256'][name],name
                references[str(path)]=sha(path)
    for j in range(3):
        path=PARENT/'output/cache'/f'jac-swiglu128-{j}.npy'
        references[str(path)]=sha(path)
    signatures={}
    for j in range(3):
        module=target('flow_map',j);before=signature(module)
        for split in SPLITS:
            z=np.load(PARENT/'output/cache'/old.file_name(split,j),mmap_mode='r')[:counts[split]]
            np.save(cache/f'{split}-flow-y{j}.npy',old.predict(module,z))
            if split=='validation':
                _,jac=old.native_jacobian(module,torch.from_numpy(np.array(z[:128],copy=True)))
                np.save(cache/f'jac-flow-{j}.npy',jac.numpy())
        assert signature(module)==before;signatures[str(j)]=before
    write(cache/'metadata.json',dict(counts=counts,flow_target_signatures=signatures,external_arrays=references,
          files_sha256={p.name:sha(p) for p in cache.glob('*.npy')}))


def metrics(module: torch.nn.Module,cache: Path,name: str,position: int,path: Path|None=None) -> dict:
    z,y=cache_arrays(cache,'validation',position,name)
    value=old.predict(module,z)
    if path is not None:np.save(path,value)
    return old.error(value,y)


def compute_budget(module: torch.nn.Module) -> dict:
    from synthetic.conservative_readout import MatrixCounter
    result=dict(parameters=sum(p.numel() for p in module.parameters()))
    for n in (1,512):
        x=torch.linspace(-.4,.7,n*64).reshape(n,64)
        with MatrixCounter() as inference:
            with torch.no_grad():module(x)
        with MatrixCounter() as training:
            value=module(x);torch.autograd.grad(value.square().mean(),tuple(module.parameters()))
        result[f'batch{n}_forward_macs']=int(inference.macs)
        result[f'batch{n}_forward_backward_macs']=int(training.macs)
    if isinstance(module,MapCore):
        folded=direct_copy(module)
        with MatrixCounter() as count:
            with torch.no_grad():folded(torch.zeros(1,64))
        result['folded_inference_macs_per_input']=int(count.macs)
    result['scope']='actual matrix MACs; Q formation included; pointwise work excluded'
    return result


def matrix_stats(module: torch.nn.Module) -> dict:
    entries={}
    if isinstance(module,MapCore):
        with torch.no_grad():
            w1,w2=module.effective_matrices()
        matrices={'effective1':w1,'effective2':w2}
        if module.coordinates=='p_map':matrices.update(P1=module.matrix1,P2=module.matrix2)
        origin=.5*w2.detach().double()@w1.detach().double()
    else:
        matrices={'w12':module.w12.weight,'w3':module.w3.weight}
        gate,value=module.w12.bias.detach().double().chunk(2)
        wg,wv=module.w12.weight.detach().double().chunk(2,0)
        s=torch.sigmoid(gate);a=gate*s;ap=s+gate*s*(1-s)
        origin=module.w3.weight.detach().double()@((value*ap)[:,None]*wg+a[:,None]*wv)
    for name,value in matrices.items():
        value=value.detach().double();singular=torch.linalg.svdvals(value)
        entry=dict(shape=list(value.shape),singular_values=singular.tolist())
        if value.shape[0]==value.shape[1]:
            sign,logdet=torch.linalg.slogdet(value)
            entry.update(determinant_sign=float(sign),logabsdet=float(logdet),
                         symmetric_minimum=float(torch.linalg.eigvalsh((value+value.T)/2).min()))
            if name.startswith('P'):
                ev=torch.linalg.eigvals(value);mods=(1+.5*(ev[:,None]+ev[None,:])).abs()
                entry.update(differential_minimum_eigenmodulus=float(mods.min()),
                             differential_condition_lower_bound=float(mods.max()/mods.min()))
                actual=matrices['effective'+name[-1]].detach().double()
                algebraic=effective(value)
                entry['effective_roundoff_spectral_norm']=float(torch.linalg.matrix_norm(actual-algebraic,ord=2))
        entries[name]=entry
    sign,logdet=torch.linalg.slogdet(origin)
    singular=torch.linalg.svdvals(origin)
    return dict(matrices=entries,origin_jacobian=dict(determinant_sign=float(sign),logabsdet=float(logdet),
                 minimum_singular_value=float(singular[-1])))


def reused_folder(position: int) -> Path:
    return PARENT/'output/runs'/f'swiglu128--swiglu128--{position}'


def record_reuse(root: Path) -> list[dict]:
    rows=[];folder=root/'output/reused';folder.mkdir(exist_ok=False)
    for j in range(3):
        original=reused_folder(j);provenance=load(original/'provenance.json')
        module=cold('swiglu128',j)
        assert signature(module)==provenance['initial_signature']
        state=torch.load(original/'step_0000000.pt',map_location='cpu',weights_only=False)
        assert_equal(module.state_dict(),state['model'])
        assert_equal(old.optimizer(module).state_dict(),state['optimizer'])
        assert load(original/'status.json')['state']=='COMPLETE'
        assert provenance['seed']==0 and provenance['position']==j
        z,y=cache_arrays(root/'output/cache','train',j,'swiglu128')
        assert len(z)==100000 and len(y)==100000
        paths=[p for p in original.iterdir() if p.is_file()]
        row=dict(teacher='swiglu128',student='swiglu128',position=j,origin='historical verified reuse',
                 folder=str(original),files_sha256={p.name:sha(p) for p in paths},
                 old_provenance_sha256=sha(original/'provenance.json'),
                 data_definition='same sealed Z/target arrays and 100000-bound index generator seed 0')
        write(folder/f'position-{j}.json',row);rows.append(row)
    return rows


def preflight(root: Path) -> None:
    torch.set_num_threads(1)
    prepare(root/'preflight-cache',small=True)
    cache=root/'preflight-cache';pairs=[];gradients=[]
    for j in range(3):
        p,w=cold('p_map',j),cold('w_map',j)
        z,_=cache_arrays(cache,'validation',j,'flow_map');x=torch.from_numpy(np.array(z[:13],copy=True))
        pv,pj=old.native_jacobian(p,x);wv,wj=old.native_jacobian(w,x)
        assert torch.equal(pv,wv) and torch.equal(pj,wj)
        for kind,subject in [('cold',p),('teacher',target('flow_map',j))]:
            pp=copy.deepcopy(subject).double()
            for parameter in pp.parameters():parameter.requires_grad_(True)
            ww=direct_copy(pp);xx=x.double()
            actual=pp(xx);reference=sequential(pp,xx)
            torch.testing.assert_close(actual,reference,rtol=1e-10,atol=1e-11)
            native_j=old.native_jacobian(pp,xx)[1]
            torch.testing.assert_close(native_j,input_jacobian(pp,xx),rtol=1e-10,atol=1e-11)
            y=torch.sin(torch.arange(xx.numel(),dtype=xx.dtype)).reshape_as(xx)
            gp=torch.autograd.grad((pp(xx)-y).square().mean(),tuple(pp.parameters()))
            gw=torch.autograd.grad((ww(xx)-y).square().mean(),tuple(ww.parameters()))
            differences=[]
            for matrix,g,h in zip((pp.matrix1,pp.matrix2),gp,gw):
                expected=h+.5*(matrix.T@h+h@matrix.T)
                torch.testing.assert_close(g,expected,rtol=1e-10,atol=1e-11)
                differences.append(float((g-expected).detach().abs().max()))
            gradients.append(dict(position=j,kind=kind,max_abs=max(differences)))
        for family in ('p_map','w_map'):
            exact=target('flow_map',j)
            if family=='w_map':exact=direct_copy(exact)
            zz,yy=cache_arrays(cache,'validation',j,'flow_map')
            assert np.array_equal(old.predict(exact,zz),yy)
        tt=target('swiglu128',j);zz,yy=cache_arrays(cache,'validation',j,'swiglu128')
        assert np.array_equal(old.predict(tt,zz),yy)
        trained_pair=[]
        for name in TEACHERS:
            for family in FAMILIES:
                if name=='swiglu128' and family=='swiglu128':continue
                histories=[]
                for resume in (False,True):
                    module=cold(family,j);opt=old.optimizer(module);rng=torch.Generator().manual_seed(0)
                    digests=dict.fromkeys(('indices','input','label'),old.EMPTY)
                    z,y=cache_arrays(cache,'train',j,name)
                    for step in range(1,4):
                        xx,yy=old.consumed(z,y,rng,digests);old.train_step(module,opt,xx,yy)
                        if resume and step==2:
                            state=old.checkpoint(module,opt,rng,step,digests)
                            module=cold(family,j);opt=old.optimizer(module)
                            digests=old.restore(module,opt,rng,state)
                    histories.append(old.checkpoint(module,opt,rng,3,digests))
                assert_equal(*histories)
                if name=='flow_map' and family in ('p_map','w_map'):trained_pair.append(module)
        assert max(float((a-b).detach().abs().max()) for a,b in zip(trained_pair[0].effective_matrices(),trained_pair[1].effective_matrices()))>0
        # Verify reusable controls on their original full-bound data stream.
        original=reused_folder(j);saved=torch.load(original/'step_0000000.pt',map_location='cpu',weights_only=False)
        histories=[]
        for restored in (False,True):
            module=cold('swiglu128',j)
            if restored:module.load_state_dict(saved['model'])
            opt=old.optimizer(module);assert_equal(opt.state_dict(),saved['optimizer'])
            rng=torch.Generator().manual_seed(0);digests=dict.fromkeys(('indices','input','label'),old.EMPTY)
            z=np.load(PARENT/'output/cache'/old.file_name('train',j),mmap_mode='r')
            y=np.load(PARENT/'output/cache'/old.file_name('train',j,'swiglu128'),mmap_mode='r')
            for step in range(3):xx,yy=old.consumed(z,y,rng,digests);old.train_step(module,opt,xx,yy)
            histories.append(old.checkpoint(module,opt,rng,3,digests))
        assert_equal(*histories)
        pairs.append(dict(position=j,bitwise_initial_values=True,bitwise_initial_input_jacobians=True))
    # Concrete determinant fixtures check the scope of the algebraic statement.
    block=torch.tensor([[-1.,-2.],[2.,-1.]],dtype=torch.float64)
    torch.testing.assert_close(effective(block),-1.5*torch.eye(2,dtype=torch.float64))
    assert float(torch.linalg.det(effective(block)))>0
    direct=MapCore(torch.diag(torch.tensor([-1.]+[1.]*63,dtype=torch.float64)),torch.eye(64,dtype=torch.float64),'w_map')
    assert float(torch.linalg.slogdet(input_jacobian(direct,torch.zeros(1,64,dtype=torch.float64))[0])[0])==-1.
    write(root/'preflight.json',dict(status='PASS',pairs=pairs,adjoint_checks=gradients,exact_flow_representers=6,
          exact_swiglu_representers=3,exact_resume_cells=15,historical_three_step_replays=3,
          costs={family:compute_budget(cold(family,0)) for family in FAMILIES}))


def cell(root: Path,name: str,family: str,position: int) -> None:
    torch.set_num_threads(1);folder=root/'output/runs'/f'{name}--{family}--{position}';folder.mkdir(exist_ok=False)
    started=time.monotonic()
    def status(state='RUNNING',step=0,**extra):write(folder/'status.json',dict(state=state,step=step,pid=os.getpid(),elapsed_seconds=time.monotonic()-started,**extra))
    status()
    try:
        module=cold(family,position);opt=old.optimizer(module);rng=torch.Generator().manual_seed(0)
        initial=signature(module);digests=dict.fromkeys(('indices','input','label'),old.EMPTY);cache=root/'output/cache'
        z,y=cache_arrays(cache,'train',position,name)
        write(folder/'provenance.json',dict(teacher=name,student=family,position=position,seed=0,steps=5000,batch_size=512,
              origin='new cold training',initial_signature=initial,device='cpu',threads=torch.get_num_threads(),
              source_manifest_sha256=sha(Path(__file__).parent/'source-sha256.json'),costs=compute_budget(module)))
        def save(step):
            torch.save(old.checkpoint(module,opt,rng,step,digests),folder/f'step_{step:07d}.pt')
            write(folder/f'matrices_{step:07d}.json',matrix_stats(module))
            return metrics(module,cache,name,position,folder/f'validation_{step:07d}.npy')
        append(folder/'metrics.jsonl',dict(step=0,**save(0)))
        loss_sum=grad_sum=0.;count=clipped=0
        for step in range(1,5001):
            xx,yy=old.consumed(z,y,rng,digests);loss,grad=old.train_step(module,opt,xx,yy)
            loss_sum+=loss;grad_sum+=grad;count+=1;clipped+=int(grad>1)
            if step%100==0:status(step=step)
            if step%500==0:
                record=dict(step=step,training_loss=loss_sum/count,mean_gradient_norm=grad_sum/count,clip_fraction=clipped/count)
                if step in SNAPSHOTS:record.update(save(step))
                append(folder/'metrics.jsonl',record);print(record,flush=True)
                loss_sum=grad_sum=0.;count=clipped=0
        panels=[]
        for t in TIMES:
            zz,yy=cache_arrays(cache,f'panel-{t:.1f}',position,name);value=old.predict(module,zz)
            np.save(folder/f'panel_{t:.1f}.npy',value);panels.append(dict(time=t,**old.error(value,yy)))
        zv,_=cache_arrays(cache,'validation',position,name)
        _,jac=old.native_jacobian(module,torch.from_numpy(np.array(zv[:128],copy=True)));np.save(folder/'jacobian.npy',jac.numpy())
        truth=jacobian_target(cache,position,name).astype(np.float64)
        je=float(np.mean(np.sum((jac.numpy().astype(np.float64)-truth)**2,axis=(1,2))))
        result=dict(teacher=name,student=family,position=position,origin='new cold training',step=5000,
                    **metrics(module,cache,name,position),panels=panels,jacobian_error=je,
                    relative_jacobian_error=je/float(np.mean(np.sum(truth**2,axis=(1,2)))),
                    consumed_digests=digests,final_signature=signature(module),elapsed_seconds=time.monotonic()-started)
        assert result['final_signature']!=initial;write(folder/'result.json',result);status(state='COMPLETE',step=5000)
    except BaseException:status(state='FAILED',error=traceback.format_exc());raise


def run(root: Path) -> None:
    torch.set_num_threads(1);ready=load(root/'ready.json');assert load(root/'preflight.json')['status']=='PASS'
    for path,digest in ready['bindings'].items():assert sha(path)==digest,path
    out=root/'output';out.mkdir(exist_ok=False);started=time.monotonic()
    stop,abort,lock=threading.Event(),threading.Event(),threading.Lock();children=set()
    progress=dict(state='RUNNING',phase='prepare',completed=0,total_new=15,reused=0,pid=os.getpid())
    def status(**extra):progress.update(extra,elapsed_seconds=time.monotonic()-started);write(out/'status.json',progress);print(progress,flush=True)
    def monitor():
        while not stop.is_set():
            row=old.observe();append(out/'monitor.jsonl',row);write(out/'monitor.json',row);stop.wait(5)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();status()
    try:
        prepare(out/'cache');references=record_reuse(root);status(phase='training',reused=len(references));(out/'runs').mkdir()
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
        def launch(job):
            name,family,j=job;label=f'{name}--{family}--{j}'
            command=[sys.executable,str(Path(__file__)),'cell','--root',str(root),'--teacher',name,'--student',family,'--position',str(j)]
            with (out/(label+'.log')).open('w') as log:
                with lock:
                    if abort.is_set():raise RuntimeError('study aborted')
                    p=subprocess.Popen(command,cwd=Path(__file__).parent,env=env,stdout=log,stderr=subprocess.STDOUT);children.add(p)
                    append(out/'launches.jsonl',dict(name=label,pid=p.pid,command=command))
                code=p.wait()
                with lock:children.remove(p)
            if code:raise RuntimeError(f'{label} exited {code}')
            return label
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(launch,(t,s,j)) for t in TEACHERS for j in range(3) for s in FAMILIES if not (t=='swiglu128' and s=='swiglu128')]
            try:
                for future in as_completed(futures):status(completed=progress['completed']+1,latest=future.result())
            except BaseException:
                abort.set()
                for future in futures:future.cancel()
                with lock:
                    for p in children:p.terminate()
                raise
        for path,digest in ready['bindings'].items():assert sha(path)==digest,path
        status(state='COMPLETE',phase='complete')
    except BaseException:status(state='FAILED',error=traceback.format_exc());raise
    finally:stop.set();watcher.join(timeout=6)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preflight','run','cell']);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--teacher',choices=TEACHERS);parser.add_argument('--student',choices=FAMILIES);parser.add_argument('--position',type=int,choices=range(3));args=parser.parse_args()
    if args.action=='preflight':preflight(args.root)
    elif args.action=='run':run(args.root)
    else:cell(args.root,args.teacher,args.student,args.position)
