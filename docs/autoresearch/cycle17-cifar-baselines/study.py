"""Fixed CIFAR baseline screen: source-bound CPU training, checkpointing and Heun."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import psutil
import torch

ARMS=('ode_k1','ode_k2','swiglu86','swiglu129')
SNAPSHOTS=(0,1000,2500,5000)
EMPTY=hashlib.sha256(b'').hexdigest()
def load(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):
    p=Path(p);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');os.replace(temp,p)
def append(p,x):
    with Path(p).open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n')
def verify(root,require_metrics=False):
    for p,h in load(root/'bindings.json').items():assert sha(p)==h,p
    parent=Path(load(root/'settings.json')['parent'])
    for p,h in load(parent/'sealed-runtime.json').items():assert sha(parent/p)==h,p
    for p,h in load(parent/'closure-bindings.json').items():assert sha(p)==h,p
    if require_metrics:
        for p,h in load(root/'metric-bindings.json').items():assert sha(p)==h,p
        assert load(root/'metric-ready.json')['status']=='PASS'

def model_for(root,arm):
    sys.path.insert(0,str(root/'source'));from image_model import build
    return build(root/'source/jit',arm)
def optimizer(model):return torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=0.)
def data_for(root):
    data=Path(load(root/'settings.json')['data'])
    images=torch.from_numpy(np.array(np.load(data/'images.uint8.npy',mmap_mode='r')))
    with np.load(data/'split.npz') as f:indices=torch.from_numpy(f['train'])
    return images,indices

def draw(images,indices,rng):
    ids=indices[torch.randint(len(indices),(64,),generator=rng)]
    target=images[ids].float()/127.5-1
    noise=torch.randn(target.shape,generator=rng);t=torch.sigmoid(.8*torch.randn(64,generator=rng)-.8)
    x=t[:,None,None,None]*target+(1-t[:,None,None,None])*noise
    return dict(indices=ids,target=target,noise=noise,t=t,x=x)
def fingerprint(previous,batch):
    h=hashlib.sha256(bytes.fromhex(previous))
    for key in ('indices','noise','t'):h.update(batch[key].numpy().tobytes())
    return h.hexdigest()
def update(model,opt,batch):
    opt.zero_grad(set_to_none=True)
    output=model(batch['x'],batch['t'],torch.zeros(64,dtype=torch.long))
    loss=((output-batch['target'])/(1-batch['t'][:,None,None,None]).clamp_min(.05)).square().mean()
    assert torch.isfinite(loss);loss.backward();g=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    assert torch.isfinite(g);opt.step();return float(loss.detach()),float(g)
def checkpoint(model,opt,rng,step,digest):
    return copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),data_rng=rng.get_state(),
        torch_rng=torch.get_rng_state(),step=step,input_stream_digest=digest,
        runtime_buffers={n:b.detach().clone() for n,b in model.named_buffers()}))
def restore_model(model,state):
    model.load_state_dict(state['model']);buffers=dict(model.named_buffers())
    assert buffers.keys()==state['runtime_buffers'].keys()
    with torch.no_grad():
        for n,b in state['runtime_buffers'].items():buffers[n].copy_(b)
def restore(model,opt,rng,state):
    restore_model(model,state);opt.load_state_dict(state['optimizer']);rng.set_state(state['data_rng']);torch.set_rng_state(state['torch_rng'])
    return state['input_stream_digest']
def equal(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):assert a.keys()==b.keys();[equal(a[k],b[k]) for k in a]
    elif isinstance(a,(list,tuple)):assert len(a)==len(b);[equal(x,y) for x,y in zip(a,b)]
    else:assert a==b

def bank(path):
    with np.load(path) as f:return {k:f[k] for k in f.files}
def predict(model,b):
    with torch.no_grad():return np.concatenate([model(torch.from_numpy(b['x'][i:i+64]),torch.from_numpy(b['t'][i:i+64]),
        torch.zeros(min(64,len(b['x'])-i),dtype=torch.long)).numpy() for i in range(0,len(b['x']),64)])
def metrics(pred,b):
    error=pred.astype(np.float64)-b['target'].astype(np.float64)
    return dict(weighted_mse=float(np.mean((error/b['delta'].astype(np.float64)[:,None,None,None])**2)),
        clean_mse=float(np.mean(error**2)),prediction_rms=float(np.sqrt(np.mean(pred.astype(np.float64)**2))))

def diagnostics(model,b,folder,step):
    p_records=[]
    for j,block in enumerate(model.blocks):
        if not hasattr(block.mlp,'ode1'):continue
        for key in ('ode1','ode2'):
            layer=getattr(block.mlp,key);p=layer.proj.weight.detach().double().numpy();eig=np.linalg.eigvals(p)
            p_records.append(dict(layer=f'blocks.{j}.{key}',singular_values=np.linalg.svd(p,compute_uv=False).tolist(),
                eigenvalues_real=eig.real.tolist(),eigenvalues_imag=eig.imag.tolist(),frobenius=float(np.linalg.norm(p))))
    write(folder/f'P_{step:07d}.json',p_records);rows=[]
    for clock in (.1,.3,.5,.7,.9):
        panel=bank(b/f'panel_{clock:.1f}.npz');captured=[];hooks=[]
        for j,block in enumerate(model.blocks):
            if not hasattr(block.mlp,'ode1'):continue
            for key in ('ode1','ode2'):
                def hook(layer,inputs,name=f'blocks.{j}.{key}'):
                    captured.append((name,layer,inputs[0].detach().clone()))
                hooks.append(getattr(block.mlp,key).register_forward_pre_hook(hook))
        prediction=predict(model,panel)
        for h in hooks:h.remove()
        np.savez_compressed(folder/f'panel_{step:07d}_{clock:.1f}.npz',prediction=prediction)
        with torch.no_grad():
            for name,layer,value in captured:
                flat=value.reshape(-1,64);s=.2+.8*torch.sigmoid(layer.t_func(flat)/10)
                first=s*layer.proj(flat);second=.5*s.square()*layer.proj(layer.proj(flat))
                rms=lambda x:float(x.square().mean().sqrt())
                rows.append(dict(layer=name,time=clock,s_mean=float(s.mean()),s_std=float(s.std(unbiased=False)),
                    s_min=float(s.min()),s_max=float(s.max()),input_rms=rms(flat),output_rms=rms(layer(flat)),
                    term1_rms=rms(first),term2_rms=rms(second) if layer.orders==2 else 0.,probes=flat[:32].tolist()))
    write(folder/f'clocks_{step:07d}.json',rows)

def generate(model,source):
    generated=[]
    with torch.no_grad():
        for start in range(0,len(source),64):
            x=torch.from_numpy(source[start:start+64]).clone();labels=torch.zeros(len(x),dtype=torch.long)
            for i in range(32):
                t=torch.full((len(x),),i/32);next_t=torch.full_like(t,(i+1)/32)
                a=(model(x,t,labels)-x)/(1-t[:,None,None,None]).clamp_min(.05)
                z=x+a/32
                b=(model(z,next_t,labels)-z)/(1-next_t[:,None,None,None]).clamp_min(.05)
                x=x+(a+b)/64
            generated.append(x.numpy())
    return np.concatenate(generated)

def cell(root,arm):
    torch.set_num_threads(1);verify(root,True);folder=root/'output/runs'/arm;folder.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();model=model_for(root,arm);opt=optimizer(model);rng=torch.Generator().manual_seed(100000)
    images,indices=data_for(root);val=bank(root/'banks/validation.npz');digest=EMPTY;window=[];last={}
    def record(step):
        model.eval();pred=predict(model,val);np.savez_compressed(folder/f'validation_{step:07d}.npz',prediction=pred)
        result=dict(step=step,**metrics(pred,val))
        if window:
            result.update(train_loss=float(np.mean([x[0] for x in window])),grad_norm=float(np.mean([x[1] for x in window])),
                clip_fraction=float(np.mean([x[1]>1 for x in window])))
        append(folder/'metrics.jsonl',result);torch.save(checkpoint(model,opt,rng,step,digest),folder/f'step_{step:07d}.pt')
        if step in SNAPSHOTS:diagnostics(model,root/'banks',folder,step)
        model.train();return result
    try:
        last=record(0)
        for step in range(1,5001):
            assert time.monotonic()-start<1800,'Registered cell ceiling exceeded'
            batch=draw(images,indices,rng);digest=fingerprint(digest,batch);window.append(update(model,opt,batch))
            if step%100==0:
                row=dict(state='RUNNING',family=arm,pid=os.getpid(),step=step,total=5000,elapsed=time.monotonic()-start)
                write(folder/'status.json',row);print(json.dumps(row),flush=True)
            if step%500==0:last=record(step);window=[]
        model.eval();source=np.load(root/'banks/sampling-source.npy');samples=generate(model,source)
        assert np.isfinite(samples).all();np.save(folder/'generated.npy',samples)
        final=dict(family=arm,**last,sample_count=len(samples),out_of_range_fraction=float(np.mean(np.abs(samples)>1)),
            cell_seconds=time.monotonic()-start,input_stream_digest=digest)
        write(folder/'result.json',final);write(folder/'status.json',dict(state='COMPLETE',pid=os.getpid(),**final));verify(root,True)
    except BaseException as e:
        write(folder/'status.json',dict(state='FAILED',pid=os.getpid(),error=repr(e)));raise

def prepare(root,parent,data,envpass):
    root.mkdir(exist_ok=False);source=root/'source';source.mkdir();here=Path(__file__).resolve().parent;repo=here.parents[2]
    for p in here.iterdir():
        if p.is_file():shutil.copyfile(p,source/p.name)
    shutil.copytree(here/'results',source/'pre-registration-checks')
    shutil.copyfile(parent/'source/image_model.py',source/'image_model.py')
    shutil.copytree(parent/'source/jit',source/'jit',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(envpass,source/'environment-doc-pass.json');assert load(envpass)['status']=='PASS'
    settings=dict(parent=str(parent),data=str(data),families=ARMS,steps=5000,batch=64,workers=2,threads=1,device='cpu',
        metrics_python='/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python',metrics_device='mps',
        weight_cache='/Users/mr.z/ode-workspace-prep/image-bridge-inception',
        registration_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip())
    write(root/'settings.json',settings);banks=root/'banks';banks.mkdir()
    images=np.load(data/'images.uint8.npy',mmap_mode='r')
    with np.load(data/'split.npz') as f:ids=f['validation']
    target=torch.from_numpy(np.array(images[ids[:1024]])).float()/127.5-1;rng=torch.Generator().manual_seed(17101)
    noise=torch.randn(target.shape,generator=rng);t=torch.sigmoid(.8*torch.randn(1024,generator=rng)-.8)
    x=t[:,None,None,None]*target+(1-t[:,None,None,None])*noise
    np.savez_compressed(banks/'validation.npz',target=target.numpy(),source=noise.numpy(),x=x.numpy(),t=t.numpy(),delta=(1-t).clamp_min(.05).numpy())
    for clock in (.1,.3,.5,.7,.9):
        time_tensor=torch.full((64,),clock);z=time_tensor[:,None,None,None]*target[:64]+(1-time_tensor[:,None,None,None])*noise[:64]
        np.savez_compressed(banks/f'panel_{clock:.1f}.npz',target=target[:64].numpy(),source=noise[:64].numpy(),x=z.numpy(),t=time_tensor.numpy(),delta=(1-time_tensor).clamp_min(.05).numpy())
    rng=torch.Generator().manual_seed(17102);np.save(banks/'sampling-source.npy',torch.randn(1024,3,32,32,generator=rng).numpy())
    np.save(banks/'fid-reference.uint8.npy',np.array(images[ids[:2048]]))
    paths=[p for p in source.rglob('*') if p.is_file()]+[p for p in banks.iterdir() if p.is_file()]+[
        root/'settings.json',parent/'sealed-runtime.json',data/'manifest.json',data/'images.uint8.npy',data/'split.npz']
    metric_package=Path('/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/lib/python3.12/site-packages/torch_fidelity')
    paths+=list(metric_package.glob('*.py'))+list(Path(settings['weight_cache']).glob('checkpoints/*.pth'))
    assert len(list(Path(settings['weight_cache']).glob('checkpoints/*.pth')))==1
    write(root/'bindings.json',{str(p):sha(p) for p in paths});verify(root)
    print(json.dumps(dict(status='PREPARED',root=str(root),bindings=len(paths))),flush=True)

def preflight(root):
    from torch.utils._python_dispatch import TorchDispatchMode
    torch.set_num_threads(1);verify(root,True);images,indices=data_for(root);rows=[];common=None
    class Count(TorchDispatchMode):
        def __init__(self):self.macs=0
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            out=func(*args,**(kwargs or {}))
            if func==torch.ops.aten.mm.default:self.macs+=args[0].shape[0]*args[0].shape[1]*args[1].shape[1]
            if func==torch.ops.aten.addmm.default:self.macs+=args[1].shape[0]*args[1].shape[1]*args[2].shape[1]
            return out
    reference=model_for(root,'ode_k2')
    for arm in ARMS:
        initial=model_for(root,arm)
        for name,value in reference.state_dict().items():
            if '.mlp.' not in name or arm.startswith('ode'):
                torch.testing.assert_close(initial.state_dict()[name],value,atol=0,rtol=0)
        for name,value in reference.named_buffers():
            torch.testing.assert_close(dict(initial.named_buffers())[name],value,atol=0,rtol=0)
        if arm.startswith('ode'):
            ffn=initial.blocks[0].mlp;probe=torch.linspace(-1,1,384).reshape(2,3,64)
            g=torch.autograd.grad(ffn(probe,None).square().mean(),list(ffn.parameters()))
            assert all(torch.isfinite(v).all() and torch.count_nonzero(v) for v in g)
        ends=[]
        for interrupted in (False,True):
            model=model_for(root,arm);opt=optimizer(model);rng=torch.Generator().manual_seed(100000);digest=EMPTY
            for step in range(1,4):
                b=draw(images,indices,rng);digest=fingerprint(digest,b);update(model,opt,b)
                if interrupted and step==2:
                    state=checkpoint(model,opt,rng,step,digest);model=model_for(root,arm);opt=optimizer(model);digest=restore(model,opt,rng,state)
            ends.append(checkpoint(model,opt,rng,3,digest))
        equal(*ends)
        if common is None:common=ends[0]['input_stream_digest']
        assert common==ends[0]['input_stream_digest']
        layer=model_for(root,arm).blocks[0].mlp;x=torch.zeros(1,1,64,requires_grad=True);c=torch.zeros(1,64)
        with Count() as f:out=layer(x,c)
        layer.zero_grad(set_to_none=True);x.grad=None
        with Count() as both:layer(x,c).square().sum().backward()
        rows.append(dict(family=arm,resume_model_optimizer_rng_buffers_bitwise=True,
            dense_FFN_forward_MACs_per_token=f.macs,dense_FFN_training_MACs_per_token=both.macs))
    assert rows[2]['dense_FFN_training_MACs_per_token']>=rows[0]['dense_FFN_training_MACs_per_token']
    assert rows[3]['dense_FFN_training_MACs_per_token']>=rows[1]['dense_FFN_training_MACs_per_token']
    write(root/'preflight.json',dict(status='PASS',arms=rows,common_three_batch_digest=common))
    write(root/'ready.json',dict(bindings=sha(root/'bindings.json'),metric_bindings=sha(root/'metric-bindings.json'),preflight=sha(root/'preflight.json')))
    print(json.dumps(load(root/'preflight.json')),flush=True)

def run(root):
    verify(root,True);assert load(root/'ready.json')==dict(bindings=sha(root/'bindings.json'),metric_bindings=sha(root/'metric-bindings.json'),preflight=sha(root/'preflight.json'))
    output=root/'output';output.mkdir(exist_ok=False);pending=list(ARMS);live=[];launches=[];failures=[];started=time.monotonic()
    while pending or live:
        while pending and len(live)<2 and not failures:
            arm=pending.pop(0);log=(output/f'{arm}.log').open('w');env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
            cmd=[sys.executable,str(root/'source/study.py'),'cell','--root',str(root),'--family',arm]
            p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            row=dict(family=arm,pid=p.pid,command=cmd,started=time.time());launches.append(row);live.append((p,log,row));write(output/'launches.json',launches)
        processes=[]
        for p,log,row in list(live):
            code=p.poll()
            if code is None:
                try:
                    proc=psutil.Process(p.pid);processes.append(dict(pid=p.pid,rss_bytes=proc.memory_info().rss,cpu_seconds=sum(proc.cpu_times()[:2])))
                except psutil.NoSuchProcess:pass
                if time.time()-row['started']>1800:p.terminate();failures.append(dict(family=row['family'],reason='ceiling'))
            else:
                row['exit_code']=code;log.close();live.remove((p,log,row));write(output/'launches.json',launches)
                if code!=0:failures.append(dict(family=row['family'],exit_code=code))
                else:assert load(output/'runs'/row['family']/'status.json')['state']=='COMPLETE'
        append(output/'monitor.jsonl',dict(elapsed=time.monotonic()-started,processes=processes))
        write(output/'status.json',dict(state='RUNNING',pid=os.getpid(),pending=pending,active=[r['family'] for _,_,r in live],failures=failures,elapsed=time.monotonic()-started))
        if failures and not live:break
        time.sleep(5)
    assert not failures and not pending,failures
    verify(root,True);write(output/'status.json',dict(state='TRAINING_COMPLETE',pid=os.getpid(),families=ARMS,elapsed=time.monotonic()-started))
    print(json.dumps(load(output/'status.json')),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('prepare','preflight','cell','run'));ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--parent',type=Path);ap.add_argument('--data',type=Path);ap.add_argument('--env-pass',type=Path);ap.add_argument('--family',choices=ARMS)
    a=ap.parse_args()
    if a.action=='prepare':prepare(a.root,a.parent,a.data,a.env_pass)
    elif a.action=='preflight':preflight(a.root)
    elif a.action=='cell':cell(a.root,a.family)
    else:run(a.root)
