"""Pre-quality image compatibility checks and device throughput calibration."""
from __future__ import annotations
import argparse,copy,hashlib,importlib.metadata,json,os,platform,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import psutil
import torch
from image_model import build,ffn_cost

FAMILIES=('ode_k1','ode_k2','swiglu86','swiglu129')
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def sync(device):
    if device=='mps':torch.mps.synchronize()
def update(model,opt,batch):
    x,t,target=batch;opt.zero_grad(set_to_none=True)
    pred=model(x,t,torch.zeros(len(x),device=x.device,dtype=torch.long))
    loss=((pred-target)/(1-t[:,None,None,None]).clamp_min(.05)).square().mean()
    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    assert torch.isfinite(loss) and torch.isfinite(norm);opt.step()
    return float(loss.detach())
def opt_for(model):return torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=0.)

def prepare(root,repo):
    root.mkdir(exist_ok=False);source=root/'source';source.mkdir()
    here=Path(__file__).resolve().parent
    for p in here.iterdir():
        if p.is_file():shutil.copyfile(p,source/p.name)
    shutil.copytree(repo/'src',source/'jit',ignore=shutil.ignore_patterns('__pycache__'))
    frozen={str(p.relative_to(root)):sha(p) for p in source.rglob('*') if p.is_file()}
    write(root/'bindings.json',frozen)
    write(root/'settings.json',dict(registration_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        host=platform.node(),purpose='feasibility only, no quality training',families=FAMILIES,
        model_seed=0,threads=1,batch=64,warmup_updates=10,measured_updates=25,phase_ceiling_seconds=600))
    print(json.dumps(dict(status='PREPARED',root=str(root),source_files=len(frozen))),flush=True)

def run(root):
    s=json.loads((root/'settings.json').read_text());source=root/'source/jit';torch.set_num_threads(1)
    assert s['families']==list(FAMILIES);start=time.monotonic();proc=psutil.Process();write(root/'status.json',dict(state='RUNNING',pid=os.getpid()))
    models={family:build(source,family) for family in FAMILIES};ref=models['ode_k2']
    for family,model in models.items():
        for name,value in ref.state_dict().items():
            if '.mlp.' not in name:torch.testing.assert_close(model.state_dict()[name],value,atol=0,rtol=0)
        for name,value in ref.named_buffers():torch.testing.assert_close(dict(model.named_buffers())[name],value,atol=0,rtol=0)
    for name,value in models['ode_k1'].state_dict().items():torch.testing.assert_close(ref.state_dict()[name],value,atol=0,rtol=0)
    costs={family:ffn_cost(model) for family,model in models.items()}
    assert costs['swiglu86']['ffn_parameters_per_block']>=costs['ode_k2']['ffn_parameters_per_block']
    assert costs['swiglu129']['dense_FFN_forward_MACs_per_token_per_block']>=costs['ode_k2']['dense_FFN_forward_MACs_per_token_per_block']
    g=torch.Generator().manual_seed(0);x=torch.randn(2,5,64,generator=g,dtype=torch.float64)
    condition=torch.randn(2,64,generator=g,dtype=torch.float64,requires_grad=True);maths=[]
    for family in ('ode_k1','ode_k2'):
        ffn=copy.deepcopy(models[family].blocks[0].mlp).double();actual=ffn(x,condition)
        torch.testing.assert_close(actual,ffn(x,condition*3+1),atol=0,rtol=0)
        assert torch.autograd.grad(actual.sum(),condition,allow_unused=True)[0] is None
        for layer in (ffn.ode1,ffn.ode2):
            z=x[0,:2].clone().requires_grad_(True);p=layer.proj.weight
            s=.2+.8*torch.sigmoid(layer.t_func(z)/10)
            px=z@p.T;want=z+s*px
            if layer.orders==2:want=want+.5*s*s*(px@p.T)
            torch.testing.assert_close(layer(z),want,atol=1e-12,rtol=1e-10)
            ds=torch.func.vmap(torch.func.jacrev(lambda a:.2+.8*torch.sigmoid(layer.t_func(a)/10)))(z).squeeze(1)
            j=torch.eye(64)+s[:,:,None]*p;u=px
            if layer.orders==2:j=j+.5*s.square()[:,:,None]*(p@p);u=u+s*(px@p.T)
            j=j+u[:,:,None]*ds[:,None,:]
            measured=torch.func.vmap(torch.func.jacrev(layer))(z)
            torch.testing.assert_close(j,measured,atol=1e-12,rtol=1e-10)
            maths.append(float((j-measured).detach().abs().max()))
    # Exact native CPU checkpoint/resume on two updates plus one restored update.
    batches=[]
    for k in range(3):
        target=torch.rand(4,3,32,32,generator=g)*2-1;t=torch.full((4,),.4)
        z=.4*target+.6*torch.randn(target.shape,generator=g);batches.append((z,t,target))
    resume=[]
    for family in FAMILIES:
        m=build(source,family);opt=opt_for(m)
        update(m,opt,batches[0]);update(m,opt,batches[1]);state=copy.deepcopy((m.state_dict(),opt.state_dict()))
        update(m,opt,batches[2]);second=build(source,family);other=opt_for(second)
        second.load_state_dict(state[0]);other.load_state_dict(state[1]);update(second,other,batches[2])
        for key,value in m.state_dict().items():torch.testing.assert_close(second.state_dict()[key],value,atol=0,rtol=0)
        # Every moment, scalar step and parameter order is replayed too.
        def equal(a,b):
            if torch.is_tensor(a):assert torch.equal(a,b)
            elif isinstance(a,dict):assert a.keys()==b.keys();[equal(a[k],b[k]) for k in a]
            elif isinstance(a,(list,tuple)):assert len(a)==len(b);[equal(x,y) for x,y in zip(a,b)]
            else:assert a==b
        equal(opt.state_dict(),other.state_dict());resume.append(family)
    del models,ref
    rows=[]
    for device in ('cpu','mps'):
        if device=='mps' and not torch.backends.mps.is_available():continue
        target=torch.rand(64,3,32,32,generator=g)*2-1;t=torch.full((64,),.4)
        z=.4*target+.6*torch.randn(target.shape,generator=g)
        batch=tuple(a.to(device) for a in (z,t,target))
        for family in FAMILIES:
            assert time.monotonic()-start<600
            model=build(source,family).to(device);opt=opt_for(model)
            for k in range(s['warmup_updates']):update(model,opt,batch)
            sync(device);times=[];mps_mem=[]
            for k in range(s['measured_updates']):
                tick=time.perf_counter();update(model,opt,batch);sync(device);times.append(time.perf_counter()-tick)
                if device=='mps':mps_mem.append(torch.mps.current_allocated_memory())
            row=dict(device=device,family=family,median_seconds_per_update=float(np.median(times)),
                mean_seconds_per_update=float(np.mean(times)),updates=s['measured_updates'],batch=64,
                rss_GiB=proc.memory_info().rss/2**30,sampled_mps_allocated_GiB=max(mps_mem,default=0)/2**30,
                projected_5000_update_seconds=5000*float(np.median(times)),**costs[family])
            rows.append(row);print(json.dumps(row),flush=True)
            write(root/'status.json',dict(state='RUNNING',pid=os.getpid(),completed=len(rows),total=8,elapsed=time.monotonic()-start))
            del model,opt
            if device=='mps':torch.mps.empty_cache()
    for p,h in json.loads((root/'bindings.json').read_text()).items():assert sha(root/p)==h,p
    result=dict(status='PASS',purpose='calibration only; random synthetic image batches, no CIFAR quality result',
        seconds=time.monotonic()-start,maths_max=max(maths),input_clock_ignores_condition=True,
        common_backbone_bitwise=True,cpu_resume_models_and_moments_bitwise=resume,rows=rows,
        environment={k:importlib.metadata.version(k) for k in ('torch','numpy','einops','Pillow','psutil')})
    write(root/'preflight.json',result);write(root/'status.json',dict(state='COMPLETE',pid=os.getpid(),seconds=result['seconds']))
    print(json.dumps(dict(status='PREFLIGHT_PASS',seconds=result['seconds'])),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('prepare','run'));ap.add_argument('--root',type=Path,required=True);ap.add_argument('--repo',type=Path)
    args=ap.parse_args()
    if args.action=='prepare':prepare(args.root,args.repo)
    else:run(args.root)
