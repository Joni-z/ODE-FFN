"""Cold bias-free image SiLU controls at three derived budgets."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,csv,hashlib,importlib.util,json,math,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import psutil
import torch
from torch import nn
ARMS=('silu64','silu131','silu193');WIDTHS=dict(zip(ARMS,(64,131,193)))
SEALS={'image_parent':'62d178c4e2b88780495c9ea45c6aeef96453ce7098a8aeee84076cf58f081172','resolution_parent':'0a6fb24d1aba029be90e95f7c74919bbbdead973c101819852f04cb55cc64108','clock_parent':'c8ecd01c4c24ad6be6f3a8d1bf3aa7b5b3222f03712e4879e2cd5ecf15759d96'}
TRAIN='/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python';METRICS='/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python'
EMPTY=hashlib.sha256(b'').hexdigest();SNAPSHOTS=(0,1000,2500,5000)
def load(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def append(p,v):
    with Path(p).open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n')
def paths(root):return {k:Path(v) for k,v in load(root/'settings.json').items() if k.endswith('_parent')}
def imported(root,key):
    path=paths(root)[key]/'source/study.py';name='cycle20_bound_'+key
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    assert Path(sys.modules[name].__file__).resolve()==path;return sys.modules[name]
def verify(root):
    for file,digest in load(root/'bindings.json').items():assert sha(file)==digest,file
    for key,p in paths(root).items():
        assert sha(p/'sealed-runtime.json')==SEALS[key]
        for file,digest in load(p/'sealed-runtime.json').items():assert sha(p/file)==digest,file
    imported(root,'image_parent').verify(paths(root)['image_parent'],True)

class PlainSiLU(nn.Module):
    def __init__(self,width):
        super().__init__();self.fc1=nn.Linear(64,width,bias=False);self.fc2=nn.Linear(width,64,bias=False)
    def forward(self,x,cond=None,return_aux=False):
        out=self.fc2(torch.nn.functional.silu(self.fc1(x)))
        aux={'input_norm':x.detach().norm(dim=-1).mean(),'output_norm':out.detach().norm(dim=-1).mean()}
        return (out,aux) if return_aux else out

def model_for(root,arm):
    old=imported(root,'image_parent');model=old.model_for(paths(root)['image_parent'],'swiglu86');torch.manual_seed(0)
    for block in model.blocks:block.mlp=PlainSiLU(WIDTHS[arm])
    return model

def matrix_history(model):
    rows=[]
    for j,block in enumerate(model.blocks):
        for name in ('fc1','fc2'):
            w=getattr(block.mlp,name).weight.detach().double().numpy();sv=np.linalg.svd(w,compute_uv=False)
            row=dict(layer=f'blocks.{j}.{name}',shape=list(w.shape),singular_values=sv.tolist(),frobenius=float(np.linalg.norm(w)),sigma_max=float(sv[0]),singular_CV=float(sv.std()/sv.mean()))
            if w.shape[0]==w.shape[1]:
                eigen=np.linalg.eigvals(w);row.update(eigen_real=eigen.real.tolist(),eigen_imag=eigen.imag.tolist())
            rows.append(row)
    return rows

def prepare(root,parents):
    assert not root.exists();root.mkdir();(root/'source').mkdir()
    for f in Path(__file__).resolve().parent.iterdir():
        if f.is_file():shutil.copyfile(f,root/'source'/f.name)
    write(root/'settings.json',{**{k:str(v.resolve()) for k,v in parents.items()},'model_seed':0,'data_generator_start':100000,'updates':5000,'sample_steps':128,'arms':ARMS,'widths':WIDTHS})
    files=[*list((root/'source').iterdir()),root/'settings.json',*(p/'sealed-runtime.json' for p in parents.values())]
    write(root/'bindings.json',{str(f.resolve()):sha(f) for f in files if f.is_file()});verify(root);print(json.dumps(dict(status='PREPARED')),flush=True)

def preflight(root):
    from torch.utils._python_dispatch import TorchDispatchMode
    torch.set_num_threads(1);verify(root);old=imported(root,'image_parent');p=paths(root)['image_parent'];images,indices=old.data_for(p)
    reference=torch.load(p/'output/runs/swiglu86/step_0000000.pt',weights_only=False);trial=root/'preflight-checkpoints';trial.mkdir();rows=[];common=None
    class Count(TorchDispatchMode):
        def __init__(self):self.macs=0
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            output=func(*args,**(kwargs or {}))
            if func==torch.ops.aten.mm.default:self.macs+=args[0].shape[0]*args[0].shape[1]*args[1].shape[1]
            if func==torch.ops.aten.addmm.default:self.macs+=args[1].shape[0]*args[1].shape[1]*args[2].shape[1]
            return output
    for arm in ARMS:
        model=model_for(root,arm)
        for name,value in model.state_dict().items():
            if not name.startswith(tuple(f'blocks.{j}.mlp.' for j in range(3))):assert torch.equal(value,reference['model'][name]),name
        for name,value in model.named_buffers():assert torch.equal(value,reference['runtime_buffers'][name]),name
        total=sum(p.numel() for p in model.parameters() if p.requires_grad);assert total==161168+3*128*WIDTHS[arm]
        layer=model_for(root,arm).blocks[0].mlp.double();torch.manual_seed(0);x=torch.randn(2,3,64,dtype=torch.float64,requires_grad=True);direction=torch.randn_like(x)
        actual,jvp=torch.autograd.functional.jvp(lambda value:layer(value,None),x,direction)
        w1=layer.fc1.weight;w2=layer.fc2.weight;hidden=x@w1.T;sigma=hidden.sigmoid();derivative=sigma+hidden*sigma*(1-sigma)
        expected=(hidden*sigma)@w2.T;expected_jvp=((direction@w1.T)*derivative)@w2.T
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12);torch.testing.assert_close(jvp,expected_jvp,atol=1e-12,rtol=1e-12)
        probe=model_for(root,arm)
        with torch.no_grad():
            probe.final_layer.linear.weight.fill_(.01)
            for block in probe.blocks:block.adaLN_modulation.linear.bias[320:384]=.1
        probe(torch.randn(2,3,32,32),torch.full((2,),.4),torch.zeros(2,dtype=torch.long)).square().mean().backward()
        norms=[float(getattr(block.mlp,name).weight.grad.norm()) for block in probe.blocks for name in ('fc1','fc2')];assert all(np.isfinite(n) and n>0 for n in norms)
        ends=[]
        for interrupted in (False,True):
            model=model_for(root,arm);opt=old.optimizer(model);rng=torch.Generator().manual_seed(100000);digest=EMPTY
            for step in range(1,4):
                batch=old.draw(images,indices,rng);digest=old.fingerprint(digest,batch);old.update(model,opt,batch)
                if interrupted and step==2:
                    state=old.checkpoint(model,opt,rng,step,digest);path=trial/f'{arm}.pt';torch.save(state,path);stored=torch.load(path,weights_only=False);old.equal(state,stored)
                    model=model_for(root,arm);opt=old.optimizer(model);digest=old.restore(model,opt,rng,stored)
            ends.append(old.checkpoint(model,opt,rng,3,digest))
        old.equal(*ends)
        if common is None:common=ends[0]['input_stream_digest']
        assert common==ends[0]['input_stream_digest']==load(p/'preflight.json')['common_three_batch_digest']
        layer=model_for(root,arm).blocks[0].mlp;x=torch.zeros(1,1,64,requires_grad=True)
        with Count() as forward:layer(x,None)
        with Count() as training:layer(x,None).square().sum().backward()
        assert forward.macs==128*WIDTHS[arm] and training.macs==3*forward.macs
        source=np.load(p/'banks/sampling-source.npy')[:64];actual=imported(root,'resolution_parent').integrate(model_for(root,arm).eval(),source,128)
        factor=next(r['zero_clean_factor'] for r in load(paths(root)['resolution_parent']/'preflight.json')['fixtures'] if r['N']==128)
        np.testing.assert_allclose(actual,source*factor,atol=1e-6,rtol=1e-5)
        rows.append(dict(family=arm,parameters=total,FFN_parameters=128*WIDTHS[arm],FFN_forward_MACs=forward.macs,FFN_training_MACs=training.macs,common_backbone_bitwise=True,disk_resume_model_optimizer_rng_buffers_bitwise=True,active_matrix_grad_norms=norms,isolated_JVP_max=float((jvp-expected_jvp).detach().abs().max()),cold_sampler_matches=True))
    verify(root);write(root/'preflight.json',dict(status='PASS',arms=rows,common_three_batch_digest=common,checkpoints_physical=True))
    write(root/'ready.json',dict(bindings=sha(root/'bindings.json'),preflight=sha(root/'preflight.json')));print(json.dumps(load(root/'preflight.json')),flush=True)

def cell(root,arm):
    torch.set_num_threads(1);verify(root);assert load(root/'ready.json')['preflight']==sha(root/'preflight.json');old=imported(root,'image_parent');p=paths(root)['image_parent']
    folder=root/'output/runs'/arm;folder.mkdir(parents=True);started=time.monotonic();model=model_for(root,arm);opt=old.optimizer(model);rng=torch.Generator().manual_seed(100000)
    images,indices=old.data_for(p);val=old.bank(p/'banks/validation.npz');digest=EMPTY;window=[]
    def record(step):
        model.eval();prediction=old.predict(model,val);np.savez_compressed(folder/f'validation_{step:07d}.npz',prediction=prediction);result=dict(step=step,**old.metrics(prediction,val))
        if window:result.update(train_loss=float(np.mean([r[0] for r in window])),grad_norm=float(np.mean([r[1] for r in window])),clip_fraction=float(np.mean([r[1]>1 for r in window])))
        append(folder/'metrics.jsonl',result);torch.save(old.checkpoint(model,opt,rng,step,digest),folder/f'step_{step:07d}.pt')
        if step in SNAPSHOTS:
            write(folder/f'matrices_{step:07d}.json',matrix_history(model))
            for t in (.1,.3,.5,.7,.9):
                panel=old.bank(p/'banks'/f'panel_{t:.1f}.npz');np.savez_compressed(folder/f'panel_{step:07d}_{t:.1f}.npz',prediction=old.predict(model,panel))
        model.train();return result
    last=record(0)
    try:
        for step in range(1,5001):
            assert time.monotonic()-started<1800,'Registered cell ceiling'
            batch=old.draw(images,indices,rng);digest=old.fingerprint(digest,batch);window.append(old.update(model,opt,batch))
            if step%100==0:
                status=dict(state='RUNNING',family=arm,step=step,total=5000,seconds=time.monotonic()-started);write(folder/'status.json',status);print(json.dumps(status),flush=True)
            if step%500==0:last=record(step);window=[]
        model.eval();source=np.load(p/'banks/sampling-source.npy')
        def progress(count):write(folder/'status.json',dict(state='SAMPLING',family=arm,processed=count,total=1024,step=5000,seconds=time.monotonic()-started))
        generated=imported(root,'resolution_parent').integrate(model,source,128,progress);np.save(folder/'generated.npy',generated)
        final=dict(family=arm,**last,out_of_range_fraction=float(np.mean(np.abs(generated)>1)),sample_count=1024,steps=128,seconds=time.monotonic()-started,input_stream_digest=digest)
        write(folder/'result.json',final);write(folder/'status.json',dict(state='COMPLETE',**final));verify(root)
    except BaseException as error:write(folder/'status.json',dict(state='FAILED',error=repr(error)));raise

def run(root):
    verify(root);assert load(root/'preflight.json')['status']=='PASS';out=root/'output';out.mkdir();waiting=list(ARMS);active={};launches=[];exits=[];start=time.monotonic()
    try:
        while waiting or active:
            assert time.monotonic()-start<3600,'Queue wall ceiling'
            while waiting and len(active)<2:
                arm=waiting.pop(0);cmd=[TRAIN,str(root/'source/study.py'),'cell','--root',str(root),'--arm',arm]
                with (out/f'{arm}.log').open('w') as log:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                active[arm]=(proc,time.monotonic());launches.append(dict(family=arm,pid=proc.pid,create_time=psutil.Process(proc.pid).create_time(),command=cmd));write(out/'launches.json',launches)
            resources=[]
            for arm,(proc,started) in list(active.items()):
                code=proc.poll()
                if code is not None:
                    exits.append(dict(family=arm,exit_code=code,seconds=time.monotonic()-started));del active[arm];write(out/'exits.json',exits);assert code==0,exits[-1]
                else:
                    assert time.monotonic()-started<1800,(arm,'cell ceiling')
                    try:
                        job=psutil.Process(proc.pid);resources.append(dict(family=arm,pid=proc.pid,cpu_seconds=sum(job.cpu_times()[:2]),rss=job.memory_info().rss))
                    except psutil.NoSuchProcess:pass
            append(out/'resources.jsonl',dict(elapsed=time.monotonic()-start,processes=resources));write(out/'status.json',dict(state='RUNNING' if active or waiting else 'TRAINING_COMPLETE',completed=len(exits),total=3,seconds=time.monotonic()-start))
            if active:time.sleep(5)
        verify(root)
    except BaseException:
        for proc,_ in active.values():
            if proc.poll() is None:proc.terminate()
        write(out/'status.json',dict(state='FAILED',exits=exits));raise

def feature_path(root,arm):
    p=paths(root)
    if arm in ARMS:return root/'metric-output'/f'{arm}-features.npy'
    if arm.endswith('_constant'):return p['clock_parent']/'metric-output'/f'{arm}-features.npy'
    return p['resolution_parent']/'metric-output'/f'{arm}-128-features.npy'

def evaluate(root):
    torch.set_num_threads(1);verify(root);assert load(root/'output/status.json')['state']=='TRAINING_COMPLETE';p=paths(root);sys.path.insert(0,str(p['image_parent']/'source'));import image_metrics as m
    from metric_core import fid,uint8_images
    out=root/'metric-output';out.mkdir();extractor=m.extractor(p['image_parent']);real=np.load(p['image_parent']/'metric-inputs/reference-features.npy')
    rows=load(p['clock_parent']/'metric-metadata-v2/result.json')['rows']
    for row in rows:row.update(provenance='sealed prior endpoint')
    for arm in ARMS:
        raw=np.load(root/'output/runs'/arm/'generated.npy');features=m.features(extractor,uint8_images(raw),progress=arm);np.save(feature_path(root,arm),features)
        row=load(root/'output/runs'/arm/'result.json');row.update(fid=fid(features,real),provenance='new cold ordinary control');rows.append(row)
    for row in rows:
        f=np.load(feature_path(root,row['family'])).astype(np.float64);r=real.astype(np.float64);mean=float(np.sum((f.mean(0)-r.mean(0))**2));score=fid(f,r)
        if 'fid' in row:assert abs(score-float(row['fid']))<1e-8
        row.update(fid=score,mean_term=mean,covariance_term=score-mean,generated=1024,reference=2048,steps=128)
    verify(root);write(out/'result.json',dict(status='PASS',rows=rows))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('prepare','preflight','cell','run','evaluate'));ap.add_argument('--root',type=Path,required=True)
    for key in SEALS:ap.add_argument('--'+key.replace('_','-'),type=Path)
    ap.add_argument('--arm',choices=ARMS);a=ap.parse_args()
    if a.action=='prepare':prepare(a.root,{key:getattr(a,key) for key in SEALS})
    elif a.action=='cell':cell(a.root,a.arm)
    else:globals()[a.action](a.root)
