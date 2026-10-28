"""Export/check a frozen endpoint and measure actual synchronized inference."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse, copy, hashlib, json, shutil, subprocess, sys, threading, time, types
from pathlib import Path
import numpy as np
import psutil
import torch
from torch import nn
from deployment import CompactSiLU, CompactSwiGLU, builder, folded_block, load_export, restore

BASE=Path('/Users/mr.z/ode-workspace-prep/autoresearch')
PARENTS={k:BASE/v for k,v in {'native':'oct28-cycle17-v2','resolution':'oct28-cycle18',
    'clocks':'oct28-cycle19-v2','ordinary':'oct28-cycle20','maps':'oct28-cycle21-v2'}.items()}
SEALS={'native':'62d178c4e2b88780495c9ea45c6aeef96453ce7098a8aeee84076cf58f081172',
    'resolution':'0a6fb24d1aba029be90e95f7c74919bbbdead973c101819852f04cb55cc64108',
    'clocks':'c8ecd01c4c24ad6be6f3a8d1bf3aa7b5b3222f03712e4879e2cd5ecf15759d96',
    'ordinary':'d7fe17e92c7bd8468b6c048aa9925a4350d95168ed611682bb9c7d787634e973',
    'maps':'fa13abe370318077c521bb4202bf8f37b8ccfee6daf4cb5c0ef7ea92daf9b3cf'}
ARMS=('native','literal','cached','compact','swiglu129','silu193')
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sync(device):
    if device=='mps':torch.mps.synchronize()
def verify_parents():
    for key,root in PARENTS.items():
        assert sha(root/'sealed-runtime.json')==SEALS[key]
        for name,digest in load(root/'sealed-runtime.json').items():assert sha(root/name)==digest,(key,name)
def state_from(root,arm):
    state=torch.load(root/'output/runs'/arm/'step_0005000.pt',weights_only=False,map_location='cpu')
    assert state['step']==5000
    return state
def compact_matrices():
    with np.load(PARENTS['maps']/'operators/ode_k2-0005000.npz') as bank:
        return [(torch.from_numpy(bank[f'blocks_{j}_{name}_T32'].copy())) for j in range(3) for name in ('ode1','ode2')]
def model_for(bundle,arm):
    if arm=='compact':return load_export(bundle)
    family='swiglu129' if arm=='swiglu129' else 'swiglu86' if arm=='silu193' else 'ode_k2'
    model=builder(bundle).build(bundle/'source/jit',family)
    if arm=='silu193':
        for block in model.blocks:block.mlp=CompactSiLU(nn.Linear(64,193,bias=False),nn.Linear(193,64,bias=False))
    restore(model,state_from(PARENTS['ordinary'] if arm=='silu193' else PARENTS['native'],family if arm!='silu193' else arm))
    if arm=='swiglu129':
        for block in model.blocks:block.mlp=CompactSwiGLU(block.mlp)
    if arm in ('literal','cached'):
        constants=load(PARENTS['clocks']/'calibration.json')['ode_k2'];matrices=compact_matrices();index=0
        for j,block in enumerate(model.blocks):
            for name in ('ode1','ode2'):
                layer=getattr(block.mlp,name);c=constants[f'blocks.{j}.{name}']['value'];matrix=matrices[index];index+=1
                if arm=='literal':
                    def forward(self,x,t=None,c=c):
                        assert t is None
                        terms=[x];duration=x.new_tensor(c)
                        for i in range(self.orders):terms.append(self.proj(terms[-1])*duration/(i+1))
                        return sum(terms)
                else:
                    layer.register_buffer('cached_matrix',matrix,persistent=False)
                    def forward(self,x,t=None):
                        assert t is None
                        return torch.nn.functional.linear(x,self.cached_matrix)
                layer.forward=types.MethodType(forward,layer)
    return model.eval().requires_grad_(False)
def make_export(root):
    bundle=root/'export';(bundle/'source').mkdir(parents=True)
    source=PARENTS['native']/'source'
    shutil.copyfile(source/'image_model.py',bundle/'source/image_model.py')
    shutil.copytree(source/'jit',bundle/'source/jit',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(Path(__file__).parent/'deployment.py',bundle/'deployment.py')
    model=builder(bundle).build(bundle/'source/jit','ode_k2');restore(model,state_from(PARENTS['native'],'ode_k2'))
    maps=compact_matrices()
    for j,block in enumerate(model.blocks):block.mlp=folded_block(*maps[2*j:2*j+2])
    torch.save(dict(model=model.state_dict(),runtime_buffers={n:v for n,v in model.named_buffers()}),bundle/'weights.pt')
    write(bundle/'manifest.json',dict(family='compact_constant_k2',inference_only=True,
        parent_seals=SEALS,calibration_sha256=sha(PARENTS['clocks']/'calibration.json'),
        files={str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()},
        inherited_FID=179.7184452562523,FID_scope='Reused cycle21 CPU folded endpoint; no new quality estimate.'))
    return bundle
def error(a,b):
    d=a.astype(np.float64)-b.astype(np.float64)
    return dict(max=float(np.max(np.abs(d))),RMS=float(np.sqrt(np.mean(d*d))),bitwise=a.tobytes()==b.tobytes())
def predict(model,x,t,device):
    return np.concatenate([model(torch.from_numpy(x[i:i+64]).to(device),torch.from_numpy(t[i:i+64]).to(device),
        torch.zeros(min(64,len(x)-i),dtype=torch.long,device=device)).cpu().numpy() for i in range(0,len(x),64)])
def sample(model,source,device,deadline):
    x=torch.from_numpy(source).to(device);labels=torch.zeros(len(x),dtype=torch.long,device=device)
    for i in range(128):
        assert time.monotonic()<deadline
        t=torch.full((len(x),),i/128,device=device);nt=torch.full_like(t,(i+1)/128)
        v0=(model(x,t,labels)-x)/(1-t[:,None,None,None]).clamp_min(.05)
        proposal=x+v0/128
        v1=(model(proposal,nt,labels)-proposal)/(1-nt[:,None,None,None]).clamp_min(.05)
        x=x+(v0+v1)/256
    return x.cpu().numpy()
def main(root):
    assert not root.exists();root.mkdir(parents=True);started=time.monotonic();deadline=started+1200
    torch.set_num_threads(1);verify_parents();source=root/'source';source.mkdir()
    for name in ('benchmark.py','deployment.py','protocol.md'):shutil.copyfile(Path(__file__).parent/name,source/name)
    repo=Path(__file__).resolve().parents[3]
    write(root/'execution.json',dict(pid=os.getpid(),host=os.uname().nodename,command=sys.argv,
        executable=sys.executable,torch=torch.__version__,source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        parent_seals=SEALS,new_training=False,threads=1,backend_order=['cpu','mps']))
    stop=threading.Event()
    def monitor():
        while not stop.is_set():
            p=psutil.Process()
            with (root/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),RSS=p.memory_info().rss,CPU_percent=p.cpu_percent(),phase=load(root/'status.json') if (root/'status.json').exists() else {}))+'\n')
            stop.wait(5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        bundle=make_export(root);models={a:model_for(bundle,a) for a in ARMS}
        with np.load(PARENTS['native']/'banks/validation.npz') as bank:x,t=bank['x'],bank['t']
        cpu={};checks=[]
        expected={'native':PARENTS['native']/'output/runs/ode_k2/validation_0005000.npz',
            'literal':PARENTS['clocks']/'output/runs/ode_k2/validation.npz',
            'cached':PARENTS['maps']/'output/runs/ode_k2_folded/validation.npz',
            'compact':PARENTS['maps']/'output/runs/ode_k2_folded/validation.npz',
            'swiglu129':PARENTS['native']/'output/runs/swiglu129/validation_0005000.npz',
            'silu193':PARENTS['ordinary']/'output/runs/silu193/validation_0005000.npz'}
        with torch.no_grad():
            for arm,model in models.items():
                cpu[arm]=predict(model,x,t,'cpu')
                with np.load(expected[arm]) as bank:check=error(cpu[arm],bank['prediction'])
                assert check['bitwise'],(arm,check)
                checks.append(dict(arm=arm,device='cpu',**check))
        with torch.no_grad():
            smoke_prediction=predict(models['compact'],x[:1],t[:1],'cpu')
        np.savez(bundle/'smoke.npz',x=x[:1],t=t[:1],prediction=smoke_prediction)
        smoke="""import sys, pathlib, numpy as np, torch
bundle=pathlib.Path(sys.argv[1]); forbidden=sys.argv[2:]
def guard(event,args):
 if event=='open' and isinstance(args[0],(str,bytes)):
  path=str(pathlib.Path(args[0]).resolve())
  assert not any(path.startswith(v+'/') or path==v for v in forbidden),path
sys.addaudithook(guard);sys.path.insert(0,str(bundle))
from deployment import load_export
torch.set_num_threads(1);m=load_export(bundle)
with np.load(bundle/'smoke.npz') as b, torch.no_grad():
 p=m(torch.from_numpy(b['x']),torch.from_numpy(b['t']),torch.zeros(1,dtype=torch.long)).numpy()
 assert np.array_equal(p,b['prediction'])
print('PORTABLE_LOAD_PASS')
"""
        subprocess.run([sys.executable,'-B','-c',smoke,str(bundle),str(repo),*(str(p) for p in PARENTS.values())],check=True,cwd='/tmp',timeout=60)
        write(root/'correctness.json',dict(status='PASS',rows=checks,portable_fresh_process=True,historical_paths_forbidden=True))
        dense_work={}
        for arm,m in models.items():
            count=[0];hooks=[]
            def linear_hook(layer,inputs,output):count[0]+=output.numel()*layer.in_features
            def cached_hook(layer,inputs,output):count[0]+=output.numel()*64
            for block in m.blocks:
                for layer in block.mlp.modules():
                    if isinstance(layer,nn.Linear):hooks.append(layer.register_forward_hook(linear_hook))
                    if hasattr(layer,'cached_matrix'):hooks.append(layer.register_forward_hook(cached_hook))
            with torch.no_grad():m(torch.from_numpy(x[:1]),torch.from_numpy(t[:1]),torch.zeros(1,dtype=torch.long))
            for hook in hooks:hook.remove()
            dense_work[arm]=dict(total_FFN_dense_MACs_per_image=count[0],scope='All three tokenwise FFNs only; attention/backbone and pointwise operations excluded.')
        write(root/'dense-work.json',dense_work)
        resident=[];transfers=[];generations=[];cpu_samples={}
        source_images=np.load(PARENTS['native']/'banks/sampling-source.npy')[:64].copy()
        for device in ('cpu','mps'):
            if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('Registered MPS backend unavailable')
            active={a:m.to(device) for a,m in models.items()}
            write(root/'status.json',dict(phase='correctness',device=device))
            with torch.no_grad():
                if device=='mps':
                    for arm,m in active.items():
                        check=error(predict(m,x[:64],t[:64],device),cpu[arm][:64]);assert check['max']<=5e-4 and check['RMS']<=5e-5,(arm,check)
                        checks.append(dict(arm=arm,device=device,**check))
                for batch in (1,16,64):
                    xx=torch.from_numpy(x[:batch]).to(device);tt=torch.from_numpy(t[:batch]).to(device);labels=torch.zeros(batch,dtype=torch.long,device=device)
                    for m in active.values():
                        for _ in range(3):m(xx,tt,labels)
                    for trial in range(7):
                        for arm in ARMS[trial%6:]+ARMS[:trial%6]:
                            assert time.monotonic()<deadline
                            sync(device);tick=time.perf_counter()
                            for _ in range(5):active[arm](xx,tt,labels)
                            sync(device);seconds=(time.perf_counter()-tick)/5
                            resident.append(dict(device=device,batch=batch,trial=trial,arm=arm,seconds=seconds))
                    write(root/'status.json',dict(phase='resident_timing',device=device,batch=batch))
                for trial in range(7):
                    for arm in ARMS[trial%6:]+ARMS[:trial%6]:
                        sync(device);tick=time.perf_counter()
                        for _ in range(2):
                            active[arm](torch.from_numpy(x[:64]).to(device),torch.from_numpy(t[:64]).to(device),torch.zeros(64,dtype=torch.long,device=device)).cpu().numpy()
                        sync(device);transfers.append(dict(device=device,batch=64,trial=trial,arm=arm,seconds=(time.perf_counter()-tick)/2))
                for arm in ('native','cached','compact'):
                    write(root/'status.json',dict(phase='Heun128',device=device,arm=arm))
                    sync(device);tick=time.perf_counter();samples=sample(active[arm],source_images,device,deadline);sync(device);seconds=time.perf_counter()-tick
                    np.save(root/f'samples-{device}-{arm}.npy',samples)
                    if device=='cpu':
                        cpu_samples[arm]=samples
                        path=PARENTS['resolution']/'output/runs/ode_k2/generated-128.npy' if arm=='native' else PARENTS['maps']/'output/runs/ode_k2_folded/generated.npy'
                        check=error(samples,np.load(path)[:64]);assert check['bitwise'],(arm,check)
                    else:
                        check=error(samples,cpu_samples[arm]);assert check['max']<=.05 and check['RMS']<=1e-3,(arm,check)
                    generations.append(dict(arm=arm,device=device,images=64,steps=128,seconds=seconds,error=check))
            write(root/'resident-timings.json',resident);write(root/'transfer-timings.json',transfers);write(root/'generation-timings.json',generations)
        write(root/'correctness.json',dict(status='PASS',rows=checks,portable_fresh_process=True,historical_paths_forbidden=True))
        storage={a:dict(parameters=sum(p.numel() for p in m.parameters()),stored_bytes=sum(p.numel()*p.element_size() for p in m.parameters())+sum(b.numel()*b.element_size() for b in m.buffers())) for a,m in models.items()}
        write(root/'storage.json',storage);verify_parents()
        write(root/'status.json',dict(phase='COMPLETE',seconds=time.monotonic()-started,new_training=False))
        print(json.dumps(load(root/'status.json')),flush=True)
    finally:stop.set();thread.join(timeout=6)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args();main(a.root)
