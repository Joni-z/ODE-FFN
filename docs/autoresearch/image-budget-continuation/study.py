"""One fixed continuation of all existing image controls; native semantics."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,copy,hashlib,importlib.util,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import psutil
import torch

BASE=Path('/Users/mr.z/ode-workspace-prep/autoresearch')
PARENTS={'native':BASE/'oct28-cycle17-v2','ordinary':BASE/'oct28-cycle20','resolution':BASE/'oct28-cycle18'}
SEALS={'native':'62d178c4e2b88780495c9ea45c6aeef96453ce7098a8aeee84076cf58f081172',
       'ordinary':'d7fe17e92c7bd8468b6c048aa9925a4350d95168ed611682bb9c7d787634e973',
       'resolution':'0a6fb24d1aba029be90e95f7c74919bbbdead973c101819852f04cb55cc64108'}
ARMS=('ode_k1','ode_k2','swiglu86','swiglu129','silu64','silu131','silu193')
TRAIN='/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python'
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def append(p,v):
    with Path(p).open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n')
def imported(key):
    name='image_budget_parent_'+key;path=PARENTS[key]/'source/study.py'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return sys.modules[name]
def verify(root):
    for path,digest in load(root/'bindings.json').items():assert sha(path)==digest,path
    for key,parent in PARENTS.items():
        assert sha(parent/'sealed-runtime.json')==SEALS[key]
        for file,digest in load(parent/'sealed-runtime.json').items():assert sha(parent/file)==digest,file
def parent_for(arm):return PARENTS['ordinary' if arm.startswith('silu') else 'native']
def initial(arm):return torch.load(parent_for(arm)/'output/runs'/arm/'step_0005000.pt',weights_only=False,map_location='cpu')
def construct(arm):
    return imported('ordinary').model_for(PARENTS['ordinary'],arm) if arm.startswith('silu') else imported('native').model_for(PARENTS['native'],arm)
def restored(arm,state=None):
    old=imported('native');model=construct(arm);opt=old.optimizer(model);rng=torch.Generator()
    old.restore(model,opt,rng,initial(arm) if state is None else state)
    return model,opt,rng
def prepare(root):
    assert not root.exists();root.mkdir();source=root/'source';source.mkdir()
    for name in ('study.py','protocol.md'):shutil.copyfile(Path(__file__).parent/name,source/name)
    repo=Path(__file__).resolve().parents[3]
    write(root/'settings.json',dict(source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        arms=ARMS,start_step=5000,end_step=20000,diagnostic_steps=[5000,10000,15000,20000],sample_steps=[10000,20000],
        workers=2,threads=1,model_seed=0,new_architecture=False,parents={k:str(v) for k,v in PARENTS.items()}))
    files=[*source.iterdir(),root/'settings.json',*(p/'sealed-runtime.json' for p in PARENTS.values())]
    write(root/'bindings.json',{str(p):sha(p) for p in files});verify(root)
    print('PREPARED',flush=True)
def preflight(root):
    torch.set_num_threads(1);verify(root);assert sha(__file__)==sha(root/'source/study.py')
    old=imported('native');images,ids=old.data_for(PARENTS['native'])
    bank=old.bank(PARENTS['native']/'banks/validation.npz');receipts=[];attempt=root/'preflight';attempt.mkdir(exist_ok=True)
    number=len(list(attempt.glob('attempt-*.json')))+1
    folder=attempt/f'fixtures-{number:02d}';folder.mkdir()
    for arm in ARMS:
        state=initial(arm);assert state['step']==5000
        m,opt,rng=restored(arm,state);m.eval();pred=old.predict(m,bank)
        with np.load(parent_for(arm)/'output/runs'/arm/'validation_0005000.npz') as saved:assert pred.tobytes()==saved['prediction'].tobytes(),arm
        digest=state['input_stream_digest'];m.train()
        for step in range(3):
            batch=old.draw(images,ids,rng);digest=old.fingerprint(digest,batch);old.update(m,opt,batch)
        direct=old.checkpoint(m,opt,rng,5003,digest)
        m,opt,rng=restored(arm,state);m.train();digest=state['input_stream_digest']
        batch=old.draw(images,ids,rng);digest=old.fingerprint(digest,batch);old.update(m,opt,batch)
        file=folder/f'{arm}-after-one.pt';torch.save(old.checkpoint(m,opt,rng,5001,digest),file)
        disk=torch.load(file,weights_only=False,map_location='cpu');m,opt,rng=restored(arm,disk);m.train();digest=disk['input_stream_digest']
        for step in range(2):
            batch=old.draw(images,ids,rng);digest=old.fingerprint(digest,batch);old.update(m,opt,batch)
        resumed=old.checkpoint(m,opt,rng,5003,digest);old.equal(direct,resumed)
        if arm.startswith('ode'):
            assert any(not torch.equal(v,state['model'][k]) for k,v in resumed['model'].items() if '.proj.weight' in k and '.mlp.' in k)
        receipts.append(dict(arm=arm,native_1024_predictions_bitwise=True,physical_model_optimizer_rng_buffers_resume=True,
            source_stream=state['input_stream_digest'],resumed_stream=digest,fixture=str(file)))
    assert len({r['source_stream'] for r in receipts})==len({r['resumed_stream'] for r in receipts})==1
    receipt=dict(status='PASS',attempt=number,rows=receipts,torch=torch.__version__,numpy=np.__version__,
                 MPS_available=torch.backends.mps.is_available(),executable=sys.executable,formal_training_started=False)
    write(attempt/f'attempt-{number:02d}.json',receipt);write(root/'preflight.json',receipt);verify(root)
    print(json.dumps(dict(status='PASS',arms=len(receipts),physical_resumes=len(receipts),attempt=number)),flush=True)
def histories(model,folder,step):
    old=imported('native');old.diagnostics(model,PARENTS['native']/'banks',folder,step)
    weights=[]
    for j,block in enumerate(model.blocks):
        for name,p in block.mlp.named_parameters():
            if p.ndim==2:
                a=p.detach().double();weights.append(dict(block=j,name=name,singular_values=torch.linalg.svdvals(a).tolist(),frobenius=float(torch.linalg.norm(a))))
    write(folder/f'matrix-history-{step:07d}.json',weights)
def cell(root,arm):
    torch.set_num_threads(1);verify(root);old=imported('native');folder=root/'output/runs'/arm;folder.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();model,opt,rng=restored(arm);state=initial(arm);digest=state['input_stream_digest']
    images,ids=old.data_for(PARENTS['native']);bank=old.bank(PARENTS['native']/'banks/validation.npz');window=[]
    def record(step):
        model.eval();pred=old.predict(model,bank);np.savez_compressed(folder/f'validation_{step:07d}.npz',prediction=pred)
        row=dict(step=step,**old.metrics(pred,bank))
        if window:row.update(train_loss=float(np.mean([v[0] for v in window])),grad_norm=float(np.mean([v[1] for v in window])),clip_fraction=float(np.mean([v[1]>1 for v in window])))
        append(folder/'metrics.jsonl',row);torch.save(old.checkpoint(model,opt,rng,step,digest),folder/f'step_{step:07d}.pt')
        if step in (5000,10000,15000,20000):histories(model,folder,step)
        model.train();return row
    try:
        last=record(5000)
        for step in range(5001,20001):
            assert time.monotonic()-start<7200,'Cell ceiling exceeded'
            batch=old.draw(images,ids,rng);digest=old.fingerprint(digest,batch);window.append(old.update(model,opt,batch))
            if step%100==0:write(folder/'status.json',dict(state='RUNNING',pid=os.getpid(),arm=arm,step=step,total=20000,elapsed=time.monotonic()-start))
            if step%1000==0:last=record(step);window=[]
            if step in (10000,20000):
                model.eval();samples=imported('resolution').integrate(model,np.load(PARENTS['native']/'banks/sampling-source.npy'),128)
                np.save(folder/f'generated-{step:07d}.npy',samples);model.train()
        write(folder/'result.json',dict(arm=arm,**last,input_stream_digest=digest,seconds=time.monotonic()-start,new_architecture=False))
        write(folder/'status.json',dict(state='COMPLETE',pid=os.getpid(),arm=arm,step=20000))
    except BaseException as e:
        write(folder/'status.json',dict(state='FAILED',pid=os.getpid(),arm=arm,error=repr(e)));raise
    verify(root)
def run(root):
    verify(root);assert load(root/'preflight.json')['status']=='PASS';assert load(root/'documented-pass.json')['status']=='PASS'
    output=root/'output';output.mkdir();pending=list(ARMS);active={};launches=[];exits=[];start=time.monotonic()
    try:
        while pending or active:
            assert time.monotonic()-start<28800,'Cohort ceiling exceeded'
            while pending and len(active)<2:
                arm=pending.pop(0);command=[TRAIN,'-B',str(root/'source/study.py'),'cell','--root',str(root),'--arm',arm]
                with (output/f'{arm}.log').open('w') as f:p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
                monitor=psutil.Process(p.pid);monitor.cpu_percent();active[arm]=(p,monitor,time.monotonic())
                launches.append(dict(arm=arm,pid=p.pid,create_time=monitor.create_time(),command=command));write(output/'launches.json',launches)
            resources=[]
            for arm,(p,monitor,t0) in list(active.items()):
                code=p.poll()
                if code is not None:
                    exits.append(dict(arm=arm,pid=p.pid,exit_code=code,seconds=time.monotonic()-t0));del active[arm]
                    write(output/'exits.json',exits)
                    if code!=0:raise RuntimeError(f'{arm} failed; no automatic retry')
                else:
                    resources.append(dict(arm=arm,pid=p.pid,RSS=monitor.memory_info().rss,CPU_percent=monitor.cpu_percent()))
            append(output/'resources.jsonl',dict(time=time.time(),resources=resources))
            write(output/'status.json',dict(state='RUNNING' if active or pending else 'COMPLETE',active=list(active),pending=pending,finished=[v['arm'] for v in exits],elapsed=time.monotonic()-start))
            if active:time.sleep(5)
    except BaseException as error:
        for process,_,_ in active.values():
            if process.poll() is None:process.terminate()
        for process,_,_ in active.values():process.wait(timeout=30)
        write(output/'status.json',dict(state='FAILED',error=repr(error),exits=exits,pending=pending))
        raise
    verify(root);write(output/'summary.json',dict(state='COMPLETE',seconds=time.monotonic()-start,exits=exits))
    print(json.dumps(load(output/'summary.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','preflight','cell','run'));p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    globals()[a.action](a.root,a.arm) if a.action=='cell' else globals()[a.action](a.root)
