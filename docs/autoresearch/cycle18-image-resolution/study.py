"""Frozen CIFAR checkpoints: vary only the uniform Heun mesh."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,hashlib,json,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import psutil
import torch

ARMS=('ode_k1','ode_k2','swiglu86','swiglu129');GRIDS=(32,64,128)
PARENT_SEAL='62d178c4e2b88780495c9ea45c6aeef96453ce7098a8aeee84076cf58f081172'
TRAIN=Path('/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python')
METRICS=Path('/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python')
def load(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
def parent(root):return Path(load(root/'settings.json')['parent'])
def verify_parent(p):
    assert sha(p/'sealed-runtime.json')==PARENT_SEAL
    seal=load(p/'sealed-runtime.json')
    for name,digest in seal.items():assert sha(p/name)==digest,name
    return len(seal)
def verify(root):
    for name,digest in load(root/'bindings.json').items():assert sha(name)==digest,name
    return verify_parent(parent(root))
def native(root):
    p=parent(root);sys.path.insert(0,str(p/'source'));import study as imported
    # The frozen file itself is named study.py; avoid accidental import of this
    # standalone module when invoked/imported by other phase scripts.
    if Path(imported.__file__).resolve()!=p/'source/study.py':
        import importlib.util
        spec=importlib.util.spec_from_file_location('cycle17_native_study',p/'source/study.py');imported=importlib.util.module_from_spec(spec);spec.loader.exec_module(imported)
    return imported

def model_for(root,arm):
    old=native(root);p=parent(root);model=old.model_for(p,arm)
    state=torch.load(p/'output/runs'/arm/'step_0005000.pt',map_location='cpu',weights_only=False)
    assert state['step']==5000;old.restore_model(model,state)
    return model.eval().requires_grad_(False)

def integrate(model,source,n,progress=None):
    answers=[]
    with torch.no_grad():
        for start in range(0,len(source),64):
            x=torch.from_numpy(source[start:start+64]).clone();labels=torch.zeros(len(x),dtype=torch.long)
            for i in range(n):
                t=torch.full((len(x),),i/n);nt=torch.full_like(t,(i+1)/n)
                v0=(model(x,t,labels)-x)/(1-t[:,None,None,None]).clamp_min(.05)
                proposed=x+v0/n
                v1=(model(proposed,nt,labels)-proposed)/(1-nt[:,None,None,None]).clamp_min(.05)
                x=x+(v0+v1)/(2*n)
            assert torch.isfinite(x).all();answers.append(x.numpy())
            if progress:progress(start+len(x))
    return np.concatenate(answers)

def prepare(root,p):
    assert not root.exists();verify_parent(p);root.mkdir();src=root/'source';src.mkdir()
    for f in Path(__file__).resolve().parent.iterdir():
        if f.is_file():shutil.copyfile(f,src/f.name)
    write(root/'settings.json',dict(parent=str(p.resolve()),grids=GRIDS,arms=ARMS,images=1024,reference=2048,
        training_backend='cpu',feature_backend='mps',workers=2,threads=1,new_training=False))
    paths=[*src.iterdir(),root/'settings.json',p/'sealed-runtime.json']
    write(root/'bindings.json',{str(f.resolve()):sha(f) for f in paths if f.is_file()})
    print(json.dumps(dict(state='PREPARED',parent_files=verify(root))),flush=True)

def preflight(root):
    torch.set_num_threads(1);start=time.monotonic();verify(root)
    source=np.load(parent(root)/'banks/sampling-source.npy');assert source.shape==(1024,3,32,32)
    class Zero(torch.nn.Module):
        def forward(self,x,t,y):return torch.zeros_like(x)
    class ConstantVelocity(torch.nn.Module):
        def forward(self,x,t,y):return x+(1-t[:,None,None,None]).clamp_min(.05)*.25
    fixtures=[]
    for n in GRIDS:
        factor=1.
        for i in range(n):
            d0=max(1-i/n,.05);d1=max(1-(i+1)/n,.05)
            factor*=1+(-1/d0-(1-1/(n*d0))/d1)/(2*n)
        got=integrate(Zero(),source[:2],n)
        np.testing.assert_allclose(got,source[:2]*factor,atol=3e-8,rtol=3e-6)
        moved=integrate(ConstantVelocity(),source[:2],n)
        np.testing.assert_allclose(moved,source[:2]+.25,atol=3e-6,rtol=1e-6)
        fixtures.append(dict(N=n,zero_clean_factor=factor,constant_velocity=True))
    for arm in ARMS:
        got=integrate(model_for(root,arm),source,32)
        np.testing.assert_array_equal(got,np.load(parent(root)/'output/runs'/arm/'generated.npy'))
        print(json.dumps(dict(phase='native32_replay',family=arm,images=len(got),seconds=time.monotonic()-start)),flush=True)
    verify(root);write(root/'preflight.json',dict(status='PASS',fixtures=fixtures,all4096_native32_bitwise=True,
        no_optimizer_updates=True,seconds=time.monotonic()-start))

def cell(root,arm):
    torch.set_num_threads(1);verify(root);assert load(root/'preflight.json')['status']=='PASS'
    folder=root/'output/runs'/arm;folder.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    model=model_for(root,arm);source=np.load(parent(root)/'banks/sampling-source.npy')
    for n in (64,128):
        def progress(count):write(folder/'status.json',dict(state='RUNNING',family=arm,N=n,processed=count,total=1024,seconds=time.monotonic()-start))
        got=integrate(model,source,n,progress);np.save(folder/f'generated-{n}.npy',got)
    write(folder/'status.json',dict(state='COMPLETE',family=arm,seconds=time.monotonic()-start))

def run(root):
    assert load(root/'preflight.json')['status']=='PASS';verify(root);out=root/'output';out.mkdir(exist_ok=False)
    waiting=list(ARMS);active={};launches=[];done=[];started=time.monotonic()
    while waiting or active:
        while waiting and len(active)<2:
            arm=waiting.pop(0);log=(out/f'{arm}.log').open('w');cmd=[str(TRAIN),str(root/'source/study.py'),'cell','--root',str(root),'--arm',arm]
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);log.close()
            record=dict(family=arm,pid=proc.pid,command=cmd,start=time.time(),create_time=psutil.Process(proc.pid).create_time());launches.append(record)
            active[arm]=(proc,time.monotonic());write(out/'launches.json',launches)
        resources=[]
        for arm,(proc,t0) in list(active.items()):
            code=proc.poll()
            if code is not None:
                done.append(dict(family=arm,exit_code=code,seconds=time.monotonic()-t0));del active[arm]
                write(out/'exits.json',done);assert code==0,done[-1]
            else:
                assert time.monotonic()-t0<1200,(arm,'registered wall-time ceiling')
                try:
                    p=psutil.Process(proc.pid);resources.append(dict(family=arm,pid=proc.pid,cpu_seconds=sum(p.cpu_times()[:2]),rss=p.memory_info().rss))
                except psutil.NoSuchProcess:pass
        append(out/'resources.jsonl',dict(elapsed=time.monotonic()-started,processes=resources))
        write(out/'status.json',dict(state='RUNNING' if active or waiting else 'SAMPLING_COMPLETE',completed=len(done),total=4,seconds=time.monotonic()-started))
        if active:time.sleep(5)
    verify(root);print(json.dumps(load(out/'status.json')),flush=True)

def evaluate(root):
    torch.set_num_threads(1);start=time.monotonic();verify(root);assert load(root/'output/status.json')['state']=='SAMPLING_COMPLETE'
    p=parent(root);sys.path.insert(0,str(p/'source'));import image_metrics as m
    from metric_core import fid,uint8_images
    out=root/'metric-output';out.mkdir();extractor=m.extractor(p);reference=np.load(p/'metric-inputs/reference-features.npy');rows=[]
    for arm in ARMS:
        for n in GRIDS:
            if n==32:
                raw=np.load(p/'output/runs'/arm/'generated.npy');f=np.load(p/'metric-output'/f'{arm}-features.npy')
            else:
                raw=np.load(root/'output/runs'/arm/f'generated-{n}.npy');f=m.features(extractor,uint8_images(raw),progress=f'{arm}-N{n}')
                np.save(out/f'{arm}-{n}-features.npy',f)
            score=fid(f,reference);a=f.astype(np.float64);b=reference.astype(np.float64)
            mean=float(np.sum((a.mean(0)-b.mean(0))**2));rows.append(dict(family=arm,N=n,fid=score,mean_term=mean,covariance_term=score-mean,
                out_of_range_fraction=float(np.mean(np.abs(raw)>1)),sample_count=len(raw),reference_count=len(reference)))
            write(out/'progress.json',dict(completed=len(rows),total=12,seconds=time.monotonic()-start))
    verify(root);write(out/'result.json',dict(status='PASS',rows=rows,seconds=time.monotonic()-start))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('prepare','preflight','cell','run','evaluate'));ap.add_argument('--root',type=Path,required=True);ap.add_argument('--parent',type=Path);ap.add_argument('--arm',choices=ARMS);a=ap.parse_args()
    try:
        if a.action=='prepare':prepare(a.root,a.parent)
        elif a.action=='cell':cell(a.root,a.arm)
        else:globals()[a.action](a.root)
    except BaseException:
        traceback.print_exc();raise
