"""Immutable native image models evaluated on shared paired/generated states."""
from __future__ import annotations
import os,sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','PYTHONDONTWRITEBYTECODE'):os.environ[k]='1'
sys.dont_write_bytecode=True
import argparse,hashlib,importlib.metadata,importlib.util,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from bank_posterior import posterior,fixtures
if __name__=='__main__':sys.modules['state_risk_study']=sys.modules[__name__]

BASE=Path('/Users/mr.z/ode-workspace-prep/autoresearch')
NATIVE=BASE/'oct28-cycle17-v2';BUDGET=BASE/'oct28-image-budget-v2';COLD=BASE/'oct28-image-clock-training-v2';GATES=BASE/'oct28-image-gate-feedback'
ARMS=('ode_k1','fixed06','matched_silu64','swiglu86','silu131')
DOMAINS=('paired_validation','paired_training','generated_K1','generated_Swi86');INDICES=(13,38,64,90,115)
PYTHON='/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python'
SPEC=Path('/Users/mr.z/ODE-FFN-autoresearch/docs/autoresearch/cycle17-cifar-baselines/training-env-spec.json')
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def module(name,path):
    path=Path(path).resolve()
    if name not in sys.modules:
        s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m)
    assert Path(sys.modules[name].__file__).resolve()==path;return sys.modules[name]
def setup():
    spec=load(SPEC);assert hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]=='475428fb'
    for phase in spec['pip_phases']:
        for req in phase:
            name,version=req.split('==');assert importlib.metadata.version(name)==version
    sys.path.insert(0,str(COLD/'source'));cold=module('training_controls',COLD/'source/training_controls.py')
    native=cold.old();assert Path(native.__file__).resolve()==NATIVE/'source/study.py'
    budget=cold.imported('budget');assert Path(budget.__file__).resolve()==BUDGET/'source/study.py'
    torch.set_num_threads(1);return native,cold,budget
def model_for(arm):
    native,cold,budget=setup()
    if arm in ('fixed06','matched_silu64'):return cold.sample_model(COLD,arm,20000)
    model=budget.construct(arm);state=torch.load(BUDGET/'output/runs'/arm/'step_0020000.pt',weights_only=False,map_location='cpu')
    assert state['step']==20000;native.restore_model(model,state);return model.eval().requires_grad_(False)
def folder_for(arm):return (COLD if arm in ('fixed06','matched_silu64') else BUDGET)/'output/runs'/arm
def verify(root):
    for name,digest in load(root/'bindings.json').items():assert sha(name)==digest,name
    for parent in (NATIVE,BUDGET,COLD,GATES,BASE/'oct28-invariant-clock-v1'):
        for name,digest in load(parent/'sealed-runtime.json').items():assert sha(parent/name)==digest,(parent,name)
def train_bank():
    n,_,_=setup();images,ids=n.data_for(NATIVE)
    assert len(ids)==45000 and len(torch.unique(ids))==45000
    data=Path(load(NATIVE/'settings.json')['data'])
    with np.load(data/'split.npz') as f:assert not np.intersect1d(ids.numpy(),f['validation']).size
    return (images[ids].float()/127.5-1).reshape(45000,-1).double()
def prepare(root):
    setup();repo=Path(__file__).resolve().parents[3];assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    assert not root.exists();root.mkdir();shutil.copytree(Path(__file__).parent,root/'source');(root/'doc-validation').mkdir()
    n,_,_=setup();images,ids=n.data_for(NATIVE);train=(images[ids[:64]].float()/127.5-1).numpy()
    with np.load(NATIVE/'banks/validation.npz') as f:validation=f['target'][:64].copy()
    starts=np.load(NATIVE/'banks/sampling-source.npy')[:64];states={}
    for j in INDICES:
        t=np.float32(j/128)
        states[f'paired_validation_{j}']=t*validation+(np.float32(1)-t)*starts
        states[f'paired_training_{j}']=t*train+(np.float32(1)-t)*starts
    for arm,domain in (('ode_k1','generated_K1'),('swiglu86','generated_Swi86')):
        with np.load(GATES/'output'/f'{arm}-20000/inputs.npz') as f:
            for j in INDICES:
                states[f'{domain}_{j}']=f[f'generated-t{j:03d}']
                assert states[f'paired_validation_{j}'].tobytes()==f[f'paired-t{j:03d}'].tobytes()
    np.savez(root/'states.npz',**states);np.savez(root/'targets.npz',paired_validation=validation,paired_training=train,training_indices=ids[:64].numpy())
    write(root/'settings.json',dict(source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),arms=ARMS,domains=DOMAINS,indices=INDICES,env_spec='475428fb',new_training=False,new_FID=False))
    data=Path(load(NATIVE/'settings.json')['data']);paths=[*Path(root/'source').rglob('*'),root/'states.npz',root/'targets.npz',root/'settings.json',SPEC,data/'images.uint8.npy',data/'split.npz',data/'manifest.json']
    for parent in (NATIVE,BUDGET,COLD,GATES):paths += [parent/'sealed-runtime.json',*Path(parent/'source').rglob('*.py')]
    for arm in ARMS:paths += [folder_for(arm)/'step_0020000.pt',folder_for(arm)/'validation_0020000.npz']
    write(root/'bindings.json',{str(p.resolve()):sha(p) for p in sorted(set(paths)) if p.is_file() and p.suffix!='.pyc'});verify(root)
    print('STATE_RISK_PREPARED',flush=True)
def preflight(root):
    setup();verify(root);assert sha(__file__)==sha(root/'source/study.py');math=fixtures();n,_,_=setup();b=n.bank(NATIVE/'banks/validation.npz');rows=[]
    b={k:v[:64] for k,v in b.items()}
    for arm in ARMS:
        model=model_for(arm);p=n.predict(model,b)
        with np.load(folder_for(arm)/'validation_0020000.npz') as f:assert p.tobytes()==f['prediction'][:64].tobytes(),arm
        rows.append(dict(arm=arm,native_prediction_replayed_bitwise=64))
    bank=train_bank();assert bank.shape==(45000,3072) and bank.isfinite().all()
    with np.load(root/'states.npz') as f:assert len(f.files)==20 and all(f[k].shape==(64,3,32,32) for k in f.files)
    verify(root);write(root/'preflight.json',dict(status='PASS',models=rows,mathematics=math,bank_size=len(bank),env_spec='475428fb',formal_measurement_started=False))
    print('STATE_RISK_NATIVE_PREFLIGHT_PASS',flush=True)
def fingerprint(model):
    h=hashlib.sha256()
    for name,v in (*model.named_parameters(),*model.named_buffers()):h.update(name.encode());h.update(v.detach().numpy().tobytes())
    return h.hexdigest()
def metrics(pred,reference,t,target=None):
    a=pred.reshape(len(pred),-1).astype(np.float64);error=np.mean((a-reference['mean'])**2,axis=1)
    result=dict(bank_reference_clean_mse=error,bank_reference_velocity_mse=error/(1-t)**2)
    if target is not None:result['paired_target_mse']=np.mean((a-target.reshape(len(target),-1).astype(np.float64))**2,axis=1)
    return result
def measure(root):
    setup();verify(root);out=root/'output';out.mkdir();(out/'predictions').mkdir();(out/'posterior').mkdir();(out/'per-image').mkdir();started=time.monotonic()
    bank=train_bank();models={a:model_for(a) for a in ARMS};before={a:fingerprint(m) for a,m in models.items()};rows=[];refrows=[]
    with np.load(root/'states.npz') as states,np.load(root/'targets.npz') as targets:
        for domain in DOMAINS:
            for j in INDICES:
                assert time.monotonic()-started<1200
                key=f'{domain}_{j}';x=states[key];t=j/128;ref=posterior(bank,torch.from_numpy(x.reshape(64,-1)).double(),t)
                np.savez_compressed(out/'posterior'/f'{key}.npz',**ref)
                rr=dict(domain=domain,index=j,time=t,effective_support_median=float(np.median(ref['effective_support'])),entropy_mean=float(np.mean(ref['entropy'])),top_weight_median=float(np.median(ref['top_weight'])),log_density_per_dimension_mean=float(np.mean(ref['log_density'])/3072),paired_target_mse=None)
                target=targets[domain] if domain.startswith('paired_') else None
                if target is not None:rr['paired_target_mse']=float(np.mean((ref['mean']-target.reshape(64,-1).astype(np.float64))**2))
                refrows.append(rr)
                for arm,model in models.items():
                    with torch.no_grad():prediction=model(torch.from_numpy(x),torch.full((64,),t),torch.zeros(64,dtype=torch.long)).numpy()
                    np.save(out/'predictions'/f'{arm}_{key}.npy',prediction);per_image=metrics(prediction,ref,t,target);np.savez(out/'per-image'/f'{arm}_{key}.npz',**per_image)
                    rows.append(dict(arm=arm,domain=domain,index=j,time=t,**{k:float(np.mean(v)) for k,v in per_image.items()}))
                write(out/'progress.json',dict(state='RUNNING',batches=len(refrows),total=20,model_batches=len(rows),seconds=time.monotonic()-started))
    for arm,model in models.items():assert fingerprint(model)==before[arm]
    verify(root);write(out/'result.json',dict(status='COMPLETE',rows=rows,reference_rows=refrows,model_tensors_unchanged=True,seconds=time.monotonic()-started));print('STATE_RISK_MEASUREMENT_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','preflight','measure','audit','run'));p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    if a.action=='audit':
        from state_risk_audit import audit;audit(root)
    elif a.action=='run':
        from controller import run;run(root)
    else:globals()[a.action](root)
