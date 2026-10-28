"""Read-only native-state capture and complete FFN differential analysis."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,copy,hashlib,importlib.util,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import psutil
import torch
import feedback_math as math

BASE=Path('/Users/mr.z/ode-workspace-prep/autoresearch')
PARENTS={key:BASE/name for key,name in dict(budget='oct28-image-budget-v2',clocks20k='oct28-image-clocks20k',
    native='oct28-cycle17-v2',resolution='oct28-cycle18',ordinary='oct28-cycle20').items()}
SEALS=dict(budget='ac381ff444384e052c4cc48e9f11550f00ab410a233154972177e97fe543a48b',
    clocks20k='825980fab2105dce825380ddfa0c349b5a1f179cb68dd7e44bcd5510916a6f9b',
    native='62d178c4e2b88780495c9ea45c6aeef96453ce7098a8aeee84076cf58f081172',
    resolution='0a6fb24d1aba029be90e95f7c74919bbbdead973c101819852f04cb55cc64108',
    ordinary='d7fe17e92c7bd8468b6c048aa9925a4350d95168ed611682bb9c7d787634e973')
FAMILIES=('ode_k1','ode_k2','swiglu86','silu131')
CELLS=tuple(f'{arm}-{step}' for step in (5000,20000) for arm in FAMILIES)
INDICES=(13,38,64,90,115);TOKENS=(0,21,42,63)
TRAIN='/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python'
load=lambda p:json.loads(Path(p).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def imported(key):
    name='feedback_native_parent_'+key;path=PARENTS[key]/'source/study.py'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    assert Path(sys.modules[name].__file__).resolve()==path
    return sys.modules[name]
def verify(root):
    for path,digest in load(root/'bindings.json').items():assert sha(path)==digest,path
    for key,parent in PARENTS.items():
        assert sha(parent/'sealed-runtime.json')==SEALS[key],key
        for name,digest in load(parent/'sealed-runtime.json').items():assert sha(parent/name)==digest,(key,name)
def split(cell):
    arm,step=cell.rsplit('-',1);assert cell in CELLS;return arm,int(step)
def native_model(cell):
    arm,step=split(cell);budget=imported('budget');model=budget.construct(arm)
    state=torch.load(PARENTS['budget']/'output/runs'/arm/f'step_{step:07d}.pt',weights_only=False,map_location='cpu')
    assert state['step']==step;budget.imported('native').restore_model(model,state)
    return model.eval().requires_grad_(False)
def endpoint(cell):
    arm,step=split(cell)
    if step==20000:return PARENTS['budget']/'output/runs'/arm/'generated-0020000.npy'
    if arm=='silu131':return PARENTS['ordinary']/'output/runs'/arm/'generated.npy'
    return PARENTS['resolution']/'output/runs'/arm/'generated-128.npy'
def starts():return np.load(PARENTS['native']/'banks/sampling-source.npy')[:64].copy()
def paired():
    with np.load(PARENTS['native']/'banks/validation.npz') as f:target=f['target'][:64].copy()
    e=starts();answer={}
    for j in INDICES:
        t=np.float32(j/128);answer[j]=t*target+(np.float32(1)-t)*e
        assert answer[j].dtype==np.float32
    return answer
def tensors(model):
    return {**{f'p:{k}':v.detach().clone() for k,v in model.named_parameters()},
        **{f'b:{k}':v.detach().clone() for k,v in model.named_buffers()}}
def unchanged(model,before):
    after=tensors(model);assert after.keys()==before.keys()
    for name in before:assert torch.equal(before[name],after[name]),name
def select(x):
    assert x.shape==(64,64,64)
    return x.detach()[:,TOKENS,:].reshape(256,64).clone().numpy()
def capture(cell,manual=False):
    """Formal uses parent integrator; audit independently spells the Heun loop."""
    model=native_model(cell);before=tensors(model);records={};inputs={};state=dict(active=None,calls=0);hooks=[]
    def start(m,args):
        call=state['calls'];state['calls']+=1
        j=call//2;state['active']=('generated',j) if call%2==0 and j in INDICES else None
        if state['active']:
            assert torch.equal(args[1],torch.full((64,),j/128))
            inputs[f'generated-t{j:03d}']=args[0].detach().clone().numpy()
    def hook(name,is_local):
        def save(m,args,out):
            active=state['active']
            if active is None:return
            domain,j=active;key=f'{domain}-t{j:03d}-{name}';assert key not in records
            y=out[0] if isinstance(out,tuple) else out
            r=dict(h=select(args[0]),native_y=select(y))
            if is_local:
                with torch.no_grad():s=m.shift+m.scale*torch.sigmoid(m.t_func(args[0])/m.tau)
                assert s.shape==(64,64,1);r['s_all']=s.squeeze(-1).clone().numpy()
            records[key]=r
        return save
    front=model.register_forward_pre_hook(start)
    for block_index,block in enumerate(model.blocks):
        module=block.mlp;hooks.append(module.register_forward_hook(hook(f'block{block_index}-core',False)))
        if hasattr(module,'ode1'):
            for order in (1,2):hooks.append(getattr(module,f'ode{order}').register_forward_hook(hook(f'block{block_index}-ode{order}',True)))
    try:
        if manual:
            x=torch.from_numpy(starts());labels=torch.zeros(64,dtype=torch.long)
            with torch.no_grad():
                for j in range(128):
                    t=torch.full((64,),j/128);nt=torch.full((64,),(j+1)/128)
                    first=model(x,t,labels);a=(first-x)/(1-t[:,None,None,None]).clamp_min(.05)
                    trial=x+a/128;second=model(trial,nt,labels)
                    b=(second-trial)/(1-nt[:,None,None,None]).clamp_min(.05)
                    x=x+(a+b)/256
            generated=x.numpy()
        else:generated=imported('resolution').integrate(model,starts(),128)
        assert state['calls']==256
        np.testing.assert_array_equal(generated,np.load(endpoint(cell))[:64])
        front.remove();front=None
        for j,x in paired().items():
            state['active']=('paired',j);inputs[f'paired-t{j:03d}']=x
            with torch.no_grad():model(torch.from_numpy(x),torch.full((64,),j/128),torch.zeros(64,dtype=torch.long))
        unchanged(model,before)
    finally:
        if front is not None:front.remove()
        for h in hooks:h.remove()
    expected=90 if cell.startswith('ode') else 30;assert len(records)==expected
    assert len(inputs)==10
    return model,records,inputs,generated
def module_for(model,key):
    _,_,block,kind=key.split('-');module=model.blocks[int(block.removeprefix('block'))].mlp
    return module if kind=='core' else getattr(module,kind)
def promoted_witness(model,records):
    errors=[]
    for key,r in records.items():
        if not key.startswith('paired-t064'):continue
        module=copy.deepcopy(module_for(model,key)).double().eval().requires_grad_(False)
        h=torch.from_numpy(r['h'][:2]).double()
        with torch.no_grad():
            actual=module(h);actual=actual[0] if isinstance(actual,tuple) else actual
            expected=math.literal_layer(module,h) if '-ode' in key else math.literal_core(module,h)
        torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
        errors.append(dict(module=key,promoted_literal_error=float((actual-expected).abs().max())))
    return errors
def prepare(root):
    assert not root.exists();repo=Path(__file__).resolve().parents[3]
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    root.mkdir();source=root/'source';source.mkdir()
    for p in Path(__file__).resolve().parent.iterdir():
        if p.is_file():shutil.copyfile(p,source/p.name)
    shutil.copyfile(repo/'DERIVATION_PACKAGE.md',source/'DERIVATION_PACKAGE.md')
    write(root/'settings.json',dict(source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        cells=CELLS,Heun_indices=INDICES,token_indices=TOKENS,images=64,workers=2,threads=1,new_training=False,new_quality_result=False))
    np.savez(root/'paired-inputs.npz',**{f't{j:03d}':v for j,v in paired().items()})
    paths=[*source.iterdir(),root/'settings.json',root/'paired-inputs.npz',*(p/'sealed-runtime.json' for p in PARENTS.values())]
    write(root/'bindings.json',{str(p):sha(p) for p in paths if p.is_file()});verify(root)
    print(json.dumps(dict(status='PREPARED',root=str(root))),flush=True)
def preflight(root):
    torch.set_num_threads(1);verify(root);assert not (root/'preflight.json').exists();rows=[]
    fixture=math.fixtures()
    for cell in CELLS:
        tick=time.monotonic();model,records,_,_=capture(cell)
        witnesses=promoted_witness(model,records)
        rows.append(dict(cell=cell,endpoint64_bitwise=True,native_capture_groups=len(records),promoted=witnesses,seconds=time.monotonic()-tick))
        print(json.dumps(rows[-1]),flush=True)
    verify(root);write(root/'preflight.json',dict(status='PASS',fixtures=fixture,rows=rows,new_quality_result=False))
def cell_run(root,cell):
    torch.set_num_threads(1);verify(root);assert load(root/'documented-pass.json')['status']=='PASS'
    out=root/'output'/cell;out.mkdir(parents=True);tick=time.monotonic()
    model,records,inputs,generated=capture(cell);np.save(out/'generated-prefix.npy',generated);np.savez(out/'inputs.npz',**inputs)
    rows=[]
    for index,(key,r) in enumerate(sorted(records.items())):
        module=copy.deepcopy(module_for(model,key)).double().eval().requires_grad_(False);x=torch.from_numpy(r['h']).double()
        with torch.no_grad():
            result=math.local(module,x) if '-ode' in key else math.core(module,x);spec=math.spectra(result)
        difference=result['y'].numpy()-r['native_y'].astype(np.float64)
        np.savez_compressed(out/f'{key}-capture.npz',**r)
        np.savez_compressed(out/f'{key}-spectra.npz',**spec)
        rows.append(dict(group=key,probes=len(x),native_to_promoted_max=float(np.max(np.abs(difference))),
            native_to_promoted_RMS=float(np.sqrt(np.mean(difference**2)))))
        if index%10==0:write(out/'status.json',dict(state='RUNNING',groups=index+1,total=len(records),seconds=time.monotonic()-tick))
    write(out/'result.json',dict(status='PASS',cell=cell,rows=rows,seconds=time.monotonic()-tick))
    verify(root);write(out/'status.json',dict(state='COMPLETE',groups=len(rows),seconds=time.monotonic()-tick))
    print(json.dumps(dict(cell=cell,status='PASS',groups=len(rows),seconds=time.monotonic()-tick)),flush=True)
def run(root):
    verify(root);assert load(root/'preflight.json')['status']==load(root/'documented-pass.json')['status']=='PASS'
    out=root/'output';out.mkdir();queue=list(CELLS);active={};exits=[];launches=[];tick=time.monotonic()
    try:
        while queue or active:
            while queue and len(active)<2:
                cell=queue.pop(0);command=[TRAIN,'-B',str(root/'source/study.py'),'cell','--root',str(root),'--cell',cell]
                with (out/f'{cell}.log').open('w') as log:p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                monitor=psutil.Process(p.pid);monitor.cpu_percent();active[cell]=(p,monitor,time.monotonic())
                launches.append(dict(cell=cell,pid=p.pid,create_time=monitor.create_time(),command=command));write(out/'launches.json',launches)
            assert time.monotonic()-tick<4000;resources=[]
            for cell,(p,monitor,start) in list(active.items()):
                code=p.poll()
                if code is not None:
                    exits.append(dict(cell=cell,pid=p.pid,exit_code=code,seconds=time.monotonic()-start));del active[cell]
                    write(out/'exits.json',exits);assert code==0,exits[-1]
                else:
                    assert time.monotonic()-start<900
                    resources.append(dict(cell=cell,pid=p.pid,CPU_percent=monitor.cpu_percent(),RSS=monitor.memory_info().rss))
            with (out/'resources.jsonl').open('a') as f:f.write(json.dumps(dict(seconds=time.monotonic()-tick,resources=resources))+'\n')
            write(out/'status.json',dict(state='RUNNING' if active or queue else 'COMPLETE',completed=len(exits),total=8))
            if active:time.sleep(5)
    except BaseException:
        for p,_,_ in active.values():
            if p.poll() is None:p.terminate()
        for p,_,_ in active.values():p.wait(timeout=30)
        write(out/'status.json',dict(state='FAILED',exits=exits));raise
    verify(root)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','preflight','run','cell'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--cell',choices=CELLS);args=p.parse_args()
    if args.action=='cell':cell_run(args.root,args.cell)
    else:globals()[args.action](args.root)
