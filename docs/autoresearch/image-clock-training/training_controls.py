"""Cold constant-clock/ordinary controls on the immutable native image stream."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,copy,csv,hashlib,importlib.util,json,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import psutil
import torch
import clock_operators as ops
if __name__=='__main__':sys.modules['training_controls']=sys.modules[__name__]

BASE=Path('/Users/mr.z/ode-workspace-prep/autoresearch')
PARENTS={key:BASE/name for key,name in dict(native='oct28-cycle17-v2',ordinary='oct28-cycle20',resolution='oct28-cycle18',
    budget='oct28-image-budget-v2',clocks20k='oct28-image-clocks20k',timeclocks='oct28-image-time-clocks-v3').items()}
SEALS=dict(native='62d178c4e2b88780495c9ea45c6aeef96453ce7098a8aeee84076cf58f081172',
    ordinary='d7fe17e92c7bd8468b6c048aa9925a4350d95168ed611682bb9c7d787634e973',
    resolution='0a6fb24d1aba029be90e95f7c74919bbbdead973c101819852f04cb55cc64108',
    budget='ac381ff444384e052c4cc48e9f11550f00ab410a233154972177e97fe543a48b',
    clocks20k='825980fab2105dce825380ddfa0c349b5a1f179cb68dd7e44bcd5510916a6f9b',
    timeclocks='8917402ec1dd0ca04fec024b081bbf7b37f64e563444220592c3ab7604bc17cb')
ARMS=('fixed06','matched_silu64');STEPS=(0,1000,2500,5000,10000,15000,20000);SAMPLE_STEPS=(5000,10000,20000)
SAMPLES=tuple((arm,step) for arm in ARMS for step in SAMPLE_STEPS)+(('native_posthoc06',20000),)
TRAIN='/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python'
METRICS='/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python'
load=lambda p:json.loads(Path(p).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    p=Path(path);temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)
def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
def imported(key):
    name='clock_training_parent_'+key;path=PARENTS[key]/'source/study.py'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    assert Path(sys.modules[name].__file__).resolve()==path
    return sys.modules[name]
def old():return imported('native')
def verify(root):
    for path,digest in load(root/'bindings.json').items():assert sha(path)==digest,path
    for key,parent in PARENTS.items():
        assert sha(parent/'sealed-runtime.json')==SEALS[key],key
        for name,digest in load(parent/'sealed-runtime.json').items():assert sha(parent/name)==digest,(key,name)
def native_state(step):
    parent=PARENTS['native'] if step<=5000 else PARENTS['budget']
    state=torch.load(parent/'output/runs/ode_k1'/f'step_{step:07d}.pt',weights_only=False,map_location='cpu');assert state['step']==step
    return state
def construct(arm,native=None,promote=False,independent=False):
    assert arm in ARMS;model=imported('budget').construct('ode_k1');old().restore_model(model,native_state(0) if native is None else native)
    if promote:
        model.double();model.t_embedder.mlp.register_forward_pre_hook(lambda m,args:(args[0].to(m[0].weight.dtype),))
    model=ops.fixed(model,independent=independent) if arm=='fixed06' else ops.direct(model)
    if independent and arm=='matched_silu64':
        import types
        for block in model.blocks:
            def forward(self,x,cond=None,return_aux=False):
                h=torch.nn.functional.linear(x,self.fc1.weight);out=torch.nn.functional.linear(torch.nn.functional.silu(h),self.fc2.weight)
                aux={'input_norm':x.detach().norm(dim=-1).mean(),'output_norm':out.detach().norm(dim=-1).mean()}
                return (out,aux) if return_aux else out
            block.mlp.forward=types.MethodType(forward,block.mlp)
    return model
def restored(arm,state=None,independent=False):
    model=construct(arm,independent=independent);optimizer=old().optimizer(model);rng=torch.Generator()
    if state is None:
        original=native_state(0);rng.set_state(original['data_rng']);torch.set_rng_state(original['torch_rng']);digest=original['input_stream_digest']
    else:digest=old().restore(model,optimizer,rng,copy.deepcopy(state))
    return model,optimizer,rng,digest
def bank(t=None):
    return old().bank(PARENTS['native']/'banks'/('validation.npz' if t is None else f'panel_{t:.1f}.npz'))
def is_core_parameter(name):return name.startswith('blocks.') and '.mlp.' in name
def matrices(model,arm):
    values={}
    for j,block in enumerate(model.blocks):
        for k in (1,2):
            if arm=='fixed06':p=getattr(block.mlp,f'ode{k}').proj.weight.detach();w=torch.eye(64,dtype=p.dtype)+ops.C*p
            else:w=getattr(block.mlp,f'fc{k}').weight.detach();p=(w-torch.eye(64,dtype=w.dtype))/ops.C
            values[f'block{j}.P{k}' if arm=='fixed06' else f'block{j}.inferred_P{k}']=p.numpy().copy()
            values[f'block{j}.W{k}']=w.numpy().copy()
    return values
def history(model,arm,folder,step):
    values=matrices(model,arm);np.savez(folder/f'matrices-{step:07d}.npz',**values);rows=[]
    for name,value in values.items():
        a=value.astype(np.float64);singular=np.linalg.svd(a,compute_uv=False);eigen=np.linalg.eigvals(a)
        rows.append(dict(layer=name,coordinate='trained_P' if '.P' in name else 'inferred_P' if 'inferred_P' in name else 'effective_W',
            singular_values=singular.tolist(),sigma_max=float(singular[0]),sigma_min=float(singular[-1]),singular_CV=float(singular.std()/singular.mean()),
            frobenius=float(np.linalg.norm(a)),eigen_real=eigen.real.tolist(),eigen_imag=eigen.imag.tolist()))
    write(folder/f'matrix-history-{step:07d}.json',rows)
def shared_initial_witness():
    initial=native_state(0);a=construct('fixed06');b=construct('matched_silu64');state_a=a.state_dict();state_b=b.state_dict()
    for name,value in initial['model'].items():
        if not is_core_parameter(name):assert torch.equal(value,state_a[name]) and torch.equal(value,state_b[name]),name
    for name,value in state_a.items():assert torch.equal(value,initial['model'][name])
    for name,value in initial['runtime_buffers'].items():
        assert torch.equal(dict(a.named_buffers())[name],value) and torch.equal(dict(b.named_buffers())[name],value)
    gen=torch.Generator().manual_seed(0);x=torch.randn(8,4,64,generator=gen);maximum=0.
    with torch.no_grad():
        for left,right in zip(a.blocks,b.blocks):
            y=left.mlp(x,None);z=right.mlp(x,None);torch.testing.assert_close(y,z,rtol=1e-5,atol=1e-5);maximum=max(maximum,float((y-z).abs().max()))
    return dict(shared_parameters_buffers_bitwise=True,initial_P_native_bitwise=True,float32_core_difference_max=maximum)
def active_witness():
    state=native_state(20000);a=construct('fixed06',state,promote=True).eval();b=construct('matched_silu64',state,promote=True).eval()
    inputs=bank();x=torch.from_numpy(inputs['x'][:2]).double();t=torch.from_numpy(inputs['t'][:2]).double();y=torch.zeros(2,dtype=torch.long)
    target=torch.from_numpy(inputs['target'][:2]).double();delta=torch.from_numpy(inputs['delta'][:2]).double()[:,None,None,None]
    one=a(x,t,y);two=b(x,t,y);torch.testing.assert_close(one,two,rtol=1e-9,atol=1e-9)
    ((one-target)/delta).square().mean().backward();((two-target)/delta).square().mean().backward()
    pa=dict(a.named_parameters());pb=dict(b.named_parameters());error=0.;active=0;shared=[];timestep=[]
    for name,v in pa.items():
        if not is_core_parameter(name) and v.requires_grad:
            shared.append(name)
            if name.startswith('t_embedder.mlp.'):timestep.append(name)
            if v.grad is None:assert pb[name].grad is None
            else:
                torch.testing.assert_close(v.grad,pb[name].grad,rtol=1e-8,atol=1e-10);error=max(error,float((v.grad-pb[name].grad).abs().max()))
    for j in range(3):
        for k in (1,2):
            p=pa[f'blocks.{j}.mlp.ode{k}.proj.weight'];w=pb[f'blocks.{j}.mlp.fc{k}.weight']
            assert p.grad.norm()>1e-10;active+=1;torch.testing.assert_close(p.grad,ops.C*w.grad,rtol=1e-8,atol=1e-10)
            error=max(error,float((p.grad-ops.C*w.grad).abs().max()))
    assert len(timestep)==4
    return dict(promoted_active_prediction_max_error=float((one-two).detach().abs().max()),parameter_gradient_max_error=error,
        nonzero_P_routes=active,shared_gradient_tensors_checked=len(shared),timestep_MLP_parameters_checked=timestep)
def cost_witness():
    from torch.utils._python_dispatch import TorchDispatchMode
    class Count(TorchDispatchMode):
        def __init__(self):self.macs=0
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            result=func(*args,**(kwargs or {}))
            if func==torch.ops.aten.mm.default:self.macs+=args[0].shape[0]*args[0].shape[1]*args[1].shape[1]
            elif func==torch.ops.aten.bmm.default:self.macs+=args[0].shape[0]*args[0].shape[1]*args[0].shape[2]*args[1].shape[2]
            elif func==torch.ops.aten.addmm.default:self.macs+=args[1].shape[0]*args[1].shape[1]*args[2].shape[1]
            return result
    rows=[]
    for arm in ('native',*ARMS):
        model=imported('budget').construct('ode_k1') if arm=='native' else construct(arm)
        core=model.blocks[0].mlp;x=torch.randn(2,4,64,requires_grad=True)
        with Count() as forward:core(x,None)
        with Count() as backward:core(x,None).square().sum().backward()
        fw=forward.macs//8;bw=backward.macs//8
        assert (fw,bw)==((16512,49536) if arm=='native' else (8192,24576)),(arm,fw,bw)
        fixed_count=sum(p.numel() for n,p in model.named_parameters() if n=='pos_embed');active=sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_count=sum(p.numel() for p in model.parameters());assert fixed_count==4096
        rows.append(dict(arm=arm,active_learned_parameters=active,stored_parameter_values=all_count,fixed_position_values=fixed_count,
            FFN_forward_MACs=fw,FFN_forward_backward_MACs=bw))
    assert rows[1]['active_learned_parameters']==rows[2]['active_learned_parameters']==185744
    return rows
def prepare(root):
    assert not root.exists();repo=Path(__file__).resolve().parents[3];assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    root.mkdir();(root/'source').mkdir()
    for p in Path(__file__).resolve().parent.iterdir():
        if p.is_file():shutil.copyfile(p,root/'source'/p.name)
    write(root/'settings.json',dict(source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),arms=ARMS,
        steps=STEPS,sampling=SAMPLES,model_seed=0,constant=ops.C,new_architecture=False,workers=2,threads=1,training_steps=20000))
    paths=[*(root/'source').iterdir(),root/'settings.json',*(p/'sealed-runtime.json' for p in PARENTS.values())];preserved={}
    previous=BASE/'oct28-image-clock-training'
    if root!=previous and previous.exists():
        assert load(previous/'documented-pass.json')['status']=='PASS' and not (previous/'output').exists()
        preserved={str(p):sha(p) for p in previous.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
        write(root/'superseded-preflight-preservation.json',preserved);paths.append(root/'superseded-preflight-preservation.json')
    write(root/'bindings.json',{**{str(p):sha(p) for p in paths if p.is_file()},**preserved});verify(root)
    print(json.dumps(dict(status='PREPARED',root=str(root))),flush=True)
def preflight(root):
    torch.set_num_threads(1);verify(root);assert not (root/'preflight.json').exists()
    fixture=ops.fixtures();rules=decision_fixture();common=shared_initial_witness();active=active_witness();costs=cost_witness();replay=[]
    for arm in ('ode_k1','swiglu86','silu64','silu131'):
        model=imported('budget').construct(arm);state=torch.load(PARENTS['budget']/'output/runs'/arm/'step_0020000.pt',weights_only=False,map_location='cpu')
        old().restore_model(model,state);model.eval();prediction=old().predict(model,bank())
        with np.load(PARENTS['budget']/'output/runs'/arm/'validation_0020000.npz') as f:assert prediction.tobytes()==f['prediction'].tobytes()
        replay.append(dict(arm=arm,predictions_bitwise=1024))
    audit_path=Path(__file__).resolve().parent/'audit_training.py';audit_name='clock_training_independent_witness'
    spec=importlib.util.spec_from_file_location(audit_name,audit_path);audit=importlib.util.module_from_spec(spec);sys.modules[audit_name]=audit;spec.loader.exec_module(audit)
    assert Path(audit.study.__file__).resolve()==Path(__file__).resolve()
    images,ids=old().data_for(PARENTS['native']);folder=root/'preflight-checkpoints';folder.mkdir();resumes=[]
    for arm in ARMS:
        model,opt,rng,digest=restored(arm)
        for step in range(3):
            if step==2:before_active_update=old().checkpoint(model,opt,rng,2,digest)
            batch=old().draw(images,ids,rng);digest=old().fingerprint(digest,batch);old().update(model,opt,batch)
        expected=old().checkpoint(model,opt,rng,3,digest)
        model,opt,rng,digest=restored(arm,before_active_update,independent=True);model.train()
        batch=old().draw(images,ids,rng);digest=old().fingerprint(digest,batch);audit.update(model,opt,batch)
        old().equal(old().checkpoint(model,opt,rng,3,digest),expected)
        model,opt,rng,digest=restored(arm)
        batch=old().draw(images,ids,rng);digest=old().fingerprint(digest,batch);old().update(model,opt,batch)
        path=folder/f'{arm}-step1.pt';torch.save(old().checkpoint(model,opt,rng,1,digest),path)
        model,opt,rng,digest=restored(arm,torch.load(path,weights_only=False,map_location='cpu'))
        for step in range(2):
            batch=old().draw(images,ids,rng);digest=old().fingerprint(digest,batch);old().update(model,opt,batch)
        actual=old().checkpoint(model,opt,rng,3,digest);old().equal(expected,actual)
        initial=construct(arm).state_dict();changed=[]
        for name,v in actual['model'].items():
            if '.mlp.' in name and ('proj.weight' in name or '.fc' in name):changed.append(not torch.equal(v,initial[name]))
            if '.t_func.' in name:assert torch.equal(v,initial[name])
        assert len(changed)==6 and all(changed)
        resumes.append(dict(arm=arm,physical_resume_bitwise=True,independent_active_update_bitwise=True,active_matrices_changed=6,input_stream_digest=digest))
    assert len({v['input_stream_digest'] for v in resumes})==1
    verify(root);write(root/'preflight.json',dict(status='PASS',fixtures=fixture,decision_fixture=rules,shared_initial=common,active_witness=active,costs=costs,
        native_prediction_replays=replay,resumes=resumes,formal_training_started=False))
    print(json.dumps(load(root/'preflight.json')),flush=True)
def cell(root,arm):
    torch.set_num_threads(1);verify(root);assert load(root/'documented-pass.json')['status']=='PASS'
    folder=root/'output/runs'/arm;folder.mkdir(parents=True);tick=time.monotonic();model,opt,rng,digest=restored(arm)
    images,ids=old().data_for(PARENTS['native']);val=bank();window=[]
    def record(step):
        model.eval();prediction=old().predict(model,val);np.savez_compressed(folder/f'validation_{step:07d}.npz',prediction=prediction)
        row=dict(step=step,**old().metrics(prediction,val))
        if window:row.update(train_loss=float(np.mean([v[0] for v in window])),grad_norm=float(np.mean([v[1] for v in window])),clip_fraction=float(np.mean([v[1]>1 for v in window])))
        append(folder/'metrics.jsonl',row);torch.save(old().checkpoint(model,opt,rng,step,digest),folder/f'step_{step:07d}.pt');history(model,arm,folder,step)
        for t in (.1,.3,.5,.7,.9):np.savez_compressed(folder/f'panel_{step:07d}_{t:.1f}.npz',prediction=old().predict(model,bank(t)))
        model.train();return row
    last=record(0)
    try:
        for step in range(1,20001):
            assert time.monotonic()-tick<7200
            if step in STEPS:torch.save(old().checkpoint(model,opt,rng,step-1,digest),folder/f'before_{step:07d}.pt')
            batch=old().draw(images,ids,rng);digest=old().fingerprint(digest,batch);window.append(old().update(model,opt,batch))
            if step%100==0:write(folder/'status.json',dict(state='RUNNING',pid=os.getpid(),step=step,total=20000,seconds=time.monotonic()-tick))
            if step%1000==0 or step==2500:
                assert digest==native_state(step)['input_stream_digest'],('historical stream diverged',arm,step)
                append(folder/'streams.jsonl',dict(step=step,digest=digest,native_digest_equal=True))
            if step in STEPS:last=record(step);window=[]
        verify(root);write(folder/'result.json',dict(arm=arm,**last,seconds=time.monotonic()-tick,input_stream_digest=digest))
        write(folder/'status.json',dict(state='COMPLETE',step=20000,seconds=time.monotonic()-tick))
    except BaseException as e:write(folder/'status.json',dict(state='FAILED',error=repr(e),step=step));raise
def run(root):
    verify(root);assert load(root/'documented-pass.json')['status']=='PASS';out=root/'output';out.mkdir();active={};records=[];tick=time.monotonic()
    try:
        for arm in ARMS:
            command=[TRAIN,'-B',str(root/'source/training_controls.py'),'cell','--root',str(root),'--arm',arm]
            with (out/f'{arm}.log').open('w') as f:p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
            monitor=psutil.Process(p.pid);monitor.cpu_percent();active[arm]=(p,monitor);records.append(dict(arm=arm,pid=p.pid,create_time=monitor.create_time(),command=command));write(out/'launches.json',records)
        while active:
            assert time.monotonic()-tick<7500;resources=[]
            for arm,(p,monitor) in list(active.items()):
                code=p.poll()
                if code is not None:
                    next(r for r in records if r['arm']==arm).update(exit_code=code);del active[arm];write(out/'launches.json',records);assert code==0,(arm,code)
                else:resources.append(dict(arm=arm,pid=p.pid,CPU_percent=monitor.cpu_percent(),RSS=monitor.memory_info().rss))
            append(out/'resources.jsonl',dict(seconds=time.monotonic()-tick,resources=resources));write(out/'status.json',dict(state='RUNNING' if active else 'COMPLETE',active=list(active),completed=2-len(active),seconds=time.monotonic()-tick))
            if active:time.sleep(5)
    except BaseException:
        for p,_ in active.values():
            if p.poll() is None:p.terminate()
        for p,_ in active.values():p.wait(timeout=30)
        write(out/'status.json',dict(state='FAILED',records=records));raise
    verify(root)
def sample_model(root,arm,step,independent=False):
    if arm=='native_posthoc06':return construct('fixed06',native_state(20000),independent=independent).eval().requires_grad_(False)
    state=torch.load(root/'output/runs'/arm/f'step_{step:07d}.pt',weights_only=False,map_location='cpu')
    model=construct(arm,independent=independent);old().restore_model(model,state);return model.eval().requires_grad_(False)
def sample(root):
    torch.set_num_threads(1);verify(root);assert load(root/'audit-training/result.json')['status']=='PASS';out=root/'sampling';out.mkdir();tick=time.monotonic()
    # One worker avoids parallel model/data duplication; sampling has no training data.
    rows=[]
    for arm,step in SAMPLES:
        assert time.monotonic()-tick<3600;start=time.monotonic();model=sample_model(root,arm,step)
        if arm=='native_posthoc06':
            b=bank();pred=old().predict(model,b);np.savez_compressed(out/'posthoc-validation.npz',prediction=pred)
            metrics=old().metrics(pred,b)
            for t in (.1,.3,.5,.7,.9):np.savez_compressed(out/f'posthoc-panel-{t:.1f}.npz',prediction=old().predict(model,bank(t)))
        else:metrics=next(r for r in [json.loads(v) for v in (root/'output/runs'/arm/'metrics.jsonl').read_text().splitlines()] if r['step']==step)
        def progress(count):write(out/'progress.json',dict(completed=len(rows),total=7,arm=arm,step=step,images=count,seconds=time.monotonic()-tick))
        generated=imported('resolution').integrate(model,np.load(PARENTS['native']/'banks/sampling-source.npy'),128,progress)
        assert time.monotonic()-start<900,('sampling cell ceiling',arm,step)
        np.save(out/f'{arm}-{step:07d}.npy',generated)
        rows.append(dict(arm=arm,step=step,weighted_mse=metrics['weighted_mse'],clean_mse=metrics['clean_mse'],
            out_of_range_fraction=float(np.mean(np.abs(generated)>1)),sampling_seconds=time.monotonic()-start))
        write(out/'progress.json',dict(completed=len(rows),total=7,seconds=time.monotonic()-tick));print(json.dumps(rows[-1]),flush=True)
    verify(root);write(out/'result.json',dict(status='PASS',rows=rows))
def cached_rows():
    rows=[]
    for r in csv.DictReader((PARENTS['budget']/'curation/comparison.csv').open()):
        row=dict(arm=r['arm'],step=int(r['step']),**{k:float(r[k]) for k in ('weighted_mse','clean_mse','fid')},
            parameters=int(r['parameters']),FFN_forward_MACs=int(r['FFN_forward_MACs']),provenance=r['provenance']);rows.append(row)
    for r in load(PARENTS['timeclocks']/'metric-output/result.json')['rows']:
        if r['family'] in ('ode_k1_constant','ode_k2_constant','ode_k1_time','ode_k1_pooled'):
            rows.append(dict(arm=r['family'],step=20000,**{k:r[k] for k in ('weighted_mse','clean_mse','fid','parameters','FFN_forward_MACs')},provenance='sealed earlier frozen intervention'))
    assert len(rows)==25;return rows
def feature_path(root,arm,step):
    if arm in ARMS or arm=='native_posthoc06':return root/'metric-output'/f'{arm}-{step:07d}-features.npy'
    if arm in ('ode_k1_constant','ode_k2_constant','ode_k1_time','ode_k1_pooled'):return imported('timeclocks').feature_bank(PARENTS['timeclocks'],arm)
    if step==5000:
        return PARENTS['ordinary']/'metric-output'/f'{arm}-features.npy' if arm.startswith('silu') else PARENTS['resolution']/'metric-output'/f'{arm}-128-features.npy'
    return PARENTS['budget']/'metric-output'/f'{arm}-{step:07d}-features.npy'
def decisions(rows):
    by={(r['arm'],r['step']):r for r in rows};results=[]
    for arm in ARMS:
        x=by[(arm,20000)];refs=[by[(c,20000)] for c in ('swiglu86','silu131')]
        results.append(dict(arm=arm,strict_ordering=all(x['fid']<r['fid'] for r in refs),
            allocation_margin=all(x['fid']<=.95*r['fid'] for r in refs) and all(x[k]<=1.05*min(r[k] for r in refs) for k in ('weighted_mse','clean_mse')),
            SiLU64_strict_ordering=x['fid']<by[('silu64',20000)]['fid'],
            native_deltas={k:x[k]-by[('ode_k1',20000)][k] for k in ('fid','weighted_mse','clean_mse')}))
    return dict(rows=results,primary_step=20000,ordinary_controls_only=True,new_ODE_method=False,posthoc_reference_nominated=False)
def decision_fixture():
    rows=[dict(arm=a,step=20000,fid=v,weighted_mse=1.,clean_mse=1.) for a,v in
        [('ode_k1',90.),('swiglu86',120.),('silu131',100.),('silu64',110.),('fixed06',95.),('matched_silu64',96.)]]
    rows[-2]['weighted_mse']=1.05;assert decisions(rows)['rows'][0]['allocation_margin']
    assert decisions(rows)['rows'][1]['strict_ordering'] and not decisions(rows)['rows'][1]['allocation_margin']
    x=copy.deepcopy(rows);x[-2]['fid']=100.;assert not decisions(x)['rows'][0]['strict_ordering']
    x=copy.deepcopy(rows);x[-2]['weighted_mse']=1.05001;assert not decisions(x)['rows'][0]['allocation_margin']
    x=copy.deepcopy(rows);x.append(dict(arm='matched_silu64',step=10000,fid=10.,weighted_mse=.1,clean_mse=.1))
    assert not decisions(x)['rows'][1]['allocation_margin']
    return dict(status='PASS',cases=5,primary_step=20000)
def metric_preflight(root):
    torch.set_num_threads(1);verify(root);sys.path.insert(0,str(PARENTS['native']/'source'));import image_metrics as m
    from metric_core import uint8_images,fid
    raw=np.load(PARENTS['budget']/'output/runs/ode_k1/generated-0020000.npy');ex=m.extractor(PARENTS['native'])
    actual=m.features(ex,uint8_images(raw[:32]));assert actual.tobytes()==np.load(feature_path(root,'ode_k1',20000))[:32].tobytes()
    value=fid(np.load(feature_path(root,'ode_k1',20000)),np.load(PARENTS['native']/'metric-inputs/reference-features.npy'))
    assert abs(value-next(r['fid'] for r in cached_rows() if r['arm']=='ode_k1' and r['step']==20000))<1e-8
    for row in cached_rows():
        cached=np.load(feature_path(root,row['arm'],row['step']),mmap_mode='r');assert cached.shape==(1024,2048) and cached.dtype==np.float32
    write(root/'metric-preflight.json',dict(status='PASS',historical_feature_rows_bitwise=32,historical_native_FID=value,cached_feature_headers=25));print(json.dumps(load(root/'metric-preflight.json')),flush=True)
def evaluate(root):
    torch.set_num_threads(1);verify(root);assert load(root/'audit-sampling/result.json')['status']=='PASS'
    sys.path.insert(0,str(PARENTS['native']/'source'));import image_metrics as m
    from metric_core import uint8_images,fid
    out=root/'metric-output';out.mkdir();rows=cached_rows();reference=np.load(PARENTS['native']/'metric-inputs/reference-features.npy');extractor=m.extractor(PARENTS['native'])
    for r in rows:assert abs(fid(np.load(feature_path(root,r['arm'],r['step'])),reference)-r['fid'])<1e-8
    for r in load(root/'sampling/result.json')['rows']:
        arm,step=r['arm'],r['step'];feature=m.features(extractor,uint8_images(np.load(root/'sampling'/f'{arm}-{step:07d}.npy')),progress=f'{arm}-{step}')
        np.save(feature_path(root,arm,step),feature)
        row=dict(r,fid=fid(feature,reference),parameters=185744 if arm=='matched_silu64' else 211094,
            active_learned_parameters=185744,training_active_learned_parameters=211094 if arm=='native_posthoc06' else 185744,
            FFN_forward_MACs=8192,provenance='new cold control' if arm in ARMS else 'frozen native common-clock reference')
        rows.append(row)
    assert len(rows)==32
    for r in rows:
        f=np.load(feature_path(root,r['arm'],r['step'])).astype(np.float64);ref=reference.astype(np.float64)
        r['FID_mean_term']=float(np.sum((f.mean(0)-ref.mean(0))**2));r['FID_covariance_term']=r['fid']-r['FID_mean_term']
    verify(root);write(out/'result.json',dict(status='PASS',rows=rows));write(out/'decision.json',decisions(rows))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','preflight','metric-preflight','cell','run','sample','evaluate','fixtures'))
    p.add_argument('--root',type=Path);p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    if a.action=='fixtures':print(json.dumps(ops.fixtures()))
    elif a.action=='cell':cell(a.root,a.arm)
    else:globals()[a.action.replace('-','_')](a.root)
