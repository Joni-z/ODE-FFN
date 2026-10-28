"""One frozen K1 time profile and its same-calibration pooled control."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,copy,hashlib,importlib.util,json,shutil,subprocess,sys,time,types
from pathlib import Path
import numpy as np
import psutil
import torch

BASE=Path('/Users/mr.z/ode-workspace-prep/autoresearch')
PARENTS={key:BASE/name for key,name in dict(clocks20k='oct28-image-clocks20k',feedback='oct28-image-gate-feedback',
    budget='oct28-image-budget-v2',native='oct28-cycle17-v2',resolution='oct28-cycle18',ordinary='oct28-cycle20').items()}
SEALS=dict(clocks20k='825980fab2105dce825380ddfa0c349b5a1f179cb68dd7e44bcd5510916a6f9b',
    feedback='faeac7fd69f048c0875184af92c6278956bdf478083c75d77bbd83f747aa6345',
    budget='ac381ff444384e052c4cc48e9f11550f00ab410a233154972177e97fe543a48b',
    native='62d178c4e2b88780495c9ea45c6aeef96453ce7098a8aeee84076cf58f081172',
    resolution='0a6fb24d1aba029be90e95f7c74919bbbdead973c101819852f04cb55cc64108',
    ordinary='d7fe17e92c7bd8468b6c048aa9925a4350d95168ed611682bb9c7d787634e973')
ARMS=('ode_k1_time','ode_k1_pooled')
TRAIN='/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python'
METRICS='/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python'
load=lambda p:json.loads(Path(p).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
def imported(key):
    name='time_clock_parent_'+key;path=PARENTS[key]/'source/study.py'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    assert Path(sys.modules[name].__file__).resolve()==path
    return sys.modules[name]
def verify(root,calibration=False):
    for path,digest in load(root/'bindings.json').items():assert sha(path)==digest,path
    for key,parent in PARENTS.items():
        assert sha(parent/'sealed-runtime.json')==SEALS[key]
        for name,digest in load(parent/'sealed-runtime.json').items():assert sha(parent/name)==digest,(key,name)
    if calibration:
        for path,digest in load(root/'calibration-bindings.json').items():assert sha(path)==digest,path
def native_model():return imported('clocks20k').native_model('ode_k1')
def old():return imported('clocks20k').old()
def clocks():return imported('clocks20k').clocks()
def bank(t=None):return imported('clocks20k').bank(t)
def calibration_input(j,b=None):
    if b is None:b=bank()
    t=np.float32(j/128)
    return t*b['target'][:64]+(np.float32(1)-t)*b['source'][:64]
def calibrate():
    model=native_model();before=clocks().tensors(model);raw={name:[] for name,_ in clocks().layers(model)};hooks=[];b=bank()
    for name,layer in clocks().layers(model):
        def save(m,args,name=name):
            assert (m.orders,m.tau,m.scale,m.shift,m.t_from_embed)==(1,10.,.8,.2,None)
            s=torch.sigmoid(m.t_func(args[0])/m.tau)*m.scale+m.shift
            assert s.shape==(64,64,1);raw[name].append(s.squeeze(-1).clone().numpy())
        hooks.append(layer.register_forward_pre_hook(save))
    try:
        with torch.no_grad():
            for j in range(129):model(torch.from_numpy(calibration_input(j,b)),torch.full((64,),j/128),torch.zeros(64,dtype=torch.long))
    finally:
        for hook in hooks:hook.remove()
    clocks().unchanged(model,before);raw={k:np.stack(v) for k,v in raw.items()};table={};pooled={}
    for name,values in raw.items():
        assert values.shape==(129,64,64);d=values.astype(np.float64)
        table[name]=d.mean(axis=(1,2)).astype(np.float32);pooled[name]=np.float32(d.mean())
    return raw,table,pooled
def lookup(values,t):
    assert t.ndim==1 and bool(torch.all((t>=0)&(t<=1)))
    u=t*128;a=u.floor().long();b=(a+1).clamp_max(128);r=u-a.to(u.dtype)
    return values[a]+(values[b]-values[a])*r
def patch(model,table,pooled,mode,folded=False):
    assert mode in ('time','pooled');context={}
    def external(m,args):context['t']=args[1]
    model.register_forward_pre_hook(external)
    for name,layer in clocks().layers(model):
        assert layer.orders==1
        values=torch.as_tensor(np.array(table[name],copy=True));constant=float(pooled[name])
        def forward(self,x,t=None,values=values,constant=constant):
            assert t is None
            external_time=context['t'].to(dtype=x.dtype,device=x.device)
            duration=lookup(values.to(dtype=x.dtype,device=x.device),external_time) if mode=='time' else x.new_full((len(x),),constant)
            if folded:
                assert len(x)==1;matrix=torch.eye(x.shape[-1],dtype=x.dtype,device=x.device)+duration[0]*self.proj.weight
                return torch.nn.functional.linear(x,matrix)
            parts=[x];parts.append(self.proj(parts[-1])*duration[:,None,None]/1)
            return sum(parts)
        layer.forward=types.MethodType(forward,layer)
    return model
def calibration(root):
    with np.load(root/'clock-table.npz') as f:table={k:f[k] for k in f.files}
    return table,load(root/'pooled-clocks.json')
def feature_bank(root,family):
    return root/'metric-output'/f'{family}-features.npy' if family in ARMS else imported('clocks20k').feature_bank(PARENTS['clocks20k'],family)
def sample_bank(root,family):
    return root/'output/runs'/family/'generated.npy' if family in ARMS else imported('clocks20k').sample_bank(PARENTS['clocks20k'],family)
def decision(rows):
    by={r['family']:r for r in rows};refs=('swiglu86','silu131');time_row=by[ARMS[0]];pool=by[ARMS[1]];records=[]
    for name in ARMS:
        x=by[name];fit={k:x[k]/min(by[c][k] for c in refs) for k in ('weighted_mse','clean_mse')}
        records.append(dict(family=name,strict_ordering=all(x['fid']<by[c]['fid'] for c in refs),
            allocation_margin=all(x['fid']<=.95*by[c]['fid'] for c in refs) and all(v<=1.05 for v in fit.values()),
            fitting_ratios=fit,SiLU64_strict_ordering=x['fid']<by['silu64']['fid'],
            native_delta={k:x[k]-by['ode_k1'][k] for k in ('fid','weighted_mse','clean_mse')}))
    time_record=records[0]
    specificity=time_row['fid']<=.95*pool['fid'] and all(time_row[k]<=1.05*pool[k] for k in ('weighted_mse','clean_mse'))
    return dict(rows=records,time_over_pooled_criterion=specificity,
        time_specific_explanation_supported=time_record['allocation_margin'] and time_record['SiLU64_strict_ordering'] and specificity,
        time_minus_pooled={k:time_row[k]-pool[k] for k in ('fid','weighted_mse','clean_mse')},
        new_trained_method=False,pooled_control_nominated=False,old_constant_decision_changed=False)
def fixtures():
    torch.manual_seed(0);p=torch.randn(5,5,dtype=torch.float64)*.2;x=torch.randn(5,dtype=torch.float64)
    values=torch.linspace(.25,.9,129,dtype=torch.float64)+.02*torch.sin(torch.arange(129,dtype=torch.float64));rows=[]
    for value in (.123,.789):
        t=torch.tensor([value],dtype=torch.float64,requires_grad=True);q=lookup(values,t)[0];m=torch.eye(5,dtype=torch.float64)+q*p
        f=lambda h:h+lookup(values,t)[0]*(p@h)
        observed=torch.autograd.functional.jacobian(f,x);torch.testing.assert_close(observed,m,rtol=1e-12,atol=1e-12)
        j=int(value*128);expected=128*(values[j+1]-values[j])*(p@x)
        dt=torch.autograd.functional.jacobian(lambda s:x+lookup(values,s)[0]*(p@x),t)[:,0]
        torch.testing.assert_close(dt,expected,rtol=1e-12,atol=1e-12)
        p2=p.T*.7;reverse=values.flip(0);q2=lookup(reverse,t)[0];m2=torch.eye(5,dtype=torch.float64)+q2*p2
        u=m@x;z=torch.nn.functional.silu(u);sig=torch.sigmoid(u);a=sig*(1+u*(1-sig))
        fn=lambda h,tm:(lambda y:y+lookup(reverse,tm)[0]*(p2@y))(torch.nn.functional.silu(h+lookup(values,tm)[0]*(p@h)))
        core_state=torch.autograd.functional.jacobian(lambda h:fn(h,t),x);expected_state=m2@(a[:,None]*m)
        core_time=torch.autograd.functional.jacobian(lambda tm:fn(x,tm),t)[:,0]
        expected_time=128*(reverse[j+1]-reverse[j])*(p2@z)+m2@(a*expected)
        torch.testing.assert_close(core_state,expected_state,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(core_time,expected_time,rtol=1e-12,atol=1e-12)
        rows.append(dict(time=value,state_max_error=float((observed-m).detach().abs().max()),time_max_error=float((dt-expected).detach().abs().max()),
            complete_FFN_state_max_error=float((core_state-expected_state).detach().abs().max()),complete_FFN_time_max_error=float((core_time-expected_time).detach().abs().max())))
    torch.testing.assert_close(lookup(values,torch.arange(129,dtype=torch.float64)/128),values,rtol=0,atol=0)
    example=[dict(family=a,fid=120.,weighted_mse=1.,clean_mse=1.) for a in ('ode_k1','swiglu86','silu131','silu64')]
    example.extend(dict(family=a,fid=v,weighted_mse=1.,clean_mse=1.) for a,v in zip(ARMS,(100.,115.)))
    assert decision(example)['time_specific_explanation_supported']
    x=copy.deepcopy(example);x[-1]['fid']=101.;assert not decision(x)['time_specific_explanation_supported']
    x=copy.deepcopy(example);x[-2]['fid']=120.;assert not decision(x)['rows'][0]['strict_ordering']
    return dict(status='PASS',operator_rows=rows,mesh_entries_exact=129,decision_cases=3)
def prepare(root):
    assert not root.exists();repo=Path(__file__).resolve().parents[3]
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    root.mkdir();(root/'source').mkdir()
    for p in Path(__file__).resolve().parent.iterdir():
        if p.is_file():shutil.copyfile(p,root/'source'/p.name)
    write(root/'settings.json',dict(source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        arms=ARMS,checkpoint_step=20000,calibration_images=64,calendar_nodes=129,model_seed=0,new_training=False,
        generated=1024,reference=2048,steps=128,workers=2,threads=1))
    paths=[*(root/'source').iterdir(),root/'settings.json',*(p/'sealed-runtime.json' for p in PARENTS.values())]
    write(root/'bindings.json',{str(p):sha(p) for p in paths if p.is_file()});verify(root)
    print(json.dumps(dict(status='PREPARED',root=str(root))),flush=True)
def preflight(root):
    torch.set_num_threads(1);verify(root);assert not (root/'preflight.json').exists();fixture=fixtures()
    b=bank();model=native_model();before=clocks().tensors(model);prediction=old().predict(model,b)
    with np.load(PARENTS['budget']/'output/runs/ode_k1/validation_0020000.npz') as f:assert prediction.tobytes()==f['prediction'].tobytes()
    source=np.load(PARENTS['native']/'banks/sampling-source.npy')[:64]
    assert not np.array_equal(source,b['source'][:64])
    got=imported('resolution').integrate(model,source,128);assert got.tobytes()==np.load(sample_bank(root,'ode_k1'))[:64].tobytes()
    clocks().unchanged(model,before);raw,table,pooled=calibrate()
    np.savez_compressed(root/'native-calibration.npz',**raw);np.savez(root/'clock-table.npz',**table)
    write(root/'pooled-clocks.json',{k:float(v) for k,v in pooled.items()})
    paths=[root/'native-calibration.npz',root/'clock-table.npz',root/'pooled-clocks.json']
    write(root/'calibration-bindings.json',{str(p):sha(p) for p in paths})
    constant_table={k:np.full(129,np.float32(.6)) for k in table};constant={k:float(np.float32(.6)) for k in table}
    expected=clocks().patch(native_model(),{k:dict(value=v) for k,v in constant.items()})
    for mode in ('time','pooled'):
        actual=patch(native_model(),constant_table,constant,mode)
        assert old().predict(actual,b).tobytes()==old().predict(expected,b).tobytes()
    witnesses=[];costs=[]
    for mode in ('time','pooled'):
        literal=patch(imported('clocks20k').witness_model('ode_k1'),table,pooled,mode)
        folded=patch(imported('clocks20k').witness_model('ode_k1'),table,pooled,mode,True)
        for t in (.1,.3,.5,.7,.9):
            panel=bank(t);x=torch.from_numpy(panel['x'][:1]).double();times=torch.from_numpy(panel['t'][:1]).double();y=torch.zeros(1,dtype=torch.long)
            with torch.no_grad():error=float((literal(x,times,y)-folded(x,times,y)).abs().max())
            assert error<1e-9;witnesses.append(dict(mode=mode,time=t,max_error=error))
        count=[0];handles=[];model=patch(native_model(),table,pooled,mode)
        for block in model.blocks:
            for m in block.mlp.modules():
                if isinstance(m,torch.nn.Linear):
                    def count_macs(m,args,out):count[0]+=int(args[0].numel()/m.in_features)*m.in_features*m.out_features
                    handles.append(m.register_forward_hook(count_macs))
        with torch.no_grad():model(torch.from_numpy(b['x'][:64]),torch.from_numpy(b['t'][:64]),torch.zeros(64,dtype=torch.long))
        for h in handles:h.remove()
        work=count[0]//(64*64*3);assert work==8192 and count[0]%(64*64*3)==0
        stored=sum(p.numel() for p in model.parameters());inactive=sum(p.numel() for _,l in clocks().layers(model) for p in l.t_func.parameters())
        assert (stored,inactive)==(211094,25350);costs.append(dict(mode=mode,stored_parameters=stored,inactive_parameters=inactive,
            active_parameters=stored-inactive,FFN_forward_MACs=work,clock_constants=774 if mode=='time' else 6))
    import audit as independent
    calibration_check=independent.verify_calibration(root)
    for mode in ('time','pooled'):
        formal=patch(native_model(),table,pooled,mode);reference=independent.patched(root,mode)
        with torch.no_grad():
            args=(torch.from_numpy(b['x'][:64]),torch.from_numpy(b['t'][:64]),torch.zeros(64,dtype=torch.long))
            assert formal(*args).numpy().tobytes()==reference(*args).numpy().tobytes()
    verify(root,True);write(root/'preflight.json',dict(status='PASS',native1024_predictions_bitwise=True,native64_Heun_bitwise=True,
        calibration_clock_arrays=6,calendar_scalar_means=774,pooled_scalar_means=6,constant_injection1024_bitwise=True,
        promoted_witnesses=witnesses,costs=costs,fixtures=fixture,independent_calibration=calibration_check,new_quality_result=False))
    print(json.dumps(load(root/'preflight.json')),flush=True)
def metric_preflight(root):
    torch.set_num_threads(1);verify(root,True);assert not (root/'metric-preflight.json').exists()
    sys.path.insert(0,str(PARENTS['native']/'source'));import image_metrics as m
    from metric_core import uint8_images,fid
    f=m.features(m.extractor(PARENTS['native']),uint8_images(np.load(sample_bank(root,'ode_k1'))[:32]))
    assert f.tobytes()==np.load(feature_bank(root,'ode_k1'))[:32].tobytes()
    reference=np.load(PARENTS['native']/'metric-inputs/reference-features.npy');value=fid(np.load(feature_bank(root,'ode_k1')),reference)
    old_row=next(r for r in load(PARENTS['clocks20k']/'metric-output/result.json')['rows'] if r['family']=='ode_k1');assert abs(value-old_row['fid'])<1e-8
    write(root/'metric-preflight.json',dict(status='PASS',historical_feature_rows_bitwise=32,native20k_FID=value))
    print(json.dumps(load(root/'metric-preflight.json')),flush=True)
def cell(root,arm):
    torch.set_num_threads(1);verify(root,True);assert load(root/'documented-pass.json')['status']=='PASS'
    table,pooled=calibration(root);model=patch(native_model(),table,pooled,arm.removeprefix('ode_k1_'));before=clocks().tensors(model)
    folder=root/'output/runs'/arm;folder.mkdir(parents=True);tick=time.monotonic();b=bank();prediction=old().predict(model,b)
    np.savez(folder/'validation.npz',prediction=prediction)
    for t in (.1,.3,.5,.7,.9):np.savez(folder/f'panel-{t:.1f}.npz',prediction=old().predict(model,bank(t)))
    def progress(count):write(folder/'status.json',dict(state='RUNNING',images=count,total=1024,seconds=time.monotonic()-tick))
    start=time.monotonic();generated=imported('resolution').integrate(model,np.load(PARENTS['native']/'banks/sampling-source.npy'),128,progress)
    seconds=time.monotonic()-start;np.save(folder/'generated.npy',generated);clocks().unchanged(model,before)
    write(folder/'result.json',dict(family=arm,step=20000,**old().metrics(prediction,b),out_of_range_fraction=float(np.mean(np.abs(generated)>1)),
        sampling_seconds=seconds,seconds=time.monotonic()-tick,parameters_buffers_unchanged=True))
    verify(root,True);write(folder/'status.json',dict(state='COMPLETE',images=1024,seconds=time.monotonic()-tick))
def run(root):
    verify(root,True);assert load(root/'documented-pass.json')['status']=='PASS';out=root/'output';out.mkdir();active={};records=[];tick=time.monotonic()
    try:
        for arm in ARMS:
            command=[TRAIN,'-B',str(root/'source/study.py'),'cell','--root',str(root),'--arm',arm]
            with (out/f'{arm}.log').open('w') as log:p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            monitor=psutil.Process(p.pid);monitor.cpu_percent();active[arm]=(p,monitor,time.monotonic())
            records.append(dict(arm=arm,pid=p.pid,create_time=monitor.create_time(),command=command));write(out/'launches.json',records)
        while active:
            assert time.monotonic()-tick<1300;resources=[]
            for arm,(p,monitor,start) in list(active.items()):
                code=p.poll()
                if code is not None:
                    record=next(r for r in records if r['arm']==arm);record.update(exit_code=code,seconds=time.monotonic()-start)
                    del active[arm];write(out/'launches.json',records);assert code==0,record
                else:
                    assert time.monotonic()-start<1200;resources.append(dict(arm=arm,pid=p.pid,CPU_percent=monitor.cpu_percent(),RSS=monitor.memory_info().rss))
            append(out/'resources.jsonl',dict(seconds=time.monotonic()-tick,resources=resources));write(out/'status.json',dict(state='RUNNING' if active else 'COMPLETE',completed=2-len(active),total=2))
            if active:time.sleep(5)
    except BaseException:
        for p,_,_ in active.values():
            if p.poll() is None:p.terminate()
        for p,_,_ in active.values():p.wait(timeout=30)
        write(out/'status.json',dict(state='FAILED',records=records));raise
    verify(root,True)
def evaluate(root):
    torch.set_num_threads(1);verify(root,True);assert load(root/'audit-sampling/result.json')['status']=='PASS'
    sys.path.insert(0,str(PARENTS['native']/'source'));import image_metrics as m
    from metric_core import uint8_images,fid
    out=root/'metric-output';out.mkdir();extractor=m.extractor(PARENTS['native']);reference=np.load(PARENTS['native']/'metric-inputs/reference-features.npy')
    rows=copy.deepcopy(load(PARENTS['clocks20k']/'metric-output/result.json')['rows'])
    for r in rows:assert abs(fid(np.load(feature_bank(root,r['family'])),reference)-r['fid'])<1e-8
    for arm in ARMS:
        feature=m.features(extractor,uint8_images(np.load(sample_bank(root,arm))),progress=arm);np.save(feature_bank(root,arm),feature)
        row=load(root/'output/runs'/arm/'result.json');cost=next(r for r in load(root/'preflight.json')['costs'] if r['mode']==arm.removeprefix('ode_k1_'))
        row.update(fid=fid(feature,reference),provenance='frozen native20k K1 time-table or matched pooled control',parameters=cost['stored_parameters'],
            active_parameters=cost['active_parameters'],clock_constants=cost['clock_constants'],FFN_forward_MACs=cost['FFN_forward_MACs'],generated=1024,reference=2048,Heun_steps=128)
        rows.append(row)
    for row in rows:
        feature=np.load(feature_bank(root,row['family'])).astype(np.float64);ref=reference.astype(np.float64)
        row['mean_term']=float(np.sum((feature.mean(0)-ref.mean(0))**2));row['covariance_term']=row['fid']-row['mean_term']
    assert len(rows)==11;verify(root,True);write(out/'result.json',dict(status='PASS',rows=rows,new_training=False));write(out/'decision.json',decision(rows))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','preflight','metric-preflight','cell','run','evaluate','fixtures'))
    p.add_argument('--root',type=Path);p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    if a.action=='fixtures':print(json.dumps(fixtures()))
    elif a.action=='cell':cell(a.root,a.arm)
    else:globals()[a.action.replace('-','_')](a.root)
