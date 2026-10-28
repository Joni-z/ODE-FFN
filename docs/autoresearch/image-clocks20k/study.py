"""One frozen clock intervention at the sealed native 20k checkpoints."""
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

BASE=Path('/Users/mr.z/ode-workspace-prep/autoresearch')
PARENTS={'budget':BASE/'oct28-image-budget-v2','clocks':BASE/'oct28-cycle19-v2',
         'native':BASE/'oct28-cycle17-v2','resolution':BASE/'oct28-cycle18'}
SEALS={'budget':'ac381ff444384e052c4cc48e9f11550f00ab410a233154972177e97fe543a48b',
       'clocks':'c8ecd01c4c24ad6be6f3a8d1bf3aa7b5b3222f03712e4879e2cd5ecf15759d96',
       'native':'62d178c4e2b88780495c9ea45c6aeef96453ce7098a8aeee84076cf58f081172',
       'resolution':'0a6fb24d1aba029be90e95f7c74919bbbdead973c101819852f04cb55cc64108'}
ARMS=('ode_k1','ode_k2')
NATIVE=(*ARMS,'swiglu86','swiglu129','silu64','silu131','silu193')
CONTROLS={'ode_k1':('swiglu86','silu131'),'ode_k2':('swiglu129','silu193')}
TRAIN='/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python'
METRICS='/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python'
load=lambda p:json.loads(Path(p).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    p=Path(path);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(p)
def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
def imported(key):
    name='clock20k_parent_'+key;path=PARENTS[key]/'source/study.py'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    assert Path(sys.modules[name].__file__).resolve()==path
    return sys.modules[name]
def verify(root,calibration=False):
    for path,digest in load(root/'bindings.json').items():assert sha(path)==digest,path
    for key,parent in PARENTS.items():
        assert sha(parent/'sealed-runtime.json')==SEALS[key]
        for name,digest in load(parent/'sealed-runtime.json').items():assert sha(parent/name)==digest,name
    if calibration:
        for path,digest in load(root/'calibration-bindings.json').items():assert sha(path)==digest,path
def native_model(arm):
    budget=imported('budget');model=budget.construct(arm)
    state=torch.load(PARENTS['budget']/'output/runs'/arm/'step_0020000.pt',weights_only=False,map_location='cpu')
    assert state['step']==20000
    budget.imported('native').restore_model(model,state)
    return model.eval().requires_grad_(False)
def old():return imported('budget').imported('native')
def clocks():return imported('clocks')
def witness_model(arm):
    model=native_model(arm).double()
    model.t_embedder.mlp.register_forward_pre_hook(lambda m,x:(x[0].to(m[0].weight.dtype),))
    return model
def bank(time_value=None):
    name='validation.npz' if time_value is None else f'panel_{time_value:.1f}.npz'
    return old().bank(PARENTS['native']/'banks'/name)
def feature_bank(root,family):
    if family in NATIVE:return PARENTS['budget']/'metric-output'/f'{family}-0020000-features.npy'
    return root/'metric-output'/f'{family}-features.npy'
def sample_bank(root,family):
    if family in NATIVE:return PARENTS['budget']/'output/runs'/family/'generated-0020000.npy'
    return root/'output/runs'/family.removesuffix('_constant')/'generated.npy'

def decisions(rows,native_decision):
    by={r['family']:r for r in rows};out=[]
    for arm,references in CONTROLS.items():
        native=by[arm];constant=by[arm+'_constant'];best=min(by[a]['fid'] for a in references)
        ratios=[dict(reference=a,**{k+'_ratio':constant[k]/by[a][k] for k in ('fid','weighted_mse','clean_mse')}) for a in references]
        fit={k:constant[k]/min(by[a][k] for a in references) for k in ('weighted_mse','clean_mse')}
        out.append(dict(arm=arm,strict_FID_order_retained=constant['fid']<best,
            allocation_margin_retained=all(r['fid_ratio']<=.95 for r in ratios) and all(v<=1.05 for v in fit.values()),
            comparisons=ratios,fitting_ratios_to_best_control=fit,
            native_to_constant={k:constant[k]-native[k] for k in ('fid','weighted_mse','clean_mse')},
            ordinary_gap_retained_fraction=(best-constant['fid'])/(best-native['fid']) if best!=native['fid'] else None,
            native_allocation_margin_met=next(r['follow_up_priority_margin_met'] for r in native_decision['criteria'] if r['arm']==arm),
            new_method_established=False,old_failed_gate_relabelled=False))
    return out
def decision_fixtures():
    native=dict(criteria=[dict(arm=a,follow_up_priority_margin_met=a=='ode_k1') for a in ARMS])
    rows=[dict(family=a,fid=120.,weighted_mse=1.,clean_mse=1.) for a in NATIVE]
    next(r for r in rows if r['family']=='ode_k1')['fid']=90.
    next(r for r in rows if r['family']=='ode_k2')['fid']=119.
    rows.extend(dict(family=a+'_constant',fid=95.,weighted_mse=1.05,clean_mse=1.05) for a in ARMS)
    next(r for r in rows if r['family']=='silu131')['fid']=100.
    assert decisions(rows,native)[0]['allocation_margin_retained']
    x=copy.deepcopy(rows);next(r for r in x if r['family']=='ode_k1_constant')['fid']=99.
    d=decisions(x,native)[0];assert d['strict_FID_order_retained'] and not d['allocation_margin_retained']
    next(r for r in x if r['family']=='ode_k1_constant')['fid']=100.
    assert not decisions(x,native)[0]['strict_FID_order_retained']
    x=copy.deepcopy(rows);next(r for r in x if r['family']=='ode_k1_constant')['weighted_mse']=1.051
    assert not decisions(x,native)[0]['allocation_margin_retained']
    assert not decisions(rows,native)[1]['native_allocation_margin_met']
    return dict(status='PASS',cases=['inclusive margin','ordering distinct from margin','strict tie','fitting guard','old K2 decision preserved'])

def operator_fixtures():
    torch.manual_seed(0);errors=[]
    for order in (1,2):
        for _ in range(3):
            p=torch.randn(7,7,dtype=torch.float64)*.2;a=torch.randn(7,dtype=torch.float64)
            x=torch.randn(7,dtype=torch.float64,requires_grad=True)
            clock=lambda h:.2+.8*torch.sigmoid((a*h).sum()/10)
            def function(h,fixed=None):
                s=clock(h) if fixed is None else fixed;term=h;answer=h
                for j in range(1,order+1):term=(p@term)*s/j;answer=answer+term
                return answer
            s=clock(x);g=torch.autograd.grad(s,x)[0]
            matrix=torch.eye(7,dtype=torch.float64)+s*p;v=p@x
            if order==2:matrix=matrix+.5*s*s*(p@p);v=v+s*(p@p@x)
            expected=matrix+torch.outer(v,g)
            observed=torch.autograd.functional.jacobian(function,x)
            torch.testing.assert_close(observed,expected,rtol=1e-12,atol=1e-12)
            fixed=torch.autograd.functional.jacobian(lambda h:function(h,.6),x)
            target=torch.eye(7,dtype=torch.float64)+.6*p
            if order==2:target=target+.18*(p@p)
            torch.testing.assert_close(fixed,target,rtol=1e-12,atol=1e-12)
            errors.append(dict(order=order,complete_Jacobian_max_error=float((observed-expected).detach().abs().max()),constant_Jacobian_max_error=float((fixed-target).abs().max())))
    return dict(status='PASS',rows=errors)

def prepare(root):
    assert not root.exists();root.mkdir();(root/'source').mkdir()
    for p in Path(__file__).resolve().parent.iterdir():
        if p.is_file():shutil.copyfile(p,root/'source'/p.name)
    repo=Path(__file__).resolve().parents[3]
    write(root/'settings.json',dict(source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        checkpoint_step=20000,interventions=ARMS,reused_native=NATIVE,calibration_images=64,steps=128,
        generated=1024,reference=2048,new_training=False,new_architecture=False))
    files=[*(root/'source').iterdir(),root/'settings.json',*(p/'sealed-runtime.json' for p in PARENTS.values())]
    write(root/'bindings.json',{str(p.resolve()):sha(p) for p in files if p.is_file()});verify(root)
    print(json.dumps(dict(status='PREPARED',new_training=False)),flush=True)
def preflight(root):
    torch.set_num_threads(1);verify(root);assert not (root/'preflight.json').exists()
    assert sha(__file__)==sha(root/'source/study.py');fixture=decision_fixtures();operators=operator_fixtures()
    constants={};witnesses=[];c=clocks();b=bank()
    for arm in ARMS:
        model=native_model(arm);before=c.tensors(model);prediction=old().predict(model,b)
        with np.load(PARENTS['budget']/'output/runs'/arm/'validation_0020000.npz') as f:assert prediction.tobytes()==f['prediction'].tobytes()
        values,raw,first=c.calibrate(model,b);assert first.tobytes()==prediction[:64].tobytes()
        c.unchanged(model,before);constants[arm]=values
        np.savez(root/f'{arm}-native-calibration.npz',**raw)
        literal=c.patch(witness_model(arm),values);folded=c.patch(witness_model(arm),values,'folded')
        for t in (.1,.3,.5,.7,.9):
            p=bank(t);x=torch.from_numpy(p['x'][:1]).double();time_tensor=torch.from_numpy(p['t'][:1]).double();label=torch.zeros(1,dtype=torch.long)
            with torch.no_grad():error=float((literal(x,time_tensor,label)-folded(x,time_tensor,label)).abs().max())
            assert error<1e-9,(arm,t,error);witnesses.append(dict(arm=arm,time=t,max_error=error))
    write(root/'calibration.json',constants)
    write(root/'calibration-bindings.json',{str(p):sha(p) for p in [root/'calibration.json',*root.glob('*-native-calibration.npz')]})
    verify(root,True)
    write(root/'preflight.json',dict(status='PASS',checkpoint_step=20000,native_1024_predictions_bitwise=True,
        unchanged_parameters_buffers=True,calibrated_clocks=12,whole_network_fold_witnesses=witnesses,decision_fixtures=fixture,
        operator_fixtures=operators,new_quality_result=False))
    print(json.dumps(load(root/'preflight.json')),flush=True)
def cell(root,arm):
    torch.set_num_threads(1);verify(root,True);assert load(root/'documented-pass.json')['status']=='PASS'
    c=clocks();model=native_model(arm);before=c.tensors(model);c.patch(model,load(root/'calibration.json')[arm])
    folder=root/'output/runs'/arm;folder.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    b=bank();prediction=old().predict(model,b);np.savez(folder/'validation.npz',prediction=prediction)
    for t in (.1,.3,.5,.7,.9):np.savez(folder/f'panel-{t:.1f}.npz',prediction=old().predict(model,bank(t)))
    def progress(count):write(folder/'status.json',dict(state='RUNNING',images=count,total=1024,seconds=time.monotonic()-start))
    generated=imported('resolution').integrate(model,np.load(PARENTS['native']/'banks/sampling-source.npy'),128,progress)
    np.save(folder/'generated.npy',generated);c.unchanged(model,before);verify(root,True)
    result=dict(family=arm+'_constant',step=20000,**old().metrics(prediction,b),
        out_of_range_fraction=float(np.mean(np.abs(generated)>1)),parameters_buffers_unchanged=True,seconds=time.monotonic()-start)
    write(folder/'result.json',result);write(folder/'status.json',dict(state='COMPLETE',**result))
def run(root):
    verify(root,True);assert load(root/'preflight.json')['status']=='PASS' and load(root/'documented-pass.json')['status']=='PASS'
    out=root/'output';out.mkdir();active={};exits=[];launches=[];tick=time.monotonic()
    try:
        for arm in ARMS:
            command=[TRAIN,'-B',str(root/'source/study.py'),'cell','--root',str(root),'--arm',arm]
            with (out/f'{arm}.log').open('w') as log:p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            monitor=psutil.Process(p.pid);monitor.cpu_percent();active[arm]=(p,monitor,time.monotonic())
            launches.append(dict(arm=arm,pid=p.pid,create_time=monitor.create_time(),command=command));write(out/'launches.json',launches)
        while active:
            assert time.monotonic()-tick<1300;resources=[]
            for arm,(p,monitor,start) in list(active.items()):
                code=p.poll()
                if code is not None:
                    exits.append(dict(arm=arm,pid=p.pid,exit_code=code,seconds=time.monotonic()-start));del active[arm]
                    write(out/'exits.json',exits);assert code==0,exits[-1]
                else:
                    assert time.monotonic()-start<1200
                    resources.append(dict(arm=arm,pid=p.pid,RSS=monitor.memory_info().rss,CPU_percent=monitor.cpu_percent()))
            append(out/'resources.jsonl',dict(seconds=time.monotonic()-tick,resources=resources))
            write(out/'status.json',dict(state='RUNNING' if active else 'COMPLETE',completed=len(exits),total=2))
            if active:time.sleep(5)
    except BaseException:
        for p,_,_ in active.values():
            if p.poll() is None:p.terminate()
        for p,_,_ in active.values():p.wait(timeout=30)
        write(out/'status.json',dict(state='FAILED',exits=exits));raise
    verify(root,True)
def evaluate(root):
    torch.set_num_threads(1);verify(root,True);assert load(root/'output/status.json')['state']=='COMPLETE'
    sys.path.insert(0,str(PARENTS['native']/'source'));import image_metrics as m
    from metric_core import uint8_images,fid
    out=root/'metric-output';out.mkdir();extractor=m.extractor(PARENTS['native'])
    reference=np.load(PARENTS['native']/'metric-inputs/reference-features.npy');rows=[]
    formal=load(PARENTS['budget']/'metric-output/result.json')['rows']
    costs={r['arm']:r for r in csv.DictReader((PARENTS['budget']/'curation/comparison.csv').open()) if int(r['step'])==20000}
    for arm in NATIVE:
        record=copy.deepcopy(next(r for r in formal if r['arm']==arm and r['step']==20000));feature=np.load(feature_bank(root,arm))
        assert abs(fid(feature,reference)-record['fid'])<1e-8
        record['family']=record.pop('arm');record.update(provenance='sealed native20k',parameters=int(costs[arm]['parameters']),FFN_forward_MACs=int(costs[arm]['FFN_forward_MACs']))
        rows.append(record)
    for arm in ARMS:
        family=arm+'_constant';raw=np.load(sample_bank(root,family));feature=m.features(extractor,uint8_images(raw),progress=family)
        np.save(feature_bank(root,family),feature)
        row=load(root/'output/runs'/arm/'result.json');row.update(fid=fid(feature,reference),provenance='frozen native20k clock intervention',
            generated=1024,reference=2048,Heun_steps=128,parameters=int(costs[arm]['parameters']),FFN_forward_MACs=8192 if arm=='ode_k1' else 16384)
        rows.append(row)
    for row in rows:
        feature=np.load(feature_bank(root,row['family'])).astype(np.float64)
        row['mean_term']=float(np.sum((feature.mean(0)-reference.astype(np.float64).mean(0))**2));row['covariance_term']=row['fid']-row['mean_term']
    verify(root,True);write(out/'result.json',dict(status='PASS',rows=rows,new_training=False))
    write(out/'decision.json',dict(criteria=decisions(rows,load(PARENTS['budget']/'curation/decision.json')),new_method_established=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','preflight','cell','run','evaluate','check-rule'))
    p.add_argument('--root',type=Path);p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    if a.action=='check-rule':print(json.dumps(decision_fixtures()))
    elif a.action=='cell':cell(a.root,a.arm)
    else:globals()[a.action](a.root)
