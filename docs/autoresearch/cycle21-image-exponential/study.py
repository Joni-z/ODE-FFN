"""Frozen cached polynomial/exponential maps; no training or new data."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,copy,csv,hashlib,importlib.util,json,shutil,subprocess,sys,time,types
from pathlib import Path
import numpy as np
import psutil
import torch

ARMS=('ode_k1_folded','ode_k1_exp','ode_k2_folded','ode_k2_exp')
SNAPSHOTS=(0,1000,2500,5000)
BASE=Path('/Users/mr.z/ode-workspace-prep/autoresearch')
PARENTS={k:BASE/v for k,v in dict(image_parent='oct28-cycle17-v2',resolution_parent='oct28-cycle18',clock_parent='oct28-cycle19-v2',ordinary_parent='oct28-cycle20').items()}
SEALS=dict(image_parent='62d178c4e2b88780495c9ea45c6aeef96453ce7098a8aeee84076cf58f081172',resolution_parent='0a6fb24d1aba029be90e95f7c74919bbbdead973c101819852f04cb55cc64108',clock_parent='c8ecd01c4c24ad6be6f3a8d1bf3aa7b5b3222f03712e4879e2cd5ecf15759d96',ordinary_parent='d7fe17e92c7bd8468b6c048aa9925a4350d95168ed611682bb9c7d787634e973')
TRAIN='/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python'
CHECK='/Users/mr.z/ode-workspace-prep/.venv-test/bin/python'
METRICS='/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python'

def load(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(tmp,path)
def append(path,value):
    with Path(path).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
def paths(root):return {k:Path(v) for k,v in load(root/'settings.json').items() if k.endswith('_parent')}
def imported(root,key):
    path=paths(root)[key]/'source/study.py';name='cycle21_bound_'+key
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    assert Path(sys.modules[name].__file__).resolve()==path
    return sys.modules[name]
def verify(root,ready=False):
    for file,digest in load(root/'bindings.json').items():assert sha(file)==digest,file
    for key,parent in paths(root).items():
        assert sha(parent/'sealed-runtime.json')==SEALS[key]
        for file,digest in load(parent/'sealed-runtime.json').items():assert sha(parent/file)==digest,file
    imported(root,'image_parent').verify(paths(root)['image_parent'],True)
    if ready:
        for file,digest in load(root/'operator-bindings.json').items():assert sha(file)==digest,file
        assert load(root/'ready.json')['preflight']==sha(root/'preflight.json')
        assert load(root/'ready.json')['operator_bindings']==sha(root/'operator-bindings.json')

def prepare(root):
    assert not root.exists();root.mkdir();(root/'source').mkdir()
    for file in Path(__file__).resolve().parent.iterdir():
        if file.is_file():shutil.copyfile(file,root/'source'/file.name)
    write(root/'settings.json',{**{k:str(p) for k,p in PARENTS.items()},'source_commit':subprocess.check_output(['git','-C',str(Path(__file__).resolve().parent),'rev-parse','HEAD'],text=True).strip(),'arms':ARMS,'new_training':False,'steps':128,'generated':1024,'reference':2048})
    files=[*(root/'source').iterdir(),root/'settings.json',*(p/'sealed-runtime.json' for p in PARENTS.values())]
    write(root/'bindings.json',{str(f):sha(f) for f in files if f.is_file()});verify(root)
    print(json.dumps(dict(status='PREPARED')),flush=True)

def native(root,arm,step=5000):
    old=imported(root,'image_parent');p=paths(root)['image_parent'];model=old.model_for(p,arm)
    state=torch.load(p/'output/runs'/arm/f'step_{step:07d}.pt',weights_only=False,map_location='cpu');assert state['step']==step
    old.restore_model(model,state);return model.eval().requires_grad_(False)
def layers(root,model):return imported(root,'clock_parent').layers(model)
def tensors(root,model):return imported(root,'clock_parent').tensors(model)
def unchanged(root,model,before):imported(root,'clock_parent').unchanged(model,before)
def polynomial(a,order):
    result=torch.eye(len(a),dtype=a.dtype)+a
    return result if order==1 else result+.5*(a@a)
def token(name):return name.replace('.','_')

def operator_rows(root,arm,step,save=False):
    constants=load(paths(root)['clock_parent']/'calibration.json')[arm];rows=[];arrays={};model=native(root,arm,step)
    for name,layer in layers(root,model):
        p=layer.proj.weight.detach().double();c=constants[name]['value'];a=c*p;t=polynomial(a,layer.orders);e=torch.linalg.matrix_exp(a)
        sv=torch.linalg.svdvals(p);eig=torch.linalg.eigvals(p);mu=torch.linalg.eigvalsh((p+p.T)/2).max()
        defect=polynomial(a/2,layer.orders)@polynomial(a/2,layer.orders)-t
        half=torch.linalg.matrix_exp(a/2);norm=float(torch.linalg.norm(e));prefix=token(name)
        arrays.update({prefix+'_P':p.numpy(),prefix+'_T':t.numpy(),prefix+'_E':e.numpy(),prefix+'_T32':t.float().numpy(),prefix+'_E32':e.float().numpy()})
        rows.append(dict(family=arm,step=step,layer=name,c=c,order=layer.orders,P_sigma_max=float(sv[0]),P_singular_values=sv.tolist(),P_eigen_real=eig.real.tolist(),P_eigen_imag=eig.imag.tolist(),numerical_abscissa=float(mu),spectral_abscissa=float(eig.real.max()),nonnormality=float(torch.linalg.norm(p.T@p-p@p.T)/(torch.linalg.norm(p)**2)),A_norm=float(torch.linalg.norm(a,ord=2)),polynomial_norm=float(torch.linalg.norm(t,ord=2)),exponential_norm=float(torch.linalg.norm(e,ord=2)),relative_Frobenius_error=float(torch.linalg.norm(e-t)/norm),polynomial_semigroup_defect=float(torch.linalg.norm(defect)/norm),exponential_semigroup_defect=float(torch.linalg.norm(half@half-e)/norm)))
    if save:np.savez(root/'operators'/f'{arm}-{step:07d}.npz',**arrays)
    return rows

def patch(root,model,arm,kind):
    with np.load(root/'operators'/f'{arm}-0005000.npz') as f:
        for name,layer in layers(root,model):
            matrix=torch.from_numpy(f[token(name)+('_T32' if kind=='folded' else '_E32')])
            def forward(self,x,t=None,matrix=matrix):
                assert t is None
                return torch.nn.functional.linear(x,matrix)
            layer.forward=types.MethodType(forward,layer)
    return model
def model_for(root,arm):
    base,kind=arm.rsplit('_',1);return patch(root,native(root,base),base,kind)

def preflight(root):
    torch.set_num_threads(1);verify(root);old=imported(root,'image_parent');p=paths(root)['image_parent'];clocks=imported(root,'clock_parent');cparent=paths(root)['clock_parent']
    fixtures=[]
    for key,a in [('zero',torch.zeros(3,3,dtype=torch.float64)),('negative',torch.diag(torch.tensor([-3.,-1.,-.25],dtype=torch.float64))),('nonnormal',-.25*torch.eye(3,dtype=torch.float64)+4*torch.diag(torch.ones(2,dtype=torch.float64),diagonal=1))]:
        e=torch.linalg.matrix_exp(a)
        if key=='zero':expected=np.eye(3)
        elif key=='negative':expected=np.diag(np.exp(np.diag(a.numpy())))
        else:
            n=a.numpy()+.25*np.eye(3);expected=np.exp(-.25)*(np.eye(3)+n+.5*n@n)
        np.testing.assert_allclose(e.numpy(),expected,rtol=1e-11,atol=1e-11)
        for order in (1,2):
            defect=polynomial(a/2,order)@polynomial(a/2,order)-polynomial(a,order)
            exact=a@a/4 if order==1 else a@a@a/8+a@a@a@a/64
            torch.testing.assert_close(defect,exact,rtol=1e-12,atol=1e-12)
        if key=='nonnormal':
            n=a+.25*torch.eye(3,dtype=torch.float64);closed=np.exp(-.25)*(torch.eye(3,dtype=torch.float64)+n+.5*n@n)
            torch.testing.assert_close(e,closed,rtol=1e-12,atol=1e-12)
        fixtures.append(dict(name=key,max_expm_difference=float(np.max(np.abs(e.numpy()-expected)))))
    (root/'operators').mkdir();history=[];replays=[];promoted=[]
    for arm in ('ode_k1','ode_k2'):
        for step in SNAPSHOTS:
            history.extend(operator_rows(root,arm,step,True))
        bank=old.bank(p/'banks/validation.npz');model=native(root,arm);before=tensors(root,model)
        pred=old.predict(model,bank)
        with np.load(p/'output/runs'/arm/'validation_0005000.npz') as saved:assert pred.tobytes()==saved['prediction'].tobytes()
        constants=load(cparent/'calibration.json')[arm];clocks.patch(model,constants);pred=old.predict(model,bank)
        with np.load(cparent/'output/runs'/arm/'validation.npz') as saved:assert pred.tobytes()==saved['prediction'].tobytes()
        unchanged(root,model,before);replays.append(dict(family=arm,native_and_constant_bitwise=True))
        literal=clocks.witness_model(root,arm);folded=copy.deepcopy(literal);clocks.patch(literal,constants);clocks.patch(folded,constants,'folded')
        for value in (.1,.3,.5,.7,.9):
            panel=old.bank(p/'banks'/f'panel_{value:.1f}.npz');x=torch.from_numpy(panel['x'][:1]).double();time_value=torch.from_numpy(panel['t'][:1]).double();y=torch.zeros(1,dtype=torch.long)
            with torch.no_grad():error=float((literal(x,time_value,y)-folded(x,time_value,y)).abs().max())
            assert error<1e-9;promoted.append(dict(family=arm,time=value,max_error=error))
    assert len(history)==48;write(root/'operator-history.json',history)
    subprocess.run([CHECK,str(root/'source/operator_check.py'),'--root',str(root),'--output',str(root/'operator-check.json')],check=True)
    files=[*list((root/'operators').iterdir()),root/'operator-history.json',root/'operator-check.json']
    write(root/'operator-bindings.json',{str(f):sha(f) for f in files})
    write(root/'preflight.json',dict(status='PASS',fixtures=fixtures,operator_records=48,independent_expm_rtol=1e-11,replays=replays,promoted_errors=promoted,new_training=False))
    write(root/'ready.json',dict(preflight=sha(root/'preflight.json'),operator_bindings=sha(root/'operator-bindings.json')));verify(root,True)
    print(json.dumps(dict(status='READY',operator_records=48,replayed_models=2)),flush=True)

def cell(root,arm):
    torch.set_num_threads(1);verify(root,True);out=root/'output/runs'/arm;out.mkdir(parents=True);started=time.monotonic()
    model=model_for(root,arm);before=tensors(root,model);old=imported(root,'image_parent');p=paths(root)['image_parent'];bank=old.bank(p/'banks/validation.npz')
    pred=old.predict(model,bank);np.savez(out/'validation.npz',prediction=pred)
    for value in (.1,.3,.5,.7,.9):
        b=old.bank(p/'banks'/f'panel_{value:.1f}.npz');np.savez(out/f'panel-{value:.1f}.npz',prediction=old.predict(model,b))
    def progress(count):write(out/'status.json',dict(state='RUNNING',images=count,total=1024,seconds=time.monotonic()-started))
    samples=imported(root,'resolution_parent').integrate(model,np.load(p/'banks/sampling-source.npy'),128,progress);np.save(out/'generated.npy',samples)
    unchanged(root,model,before);verify(root,True)
    result=dict(family=arm,**old.metrics(pred,bank),out_of_range_fraction=float(np.mean(np.abs(samples)>1)),parameters_buffers_unchanged=True,seconds=time.monotonic()-started)
    write(out/'result.json',result);write(out/'status.json',dict(state='COMPLETE',**result))

def run(root):
    verify(root,True);out=root/'output';out.mkdir();active={};pending=list(ARMS);records=[];exits=[];start=time.monotonic()
    try:
        while pending or active:
            while pending and len(active)<2:
                arm=pending.pop(0);cmd=[TRAIN,str(root/'source/study.py'),'cell','--root',str(root),'--arm',arm]
                with (out/f'{arm}.log').open('w') as log:proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                record=dict(family=arm,command=cmd,pid=proc.pid,create_time=psutil.Process(proc.pid).create_time());records.append(record);active[arm]=(proc,time.monotonic(),record);write(out/'launches.json',records)
            assert time.monotonic()-start<1800
            resources=[]
            for arm,(proc,t0,record) in list(active.items()):
                code=proc.poll()
                if code is not None:
                    record.update(exit_code=code,seconds=time.monotonic()-t0);exits.append(dict(record));del active[arm];write(out/'exits.json',exits);write(out/'launches.json',records);assert code==0,record
                else:
                    assert time.monotonic()-t0<900
                    try:
                        p=psutil.Process(proc.pid);resources.append(dict(family=arm,pid=p.pid,cpu_seconds=sum(p.cpu_times()[:2]),rss=p.memory_info().rss))
                    except psutil.NoSuchProcess:pass
            append(out/'resources.jsonl',dict(seconds=time.monotonic()-start,processes=resources));write(out/'status.json',dict(state='RUNNING' if active or pending else 'COMPLETE',completed=len(exits),total=4,seconds=time.monotonic()-start))
            if active:time.sleep(5)
        verify(root,True)
    except BaseException:
        for proc,_,_ in active.values():
            if proc.poll() is None:proc.terminate()
        write(out/'status.json',dict(state='FAILED',records=records));raise

def feature_bank(root,arm):
    if arm in ARMS:return root/'metric-output'/f'{arm}-features.npy'
    if arm.startswith('silu'):return paths(root)['ordinary_parent']/'metric-output'/f'{arm}-features.npy'
    if arm.endswith('_constant'):return paths(root)['clock_parent']/'metric-output'/f'{arm}-features.npy'
    return paths(root)['resolution_parent']/'metric-output'/f'{arm}-128-features.npy'

def evaluate(root):
    torch.set_num_threads(1);verify(root,True);assert load(root/'output/status.json')['state']=='COMPLETE';out=root/'metric-output';out.mkdir()
    p=paths(root)['image_parent'];sys.path.insert(0,str(p/'source'));import image_metrics as metrics
    from metric_core import uint8_images,fid
    extractor=metrics.extractor(p);ref=np.load(p/'metric-inputs/reference-features.npy');rows=list(csv.DictReader((paths(root)['ordinary_parent']/'curation/comparison.csv').open()))
    for row in rows:
        for key in ('weighted_mse','clean_mse','fid','mean_term','covariance_term','out_of_range_fraction'):row[key]=float(row[key])
        row['provenance']='explicitly reused sealed endpoint'
    for arm in ARMS:
        raw=np.load(root/'output/runs'/arm/'generated.npy');features=metrics.features(extractor,uint8_images(raw),progress=arm);np.save(feature_bank(root,arm),features)
        score=fid(features,ref);mean=float(np.sum((features.astype(np.float64).mean(0)-ref.astype(np.float64).mean(0))**2));row=load(root/'output/runs'/arm/'result.json')
        row.update(fid=score,mean_term=mean,covariance_term=score-mean,generated=1024,reference=2048,steps=128,parameters='',FFN_forward_MACs='',FFN_training_MACs='',provenance='new frozen cached-map intervention');rows.append(row)
    assert len(rows)==13;verify(root,True);write(out/'result.json',dict(status='PASS',rows=rows))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','preflight','cell','run','evaluate'));parser.add_argument('--root',type=Path,required=True);parser.add_argument('--arm',choices=ARMS);a=parser.parse_args()
    if a.action=='cell':cell(a.root,a.arm)
    else:globals()[a.action](a.root)
