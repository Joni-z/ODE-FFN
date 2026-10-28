"""Frozen image clocks: one finite intervention, unchanged learned tensors."""
from __future__ import annotations
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,csv,hashlib,importlib.util,json,shutil,subprocess,sys,time,types
from pathlib import Path
import numpy as np
import psutil
import torch

ARMS=('ode_k1','ode_k2');NATIVE=(*ARMS,'swiglu86','swiglu129')
SEALS={'image_parent':'62d178c4e2b88780495c9ea45c6aeef96453ce7098a8aeee84076cf58f081172',
       'resolution_parent':'0a6fb24d1aba029be90e95f7c74919bbbdead973c101819852f04cb55cc64108'}
TRAIN='/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python'
METRICS='/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python'
def load(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
def append(p,v):
    with Path(p).open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n')
def paths(root):return {k:Path(v) for k,v in load(root/'settings.json').items() if k.endswith('_parent')}
def verify(root,calibration=False):
    for file,digest in load(root/'bindings.json').items():assert sha(file)==digest,file
    for key,p in paths(root).items():
        assert sha(p/'sealed-runtime.json')==SEALS[key]
        for file,digest in load(p/'sealed-runtime.json').items():assert sha(p/file)==digest,file
    if calibration:
        for file,digest in load(root/'calibration-bindings.json').items():assert sha(file)==digest,file

def imported(root,which):
    p=paths(root)[which]/'source/study.py';name='bound_'+which
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    assert Path(sys.modules[name].__file__).resolve()==p
    return sys.modules[name]
def native_model(root,arm):
    old=imported(root,'image_parent');p=paths(root)['image_parent'];model=old.model_for(p,arm)
    state=torch.load(p/'output/runs'/arm/'step_0005000.pt',weights_only=False,map_location='cpu');assert state['step']==5000
    old.restore_model(model,state);return model.eval().requires_grad_(False)
def witness_model(root,arm):
    model=native_model(root,arm).double()
    # Algebra witness only: native sinusoidal features are deliberately float32.
    # Lift their values to the promoted MLP dtype, without editing the native
    # module or the float32 model used by formal predictions and sampling.
    model.t_embedder.mlp.register_forward_pre_hook(lambda module,inputs:(inputs[0].to(module[0].weight.dtype),))
    return model

def layers(model):
    for j,block in enumerate(model.blocks):
        for name in ('ode1','ode2'):yield f'blocks.{j}.{name}',getattr(block.mlp,name)
def tensors(model):return {**{f'p:{k}':v.detach().clone() for k,v in model.named_parameters()},**{f'b:{k}':v.detach().clone() for k,v in model.named_buffers()}}
def unchanged(model,before):
    after=tensors(model);assert after.keys()==before.keys()
    for key in before:assert torch.equal(before[key],after[key]),key

def patch(model,constants,kind='literal'):
    for name,layer in layers(model):
        c=float(constants[name]['value']);assert .2<=c<=1
        if kind=='folded':
            p=layer.proj.weight.detach();matrix=torch.eye(len(p),dtype=p.dtype,device=p.device)+c*p
            if layer.orders==2:matrix=matrix+(.5*c*c)*(p@p)
            def forward(self,x,t=None,matrix=matrix):
                assert t is None;return torch.nn.functional.linear(x,matrix)
        elif kind=='independent':
            def forward(self,x,t=None,c=c):
                assert t is None;term=x;answer=x;duration=x.new_tensor(c)
                for j in range(1,self.orders+1):
                    term=(torch.nn.functional.linear(term,self.proj.weight)*duration)/j;answer=answer+term
                return answer
        else:
            def forward(self,x,t=None,c=c):
                assert t is None;parts=[x];duration=x.new_tensor(c)
                for j in range(self.orders):parts.append((self.proj(parts[-1])*duration)/(j+1))
                return sum(parts)
        layer.forward=types.MethodType(forward,layer)
    return model

def calibrate(model,b):
    rows={};raw={};hooks=[]
    for name,layer in layers(model):
        def hook(layer,inputs,name=name):
            assert layer.t_from_embed is None
            with torch.no_grad():s=torch.sigmoid(layer.t_func(inputs[0])/layer.tau)*layer.scale+layer.shift
            assert s.shape==(64,64,1) and name not in rows
            d=s.double();rows[name]=dict(value=float(np.float32(d.mean().item())),native_mean=float(d.mean()),native_SD=float(d.std(unbiased=False)),native_min=float(d.min()),native_max=float(d.max()),count=s.numel())
            raw[name]=s.numpy()
        hooks.append(layer.register_forward_pre_hook(hook))
    with torch.no_grad():pred=model(torch.from_numpy(b['x'][:64]),torch.from_numpy(b['t'][:64]),torch.zeros(64,dtype=torch.long)).numpy()
    for hook in hooks:hook.remove()
    assert len(rows)==6;return rows,raw,pred

def prepare(root,image,resolution):
    assert not root.exists();root.mkdir();(root/'source').mkdir()
    for f in Path(__file__).resolve().parent.iterdir():
        if f.is_file():shutil.copyfile(f,root/'source'/f.name)
    write(root/'settings.json',dict(image_parent=str(image.resolve()),resolution_parent=str(resolution.resolve()),steps=128,calibration_images=64,generated=1024,reference=2048,new_training=False))
    files=[*list((root/'source').iterdir()),root/'settings.json',image/'sealed-runtime.json',resolution/'sealed-runtime.json']
    write(root/'bindings.json',{str(f.resolve()):sha(f) for f in files if f.is_file()});verify(root)
    print(json.dumps(dict(status='PREPARED')),flush=True)

def preflight(root):
    torch.set_num_threads(1);verify(root);old=imported(root,'image_parent');p=paths(root)['image_parent'];b=old.bank(p/'banks/validation.npz')
    # Independent fixed-duration matrix identity at both orders, including AD.
    torch.manual_seed(0);fixture=[]
    for order in (1,2):
        weight=torch.randn(7,7,dtype=torch.float64)*.2;x=torch.randn(3,7,dtype=torch.float64,requires_grad=True);c=.6
        terms=[x]
        for j in range(order):terms.append(torch.nn.functional.linear(terms[-1],weight)*c/(j+1))
        actual=sum(terms);matrix=torch.eye(7,dtype=torch.float64)+c*weight
        if order==2:matrix=matrix+.5*c*c*(weight@weight)
        expected=torch.nn.functional.linear(x,matrix);torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        direction=torch.randn_like(x);da=torch.autograd.grad((actual*direction).sum(),x,retain_graph=True)[0];de=direction@matrix
        torch.testing.assert_close(da,de,atol=1e-12,rtol=1e-12);fixture.append(dict(order=order,max_error=float((actual-expected).detach().abs().max())))
    allrows={};errors=[]
    for arm in ARMS:
        model=native_model(root,arm);before=tensors(model);constants,raw,pred=calibrate(model,b);unchanged(model,before)
        with np.load(p/'output/runs'/arm/'validation_0005000.npz') as saved:assert pred.tobytes()==saved['prediction'][:64].tobytes()
        allrows[arm]=constants;np.savez(root/f'{arm}-native-calibration.npz',**raw)
        literal=patch(witness_model(root,arm),constants);folded=patch(witness_model(root,arm),constants,'folded')
        for value in (.1,.3,.5,.7,.9):
            panel=old.bank(p/'banks'/f'panel_{value:.1f}.npz');x=torch.from_numpy(panel['x'][:1]).double();t=torch.from_numpy(panel['t'][:1]).double();y=torch.zeros(1,dtype=torch.long)
            with torch.no_grad():error=float((literal(x,t,y)-folded(x,t,y)).abs().max())
            assert error<1e-9,(arm,value,error);errors.append(dict(family=arm,time=value,max_error=error))
    write(root/'calibration.json',allrows)
    write(root/'calibration-bindings.json',{str(f):sha(f) for f in [root/'calibration.json',*root.glob('*-native-calibration.npz')]})
    verify(root,True);write(root/'preflight.json',dict(status='PASS',native_first64_bitwise=True,calibrated_clocks=12,fixtures=fixture,full_network_fold_errors=errors))
    print(json.dumps(load(root/'preflight.json')),flush=True)

def cell(root,arm):
    torch.set_num_threads(1);verify(root,True);assert load(root/'preflight.json')['status']=='PASS'
    out=root/'output/runs'/arm;out.mkdir(parents=True);old=imported(root,'image_parent');p=paths(root)['image_parent'];started=time.monotonic()
    model=native_model(root,arm);before=tensors(model);patch(model,load(root/'calibration.json')[arm])
    b=old.bank(p/'banks/validation.npz');prediction=old.predict(model,b);np.savez(out/'validation.npz',prediction=prediction)
    for value in (.1,.3,.5,.7,.9):
        panel=old.bank(p/'banks'/f'panel_{value:.1f}.npz');pred=old.predict(model,panel);np.savez(out/f'panel-{value:.1f}.npz',prediction=pred)
    def progress(count):write(out/'status.json',dict(state='RUNNING',images=count,total=1024,seconds=time.monotonic()-started))
    samples=imported(root,'resolution_parent').integrate(model,np.load(p/'banks/sampling-source.npy'),128,progress)
    np.save(out/'generated.npy',samples);unchanged(model,before);verify(root,True)
    result=dict(family=arm+'_constant',**old.metrics(prediction,b),out_of_range_fraction=float(np.mean(np.abs(samples)>1)),parameters_buffers_unchanged=True,seconds=time.monotonic()-started)
    write(out/'result.json',result);write(out/'status.json',dict(state='COMPLETE',**result))

def run(root):
    verify(root,True);assert load(root/'preflight.json')['status']=='PASS';out=root/'output';out.mkdir();active={};launch=[];exits=[];start=time.monotonic()
    try:
        for arm in ARMS:
            command=[TRAIN,str(root/'source/study.py'),'cell','--root',str(root),'--arm',arm]
            with (out/f'{arm}.log').open('w') as log:process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            active[arm]=(process,time.monotonic());launch.append(dict(family=arm,pid=process.pid,create_time=psutil.Process(process.pid).create_time(),command=command));write(out/'launches.json',launch)
        while active:
            resources=[]
            for arm,(process,started) in list(active.items()):
                code=process.poll()
                if code is not None:
                    exits.append(dict(family=arm,exit_code=code,seconds=time.monotonic()-started));del active[arm];write(out/'exits.json',exits);assert code==0,exits[-1]
                else:
                    assert time.monotonic()-started<1200,(arm,'registered ceiling')
                    try:
                        p=psutil.Process(process.pid);resources.append(dict(family=arm,pid=p.pid,cpu_seconds=sum(p.cpu_times()[:2]),rss=p.memory_info().rss))
                    except psutil.NoSuchProcess:pass
            append(out/'resources.jsonl',dict(elapsed=time.monotonic()-start,processes=resources));write(out/'status.json',dict(state='RUNNING' if active else 'COMPLETE',completed=len(exits),total=2,seconds=time.monotonic()-start))
            if active:time.sleep(5)
        verify(root,True)
    except BaseException:
        for process,_ in active.values():
            if process.poll() is None:process.terminate()
        write(out/'status.json',dict(state='FAILED',exits=exits));raise

def feature_bank(root,arm):
    p=paths(root)
    return p['resolution_parent']/'metric-output'/f'{arm}-128-features.npy' if arm in NATIVE else root/'metric-output'/f'{arm}-features.npy'

def evaluate(root):
    torch.set_num_threads(1);verify(root,True);assert load(root/'output/status.json')['state']=='COMPLETE';p=paths(root);sys.path.insert(0,str(p['image_parent']/'source'));import image_metrics as m
    from metric_core import uint8_images,fid
    out=root/'metric-output';out.mkdir();extractor=m.extractor(p['image_parent']);reference=np.load(p['image_parent']/'metric-inputs/reference-features.npy');rows=[]
    native=list(csv.DictReader((p['image_parent']/'audit-training/comparison.csv').open()));native_fids=load(p['resolution_parent']/'metric-output/result.json')['rows']
    for arm in NATIVE:
        row=dict(next(r for r in native if r['family']==arm));features=np.load(feature_bank(root,arm));score=fid(features,reference)
        assert abs(score-next(r['fid'] for r in native_fids if r['family']==arm and r['N']==128))<1e-8
        row.update(family=arm,fid=score,provenance='sealed native control',out_of_range_fraction=float(np.mean(np.abs(np.load(p['resolution_parent']/'output/runs'/arm/'generated-128.npy'))>1)));rows.append(row)
    for arm in ARMS:
        raw=np.load(root/'output/runs'/arm/'generated.npy');features=m.features(extractor,uint8_images(raw),progress=arm+'_constant');name=arm+'_constant';np.save(out/f'{name}-features.npy',features)
        row=load(root/'output/runs'/arm/'result.json');row.update(fid=fid(features,reference),provenance='frozen clock intervention');rows.append(row)
    for row in rows:
        features=np.load(feature_bank(root,row['family'])).astype(np.float64);mean=float(np.sum((features.mean(0)-reference.astype(np.float64).mean(0))**2))
        row.update(mean_term=mean,covariance_term=row['fid']-mean,generated=1024,reference=2048,steps=128)
    verify(root,True);write(out/'result.json',dict(status='PASS',rows=rows))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('prepare','preflight','cell','run','evaluate'));ap.add_argument('--root',type=Path,required=True);ap.add_argument('--image-parent',type=Path);ap.add_argument('--resolution-parent',type=Path);ap.add_argument('--arm',choices=ARMS);a=ap.parse_args()
    if a.action=='prepare':prepare(a.root,a.image_parent,a.resolution_parent)
    elif a.action=='cell':cell(a.root,a.arm)
    else:globals()[a.action](a.root)
