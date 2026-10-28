"""One ordinary Swi86 initialization change on the immutable image protocol."""
from __future__ import annotations
import os,sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','PYTHONDONTWRITEBYTECODE'):os.environ[k]='1'
sys.dont_write_bytecode=True
import argparse,copy,hashlib,importlib.metadata,importlib.util,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
if __name__=='__main__':sys.modules['init_study']=sys.modules[__name__]
BASE=Path('/Users/mr.z/ode-workspace-prep/autoresearch');COLD=BASE/'oct28-image-clock-training-v2';NATIVE=BASE/'oct28-cycle17-v2';BUDGET=BASE/'oct28-image-budget-v2';GEOMETRY=BASE/'oct29-image-response-geometry-v1'
ARM='matched_swiglu86';STEPS=(0,1000,2500,5000,10000,15000,20000);SAMPLES=(5000,10000,20000)
TRAIN='/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python';METRICS='/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python'
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def module(name,path):
    path=Path(path).resolve()
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    assert Path(sys.modules[name].__file__).resolve()==path;return sys.modules[name]
def parents():
    sys.path.insert(0,str(COLD/'source'));c=module('training_controls',COLD/'source/training_controls.py');n=c.old();b=c.imported('budget')
    assert Path(n.__file__).resolve()==NATIVE/'source/study.py' and Path(b.__file__).resolve()==BUDGET/'source/study.py'
    torch.set_num_threads(1);return c,n,b
def environment(root,kind):
    p=(root/'source' if root.exists() else Path(__file__).parent)/(kind+'-env-spec.json');spec=load(p)
    expected={'training':'475428fb','metrics':'dc729a50'}[kind]
    assert hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]==expected
    for phase in spec['pip_phases']:
        for req in phase:
            name,version=req.split('==');assert importlib.metadata.version(name)==version
def verify(root):
    for p,h in load(root/'bindings.json').items():assert sha(p)==h,p
    for p,h in load(NATIVE/'bindings.json').items():assert sha(p)==h,p
    for parent in (COLD,NATIVE,BUDGET,GEOMETRY):
        for name,h in load(parent/'sealed-runtime.json').items():assert sha(parent/name)==h,(parent,name)
def shared(name):return not (name.startswith('blocks.') and '.mlp.' in name)
def construct(arm=ARM,native=None,promote=False,independent=False):
    assert arm==ARM and native is None and not promote and not independent
    c,n,b=parents();model=b.construct('swiglu86')
    original=torch.load(NATIVE/'output/runs/swiglu86/step_0000000.pt',weights_only=False,map_location='cpu');n.restore_model(model,original)
    silu=torch.load(COLD/'output/runs/matched_silu64/step_0000000.pt',weights_only=False,map_location='cpu')
    for name,value in model.state_dict().items():
        if shared(name):assert torch.equal(value,silu['model'][name]),name
    for name,value in model.named_buffers():assert torch.equal(value,silu['runtime_buffers'][name])
    with torch.no_grad():
        for j,block in enumerate(model.blocks):
            f=block.mlp;r=f.w3.in_features;assert r==86 and f.w12.bias is not None and f.w3.bias is not None
            f.w12.weight[:64].copy_(silu['model'][f'blocks.{j}.mlp.fc1.weight']);f.w12.bias[:64].zero_()
            f.w12.weight[r:r+64].zero_();f.w12.bias[r:r+64].fill_(1.)
            f.w3.weight.zero_();f.w3.weight[:,:64].copy_(silu['model'][f'blocks.{j}.mlp.fc2.weight']);f.w3.bias.zero_()
    return model
def restored(state=None):
    c,n,_=parents();m=construct();opt=n.optimizer(m);rng=torch.Generator()
    if state is None:
        initial=c.native_state(0);rng.set_state(initial['data_rng']);torch.set_rng_state(initial['torch_rng']);digest=initial['input_stream_digest']
    else:digest=n.restore(m,opt,rng,copy.deepcopy(state))
    return m,opt,rng,digest
def model_at(root,step):
    _,n,_=parents();m=construct();state=torch.load(root/'output/runs'/ARM/f'step_{step:07d}.pt',weights_only=False,map_location='cpu');assert state['step']==step;n.restore_model(m,state)
    return m.eval().requires_grad_(False)
def values(model):
    answer={}
    for j,block in enumerate(model.blocks):
        gate,value=block.mlp.w12.weight.detach().chunk(2,0)
        for name,w in (('gate',gate),('value',value),('output',block.mlp.w3.weight.detach())):answer[f'block{j}.{name}']=w.numpy().copy()
    return answer
def history(model,arm,folder,step):
    assert arm==ARM;data=values(model);np.savez(folder/f'matrices-{step:07d}.npz',**data);rows=[]
    for name,a in data.items():
        s=np.linalg.svd(a.astype(np.float64),compute_uv=False);rows.append(dict(layer=name,singular_values=s.tolist(),frobenius=float(np.linalg.norm(a.astype(np.float64)))))
    write(folder/f'matrix-history-{step:07d}.json',rows)
    write(folder/f'branch-history-{step:07d}.json',[dict(block=j,main_value_weight_norm=float(b.mlp.w12.weight.detach()[86:150].norm()),extra_output_norm=float(b.mlp.w3.weight.detach()[:,64:].norm())) for j,b in enumerate(model.blocks)])
def prepare(root):
    environment(root,'training');parents();repo=Path(__file__).resolve().parents[3];assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    assert not root.exists();root.mkdir();shutil.copytree(Path(__file__).parent,root/'source');(root/'doc-validation').mkdir()
    write(root/'settings.json',dict(source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),arm=ARM,model_seed=0,steps=20000,training_env='475428fb',metrics_env='dc729a50',new_ODE_method=False))
    paths=[*Path(root/'source').rglob('*'),root/'settings.json']
    for parent in (COLD,NATIVE,BUDGET,GEOMETRY):paths += [parent/'sealed-runtime.json',*Path(parent/'source').rglob('*.py')]
    weights=list(Path('/Users/mr.z/ode-workspace-prep/image-bridge-inception/checkpoints').glob('*.pth'));assert len(weights)==1;paths += weights
    for parent,arm in ((NATIVE,'swiglu86'),(NATIVE,'ode_k1'),(COLD,'matched_silu64')):paths.append(parent/'output/runs'/arm/'step_0000000.pt')
    write(root/'bindings.json',{str(p.resolve()):sha(p) for p in sorted(set(paths)) if p.is_file() and p.suffix!='.pyc'});verify(root);print('SWIGLU_INIT_PREPARED',flush=True)
def initial_witness():
    c,n,b=parents();m=construct();s=c.construct('matched_silu64');silu0=torch.load(COLD/'output/runs/matched_silu64/step_0000000.pt',weights_only=False,map_location='cpu');n.restore_model(s,silu0)
    gen=torch.Generator().manual_seed(0);x=torch.randn(8,4,64,generator=gen);maximum=0.;routes=[]
    for j,(left,right) in enumerate(zip(m.blocks,s.blocks)):
        with torch.no_grad():a=left.mlp(x);z=right.mlp(x)
        torch.testing.assert_close(a,z,rtol=1e-5,atol=1e-5);maximum=max(maximum,float((a-z).abs().max()))
        f=copy.deepcopy(left.mlp).double();r=copy.deepcopy(right.mlp).double();xx=x.double().requires_grad_(True);pattern=torch.cos(torch.arange(a.numel(),dtype=torch.float64)).reshape_as(a)
        one=(f(xx)*pattern).sum();two=(r(xx)*pattern).sum();ga=torch.autograd.grad(one,(xx,f.w12.weight,f.w3.weight));gb=torch.autograd.grad(two,(xx,r.fc1.weight,r.fc2.weight))
        for a,z in ((ga[0],gb[0]),(ga[1][:64],gb[1]),(ga[2][:,:64],gb[2])):torch.testing.assert_close(a,z,rtol=3e-7,atol=3e-9)
        assert float(ga[1][86:150].norm())>1e-8 and float(ga[2][:,64:].norm())>1e-8
        assert torch.equal(ga[1][64:86],torch.zeros_like(ga[1][64:86])) and torch.equal(ga[1][150:],torch.zeros_like(ga[1][150:]))
        opt=torch.optim.AdamW(f.parameters(),lr=1e-4,weight_decay=0.);opt.zero_grad();(f(xx.detach())*pattern).sum().backward();opt.step();opt.zero_grad();(f(xx.detach())*pattern).sum().backward()
        assert float(f.w12.weight.grad[64:86].norm())>1e-10 and float(f.w12.weight.grad[150:].norm())>1e-10
        routes.append(dict(block=j,main_value_route_nonzero=True,extra_output_route_nonzero=True,extra_gate_value_open_after_update=True))
    # An active inherited backbone exposes the initial-function/shared-gradient identity.
    active=c.native_state(20000)['model']
    for model in (m,s):
        state=model.state_dict()
        for name,value in active.items():
            if shared(name):state[name].copy_(value)
    sys.path.insert(0,str(GEOMETRY/'source'));gm=module('geometry_math',GEOMETRY/'source/geometry_math.py')
    models=[gm.literal_model(v) for v in (m,s)]
    for model,source in zip(models,(m,s)):
        flags=dict(source.named_parameters())
        for name,value in model.named_parameters():value.requires_grad_(flags[name].requires_grad)
    val=n.bank(NATIVE/'banks/validation.npz');xx=torch.from_numpy(val['x'][:2]).double();tt=torch.from_numpy(val['t'][:2]).double();labels=torch.zeros(2,dtype=torch.long);pattern=torch.cos(torch.arange(2*3*32*32,dtype=torch.float64)).reshape(2,3,32,32)
    output=[v(xx,tt,labels) for v in models];torch.testing.assert_close(*output,rtol=3e-7,atol=3e-9)
    names=[name for name,value in models[0].named_parameters() if value.requires_grad and shared(name)];grad=[]
    for model,y in zip(models,output):grad.append(torch.autograd.grad((y*pattern).sum(),[dict(model.named_parameters())[name] for name in names]))
    for a,z in zip(*grad):torch.testing.assert_close(a,z,rtol=3e-7,atol=3e-9)
    return dict(initial_FFN_float32_max_error=maximum,literal_input_main_parameter_derivatives=True,shared_active_backbone_gradients=len(names),extra_routes=routes,parameter_tangent_spaces_equal=False,active_probe_used_as_formal_start=False)
def preflight(root):
    environment(root,'training');verify(root);assert not (root/'preflight.json').exists();c,n,b=parents();witness=initial_witness();models=[];val=n.bank(NATIVE/'banks/validation.npz')
    for arm in ('ode_k1','swiglu86','swiglu129','silu131'):
        model=b.construct(arm);state=torch.load(BUDGET/'output/runs'/arm/'step_0020000.pt',weights_only=False,map_location='cpu');n.restore_model(model,state);model.eval();prediction=n.predict(model,{k:v[:64] for k,v in val.items()})
        with np.load(BUDGET/'output/runs'/arm/'validation_0020000.npz') as f:assert prediction.tobytes()==f['prediction'][:64].tobytes()
        models.append(arm)
    images,ids=n.data_for(NATIVE);ends=[];work=root/'resume-preflight';work.mkdir()
    for resume in (False,True):
        model,opt,rng,digest=restored()
        for step in range(1,6):
            batch=n.draw(images,ids,rng);digest=n.fingerprint(digest,batch);n.update(model,opt,batch)
            if resume and step==2:
                torch.save(n.checkpoint(model,opt,rng,step,digest),work/'boundary.pt');model,opt,rng,digest=restored(torch.load(work/'boundary.pt',weights_only=False))
        ends.append(n.checkpoint(model,opt,rng,5,digest))
    n.equal(*ends);sys.path.insert(0,str(NATIVE/'source'));import image_model
    cost=image_model.ffn_cost(construct());reference=image_model.ffn_cost(b.construct('swiglu86'));assert cost==reference and cost['trainable_parameters']==211412 and cost['dense_FFN_forward_MACs_per_token_per_block']==16512
    verify(root);write(root/'preflight.json',dict(status='PASS',initial_witness=witness,native_reference_replays=models,physical_five_update_resume=True,cost=cost,formal_training_started=False));print('SWIGLU_INIT_NATIVE_PREFLIGHT_PASS',flush=True)
def train(root):
    environment(root,'training');verify(root);c,_,_=parents();c.construct=construct;c.history=history;c.verify=verify;c.cell(root,ARM)
def metrics_modules(root):
    environment(root,'metrics');sys.path.insert(0,str(NATIVE/'source'));m=module('image_metrics',NATIVE/'source/image_metrics.py');core=module('metric_core',NATIVE/'source/metric_core.py');return m,core
def metric_preflight(root):
    verify(root);assert not (root/'metric-preflight.json').exists();m,core=metrics_modules(root);c,_,_=parents();raw=np.load(BUDGET/'output/runs/ode_k1/generated-0020000.npy');ex=m.extractor(NATIVE)
    actual=m.features(ex,core.uint8_images(raw[:32]));saved=np.load(c.feature_path(COLD,'ode_k1',20000));assert actual.tobytes()==saved[:32].tobytes()
    value=core.fid(saved,np.load(NATIVE/'metric-inputs/reference-features.npy'));expected=next(v['fid'] for v in load(COLD/'metric-output/result.json')['rows'] if v['arm']=='ode_k1' and v['step']==20000)
    assert abs(value-expected)<1e-8;write(root/'metric-preflight.json',dict(status='PASS',historical_feature_rows_bitwise=32,historical_FID=value));print('SWIGLU_INIT_METRIC_PREFLIGHT_PASS',flush=True)
def sample(root):
    environment(root,'training');verify(root);assert load(root/'audit-training/result.json')['status']=='PASS';c,_,_=parents();out=root/'sampling';out.mkdir();rows=[];tick=time.monotonic()
    for step in SAMPLES:
        model=model_at(root,step);start=time.monotonic()
        def progress(count):write(out/'progress.json',dict(step=step,images=count,completed=len(rows),total=3,seconds=time.monotonic()-tick))
        samples=c.imported('resolution').integrate(model,np.load(NATIVE/'banks/sampling-source.npy'),128,progress);np.save(out/f'{step:07d}.npy',samples)
        assert time.monotonic()-tick<2700;rows.append(dict(step=step,seconds=time.monotonic()-start,out_of_range_fraction=float(np.mean(abs(samples)>1))))
    verify(root);write(out/'result.json',dict(status='PASS',rows=rows));print('SWIGLU_INIT_SAMPLES_COMPLETE',flush=True)
def evaluate(root):
    verify(root);assert load(root/'audit-sampling/result.json')['status']=='PASS';m,core=metrics_modules(root);out=root/'metric-output';out.mkdir();reference=np.load(NATIVE/'metric-inputs/reference-features.npy');ex=m.extractor(NATIVE);rows=[]
    records=[json.loads(v) for v in (root/'output/runs'/ARM/'metrics.jsonl').read_text().splitlines()]
    for step in SAMPLES:
        f=m.features(ex,core.uint8_images(np.load(root/'sampling'/f'{step:07d}.npy')),progress=f'{ARM}-{step}');np.save(out/f'{step:07d}-features.npy',f)
        row=next(v for v in records if v['step']==step);rows.append(dict(step=step,weighted_mse=row['weighted_mse'],clean_mse=row['clean_mse'],fid=core.fid(f,reference)))
    verify(root);write(out/'result.json',dict(status='PASS',rows=rows,historical_rows=load(COLD/'metric-output/result.json')['rows']));print('SWIGLU_INIT_METRICS_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','preflight','metric-preflight','train','audit-training','sample','audit-sampling','evaluate','audit-metrics','run'));p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();action=a.action.replace('-','_')
    if action.startswith('audit_'):
        import init_audit;getattr(init_audit,action)(root)
    elif action=='run':
        from init_controller import run;run(root)
    else:globals()[action](root)
