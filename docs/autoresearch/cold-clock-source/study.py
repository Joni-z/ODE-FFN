"""Paired cold feature/time clocks on the immutable image training protocol."""
from __future__ import annotations
import os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','PYTHONDONTWRITEBYTECODE'):os.environ[key]='1'
sys.dont_write_bytecode=True
import argparse,copy,csv,hashlib,importlib.util,json,shutil,subprocess,time,types
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
if __name__=='__main__':sys.modules['clock_source_study']=sys.modules[__name__]
BASE=Path('/Users/mr.z/ode-workspace-prep/autoresearch');PARENT=BASE/'oct29-image-swiglu-init-v1';COORD=BASE/'oct29-p-coordinate-steps-v1'
ARMS=('feature_clock_zero','time_clock_zero');STEPS=(0,1000,2500,5000,10000,15000,20000);SAMPLES=tuple((a,s) for a in ARMS for s in (5000,10000,20000))
TRAIN='/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python';METRICS='/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python'
load=lambda p:json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def append(p,v):
    with Path(p).open('a') as f:f.write(json.dumps(v,allow_nan=False)+'\n')
def module(name,path):
    path=Path(path).resolve()
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    assert Path(sys.modules[name].__file__).resolve()==path;return sys.modules[name]
def parents():
    support=module('clock_source_support',PARENT/'source/init_study.py');c,n,b=support.parents();return support,c,n,b
def environment(root,kind):parents()[0].environment(root,kind)
def verify(root):
    support,_,_,_=parents();support.verify(root)
    for parent,digest in ((PARENT,'4a6af4a2fc61ca8a1f60d145014041f52f23379b1cb2ec454d1f12da744927cd'),(COORD,'4402f9bbde7c1cbcd0c1e0cc3b85164d21f3d699d7e21fca2257b4e36e4b2103')):
        assert sha(parent/'sealed-runtime.json')==digest
        for name,h in load(parent/'sealed-runtime.json').items():assert sha(parent/name)==h,(parent,name)
def old():return parents()[2]
def bank(t=None):return parents()[1].bank(t)
def native_state(step):return parents()[1].native_state(step)
def step_value(layer,inputs):return torch.sigmoid(layer.t_func(inputs)/layer.tau)*layer.scale+layer.shift
def time_forward(self,x,cond=None,return_aux=False):
    assert cond is not None and cond.shape==(len(x),64)
    a=step_value(self.ode1,cond)[:,None,:];one=sum([x,(self.ode1.proj(x)*a)/1])
    two=self.act(one);b=step_value(self.ode2,cond)[:,None,:];out=self.drop(sum([two,(self.ode2.proj(two)*b)/1]))
    return self._maybe_return(out,self._aux(x,out),return_aux)
def independent_forward(self,x,cond=None,return_aux=False):
    source=cond if self.clock_source=='time' else x
    a=step_value(self.ode1,source)
    if self.clock_source=='time':a=a[:,None,:]
    one=(0+x)+(F.linear(x,self.ode1.proj.weight)*a)/1;two=F.silu(one)
    source=cond if self.clock_source=='time' else two;b=step_value(self.ode2,source)
    if self.clock_source=='time':b=b[:,None,:]
    out=self.drop((0+two)+(F.linear(two,self.ode2.proj.weight)*b)/1)
    return self._maybe_return(out,self._aux(x,out),return_aux)
def construct(arm,native=None,promote=False,independent=False):
    assert arm in ARMS and not promote
    _,c,n,b=parents();model=b.construct('ode_k1');n.restore_model(model,c.native_state(0))
    with torch.no_grad():
        for block in model.blocks:
            for k in (1,2):
                layer=getattr(block.mlp,f'ode{k}');assert layer.orders==1 and layer.t_from_embed is None
                assert (layer.tau,layer.scale,layer.shift)==(10.,.8,.2)
                layer.t_func[-1].weight.zero_();layer.t_func[-1].bias.zero_()
            block.mlp.clock_source='time' if arm=='time_clock_zero' else 'feature'
            if arm=='time_clock_zero':block.mlp.forward=types.MethodType(time_forward,block.mlp)
            if independent:block.mlp.forward=types.MethodType(independent_forward,block.mlp)
    if native is not None:n.restore_model(model,native)
    return model
def restored(arm,state=None,independent=False):
    m=construct(arm,independent=independent);n=old();opt=n.optimizer(m);rng=torch.Generator()
    if state is None:
        initial=native_state(0);rng.set_state(initial['data_rng']);torch.set_rng_state(initial['torch_rng']);digest=initial['input_stream_digest']
    else:digest=n.restore(m,opt,rng,copy.deepcopy(state))
    return m,opt,rng,digest
def matrices(model,arm):
    result={}
    for j,block in enumerate(model.blocks):
        for k in (1,2):
            layer=getattr(block.mlp,f'ode{k}');result[f'block{j}.P{k}']=layer.proj.weight.detach().numpy().copy();result[f'block{j}.clock_head{k}']=layer.t_func[-1].weight.detach().numpy().copy()
    return result
def clock_profiles(model):
    rows=[];hooks=[];time_value=None
    for j,block in enumerate(model.blocks):
        for k in (1,2):
            layer=getattr(block.mlp,f'ode{k}')
            def capture(module,inputs,out,layer=layer,j=j,k=k):
                a=(torch.sigmoid(out/layer.tau)*layer.scale+layer.shift).detach().double().numpy()
                rows.append(dict(time=time_value,layer=f'block{j}.clock{k}',mean=float(a.mean()),std=float(a.std()),minimum=float(a.min()),maximum=float(a.max()),values=a.size))
            hooks.append(layer.t_func.register_forward_hook(capture))
    try:
        for time_value in (.1,.3,.5,.7,.9):old().predict(model,bank(time_value))
    finally:
        for h in hooks:h.remove()
    assert len(rows)==30
    if model.blocks[0].mlp.clock_source=='time':assert all(v['minimum']==v['maximum'] for v in rows)
    return rows
def history(model,arm,folder,step):
    arrays=matrices(model,arm);np.savez(folder/f'matrices-{step:07d}.npz',**arrays);rows=[]
    for name,value in arrays.items():
        a=value.astype(np.float64);s=np.linalg.svd(a,compute_uv=False);row=dict(layer=name,singular_values=s.tolist(),frobenius=float(np.linalg.norm(a)))
        if '.P' in name:
            e=np.linalg.eigvals(a);row.update(eigen_real=e.real.tolist(),eigen_imag=e.imag.tolist())
        rows.append(row)
    write(folder/f'matrix-history-{step:07d}.json',rows);write(folder/f'clock-profile-{step:07d}.json',clock_profiles(model))
def model_at(root,arm,step,independent=False):
    state=torch.load(root/'output/runs'/arm/f'step_{step:07d}.pt',weights_only=False,map_location='cpu');m=construct(arm,independent=independent);old().restore_model(m,state);return m.eval().requires_grad_(False)
def prepare(root):
    repo=Path(__file__).resolve().parents[3];assert not root.exists() and not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    root.mkdir();shutil.copytree(Path(__file__).parent,root/'source');(root/'doc-validation').mkdir()
    write(root/'settings.json',dict(source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),arms=ARMS,steps=STEPS,model_seed=0,data_stream=100000,training_steps=20000,threads=1,workers=2))
    paths=[*Path(root/'source').rglob('*'),root/'settings.json',PARENT/'sealed-runtime.json',COORD/'sealed-runtime.json',PARENT/'curation/comparison.csv']
    weights=list(Path('/Users/mr.z/ode-workspace-prep/image-bridge-inception/checkpoints').glob('*.pth'));assert len(weights)==1;paths+=weights
    write(root/'bindings.json',{str(p.resolve()):sha(p) for p in paths if p.is_file() and p.suffix!='.pyc'});verify(root);print('COLD_CLOCK_SOURCE_PREPARED',flush=True)
def ordinary_time(ffn,x,cond):
    one=step_value(ffn.ode1,cond);two=step_value(ffn.ode2,cond);identity=torch.eye(64,dtype=x.dtype)
    w1=identity+one[:,:,None]*ffn.ode1.proj.weight;w2=identity+two[:,:,None]*ffn.ode2.proj.weight
    return torch.bmm(F.silu(torch.bmm(x,w1.transpose(1,2))),w2.transpose(1,2))
def ordinary_forward(self,x,cond=None,return_aux=False):
    out=ordinary_time(self,x,cond)
    return self._maybe_return(out,self._aux(x,out),return_aux)
def cost_witness():
    from torch.utils._python_dispatch import TorchDispatchMode
    class Count(TorchDispatchMode):
        def __init__(self):self.macs=0
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            out=func(*args,**(kwargs or {}))
            if func==torch.ops.aten.mm.default:self.macs+=args[0].shape[0]*args[0].shape[1]*args[1].shape[1]
            elif func==torch.ops.aten.bmm.default:self.macs+=args[0].shape[0]*args[0].shape[1]*args[0].shape[2]*args[1].shape[2]
            elif func==torch.ops.aten.addmm.default:self.macs+=args[1].shape[0]*args[1].shape[1]*args[2].shape[1]
            return out
    rows=[]
    for arm in ARMS:
        m=construct(arm);x=torch.randn(2,64,64,requires_grad=True);cond=torch.randn(2,64,requires_grad=True)
        with Count() as fw:m.blocks[0].mlp(x,cond)
        with Count() as both:m.blocks[0].mlp(x,cond).square().sum().backward()
        forward=fw.macs//128;total=both.macs//128;expected=16512 if arm==ARMS[0] else 8322
        assert forward==expected and total==3*expected
        count=sum(p.numel() for p in m.parameters() if p.requires_grad);assert count==211094
        rows.append(dict(arm=arm,trainable_parameters=count,FFN_forward_MACs=forward,FFN_forward_backward_MACs=total,scope='64patches/image; dense matrix calls; pointwise/backbone excluded'))
    return rows
def preflight(root):
    environment(root,'training');verify(root);assert not (root/'preflight.json').exists();support,c,n,b=parents();a=construct(ARMS[0]);z=construct(ARMS[1]);n.equal(a.state_dict(),z.state_dict())
    for (_,v),(_,w) in zip(a.named_buffers(),z.named_buffers()):assert torch.equal(v,w)
    g=torch.Generator().manual_seed(0);x=torch.randn(8,4,64,generator=g);cond=torch.randn(8,64,generator=g)
    for aa,bb in zip(a.blocks,z.blocks):assert torch.equal(aa.mlp(x,cond),bb.mlp(x,cond))
    gm=module('clock_source_literal_math',support.GEOMETRY/'source/geometry_math.py');active=c.native_state(20000)
    for model in (a,z):
        values=model.state_dict()
        for name,v in active['model'].items():
            if not (name.startswith('blocks.') and '.mlp.' in name):values[name].copy_(v)
    promoted=[gm.literal_model(m) for m in (a,z)]
    for m,original in zip(promoted,(a,z)):
        flags=dict(original.named_parameters())
        for name,v in m.named_parameters():v.requires_grad_(flags[name].requires_grad)
    bank2={k:v[:2] for k,v in c.bank().items()};xx=torch.from_numpy(bank2['x']).double();tt=torch.from_numpy(bank2['t']);labels=torch.zeros(2,dtype=torch.long);pattern=torch.cos(torch.arange(2*3*32*32,dtype=torch.float64)).reshape(2,3,32,32)
    outputs=[m(xx,tt,labels) for m in promoted];torch.testing.assert_close(*outputs,rtol=1e-8,atol=1e-10)
    selected=[name for name,p in promoted[0].named_parameters() if p.requires_grad and '.t_func.' not in name];grad=[]
    for m,y in zip(promoted,outputs):grad.append(torch.autograd.grad((y*pattern).sum(),[dict(m.named_parameters())[name] for name in selected]))
    for v,w in zip(*grad):torch.testing.assert_close(v,w,rtol=1e-8,atol=1e-10)
    # Active nonzero clock weights expose the complete conditional derivative.
    model=construct(ARMS[1],active);literal=gm.literal_model(model);flags=dict(model.named_parameters())
    for name,p in literal.named_parameters():p.requires_grad_(flags[name].requires_grad)
    folded=copy.deepcopy(literal)
    for block in folded.blocks:block.mlp.forward=types.MethodType(ordinary_forward,block.mlp)
    output=literal(xx,tt,labels);other=folded(xx,tt,labels);torch.testing.assert_close(output,other,rtol=1e-8,atol=1e-10)
    all_names=[name for name,p in literal.named_parameters() if p.requires_grad];all_gradients=[]
    for m,y in ((literal,output),(folded,other)):all_gradients.append(torch.autograd.grad((y*pattern).sum(),[dict(m.named_parameters())[name] for name in all_names]))
    for v,w in zip(*all_gradients):torch.testing.assert_close(v,w,rtol=1e-8,atol=1e-10)
    output=literal(xx,tt,labels);loss=(output*pattern).sum();names=('t_embedder.mlp.0.weight','blocks.0.mlp.ode1.proj.weight','blocks.0.mlp.ode2.t_func.2.weight');parameters=dict(literal.named_parameters());observed=torch.autograd.grad(loss,[parameters[name] for name in names]);fd=[]
    for name,derivative in zip(names,observed):
        p=parameters[name];index=int(derivative.detach().abs().flatten().argmax());original=float(p.flatten()[index].detach());values=[]
        for sign in (-1,1):
            with torch.no_grad():p.flatten()[index]=original+sign*1e-5;values.append(float((literal(xx,tt,labels)*pattern).sum()))
        with torch.no_grad():p.flatten()[index]=original
        expected=(values[1]-values[0])/2e-5;actual=float(derivative.flatten()[index]);assert abs(actual)>1e-8;np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-8);fd.append(dict(parameter=name,flat_index=index,AD=actual,finite_difference=expected))
    ffn=copy.deepcopy(model.blocks[0].mlp).double();xc=x.double().requires_grad_(True);cc=cond.double().requires_grad_(True);yy=ffn(xc,cc);zz=ordinary_time(ffn,xc,cc);torch.testing.assert_close(yy,zz,rtol=1e-10,atol=1e-10);probe=torch.cos(torch.arange(yy.numel(),dtype=torch.float64)).reshape_as(yy);wrt=[xc,cc,*ffn.parameters()]
    first=torch.autograd.grad((yy*probe).sum(),wrt);second=torch.autograd.grad((zz*probe).sum(),wrt)
    for v,w in zip(first,second):torch.testing.assert_close(v,w,rtol=1e-9,atol=1e-10)
    refs=[]
    for arm in ('ode_k1','swiglu86','swiglu129','silu131'):
        m=b.construct(arm);state=torch.load(support.BUDGET/'output/runs'/arm/'step_0020000.pt',weights_only=False,map_location='cpu');n.restore_model(m,state);m.eval();prediction=n.predict(m,{k:v[:64] for k,v in c.bank().items()})
        with np.load(support.BUDGET/'output/runs'/arm/'validation_0020000.npz') as saved:assert prediction.tobytes()==saved['prediction'][:64].tobytes()
        refs.append(arm)
    images,ids=n.data_for(support.NATIVE);resume=[];work=root/'resume-preflight';work.mkdir()
    for arm in ARMS:
        ends=[]
        for physical in (False,True):
            m,opt,rng,digest=restored(arm)
            for step in range(1,6):
                batch=n.draw(images,ids,rng);digest=n.fingerprint(digest,batch);n.update(m,opt,batch)
                if not physical and step==4:before=n.checkpoint(m,opt,rng,4,digest)
                if physical and step==2:
                    path=work/f'{arm}.pt';torch.save(n.checkpoint(m,opt,rng,step,digest),path);m,opt,rng,digest=restored(arm,torch.load(path,weights_only=False,map_location='cpu'))
            ends.append(n.checkpoint(m,opt,rng,5,digest))
        n.equal(*ends);m,opt,rng,digest=restored(arm,before,independent=True);batch=n.draw(images,ids,rng);digest=n.fingerprint(digest,batch);n.update(m,opt,batch);n.equal(n.checkpoint(m,opt,rng,5,digest),ends[0]);resume.append(dict(arm=arm,five_update_physical_resume_bitwise=True,independent_active_update_bitwise=True))
    costs=cost_witness();verify(root);write(root/'preflight.json',dict(status='PASS',paired_initial_state_bitwise=True,paired_initial_FFN_bitwise=True,active_shared_and_P_gradients=len(selected),conditional_matrix_derivative_tensors=len(wrt),active_full_model_parameter_gradients=len(all_names),full_model_FD=fd,native_prefixes=refs,resumes=resume,costs=costs,formal_training_started=False));print('COLD_CLOCK_SOURCE_NATIVE_PASS',flush=True)
def metrics_modules(root):
    environment(root,'metrics');support,_,_,_=parents();sys.path.insert(0,str(support.NATIVE/'source'));return module('image_metrics',support.NATIVE/'source/image_metrics.py'),module('metric_core',support.NATIVE/'source/metric_core.py')
def metric_preflight(root):
    verify(root);assert not (root/'metric-preflight.json').exists();m,core=metrics_modules(root);support,c,_,_=parents();raw=np.load(support.BUDGET/'output/runs/ode_k1/generated-0020000.npy');saved=np.load(c.feature_path(support.COLD,'ode_k1',20000));actual=m.features(m.extractor(support.NATIVE),core.uint8_images(raw[:32]));assert actual.tobytes()==saved[:32].tobytes();fid=core.fid(saved,np.load(support.NATIVE/'metric-inputs/reference-features.npy'));assert abs(fid-173.43691430770065)<1e-8;write(root/'metric-preflight.json',dict(status='PASS',cached_feature_rows_bitwise=32,native_FID=fid));print('COLD_CLOCK_SOURCE_METRIC_PASS',flush=True)
def cell(root,arm):
    environment(root,'training');verify(root);_,c,_,_=parents();c.construct=construct;c.history=history;c.verify=verify;c.cell(root,arm)
def sample(root):
    environment(root,'training');verify(root);assert load(root/'audit-training/result.json')['status']=='PASS';support,c,_,_=parents();out=root/'sampling';out.mkdir();rows=[]
    for arm,step in SAMPLES:
        tick=time.monotonic();m=model_at(root,arm,step)
        def progress(count):write(out/'progress.json',dict(completed=len(rows),total=6,arm=arm,step=step,images=count))
        raw=c.imported('resolution').integrate(m,np.load(support.NATIVE/'banks/sampling-source.npy'),128,progress);np.save(out/f'{arm}-{step:07d}.npy',raw);rows.append(dict(arm=arm,step=step,seconds=time.monotonic()-tick,raw_tail_fraction=float(np.mean(abs(raw)>1))))
    verify(root);write(out/'result.json',dict(status='PASS',rows=rows));print('COLD_CLOCK_SOURCE_SAMPLES_COMPLETE',flush=True)
def historical_rows():
    values=list(csv.DictReader((PARENT/'curation/comparison.csv').open()));assert len(values)==35;return values
def evaluate(root):
    verify(root);assert load(root/'audit-sampling/result.json')['status']=='PASS';m,core=metrics_modules(root);support,_,_,_=parents();out=root/'metric-output';out.mkdir();ex=m.extractor(support.NATIVE);reference=np.load(support.NATIVE/'metric-inputs/reference-features.npy');rows=[]
    for arm,step in SAMPLES:
        raw=np.load(root/'sampling'/f'{arm}-{step:07d}.npy');features=m.features(ex,core.uint8_images(raw),progress=f'{arm}-{step}');np.save(out/f'{arm}-{step:07d}-features.npy',features);r=next(json.loads(v) for v in (root/'output/runs'/arm/'metrics.jsonl').read_text().splitlines() if json.loads(v)['step']==step);rows.append(dict(arm=arm,step=step,weighted_mse=r['weighted_mse'],clean_mse=r['clean_mse'],fid=core.fid(features,reference),parameters=211094,FFN_forward_MACs=16512 if arm==ARMS[0] else 8322))
    oldrows=historical_rows();controls=[dict(arm=v['arm'],fid=float(v['fid']),weighted_mse=float(v['weighted_mse']),clean_mse=float(v['clean_mse'])) for v in oldrows if int(v['step'])==20000];controls.append(next(v for v in rows if v['arm']==ARMS[0] and v['step']==20000));best=min(controls,key=lambda v:v['fid']);candidate=next(v for v in rows if v['arm']==ARMS[1] and v['step']==20000);checks=dict(FID=candidate['fid']<=.95*best['fid'],weighted_mse=candidate['weighted_mse']<=1.05*best['weighted_mse'],clean_mse=candidate['clean_mse']<=1.05*best['clean_mse'])
    verify(root);write(out/'result.json',dict(status='PASS',rows=rows,historical_rows=oldrows));write(out/'decision.json',dict(best_control=best,candidate=candidate,checks=checks,quality_pass=all(checks.values()),new_ODE_method_established=False));print('COLD_CLOCK_SOURCE_METRICS_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','preflight','metric-preflight','cell','audit-training','sample','audit-sampling','evaluate','audit-metrics','run'));p.add_argument('--root',type=Path,required=True);p.add_argument('--arm',choices=ARMS);args=p.parse_args();root=args.root.resolve();action=args.action.replace('-','_')
    if action.startswith('audit_'):getattr(module('clock_source_audit',root/'source/audit_family.py'),action)(root)
    elif action=='run':module('clock_source_controller',root/'source/controller.py').run(root)
    elif action=='cell':cell(root,args.arm)
    else:globals()[action](root)
