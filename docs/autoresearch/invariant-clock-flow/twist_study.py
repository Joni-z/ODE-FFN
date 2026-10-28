"""Source-bound old-law screen, reusing the immutable native training loop."""
from __future__ import annotations
import os,sys
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','PYTHONDONTWRITEBYTECODE'):os.environ[key]='1'
sys.dont_write_bytecode=True
import argparse,copy,hashlib,importlib.metadata,importlib.util,json,shutil,subprocess,threading,time,traceback
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import psutil
import torch
import invariant_clocks as operators
from invariant_clocks import InvariantClockFFN,InvariantClockLayer,IdentitySiLU
if __name__=='__main__':sys.modules['twist_study']=sys.modules[__name__]

BASE=Path('/Users/mr.z/ode-workspace-prep/autoresearch');PARENT=BASE/'oct27-cycle04'
ARMS=('rotation','euler','normalized_euler','silu_identity','fixed06_identity','swiglu129','native_k1')
LAWS=('rotated_product_bimodal_16d','sign_code','half_mixture');STEPS=(0,1000,2500,5000)
HISTORY=('ode_k2','free_isru','swiglu64','swiglu128','swiglu160')
PYTHON='/Users/mr.z/ode-workspace-prep/.venv-test/bin/python'
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
def env_hash(spec):return hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:8]
def setup(root):
    s=load(root/'settings.json');spec=load(root/'source/synthetic-env-spec.json');assert env_hash(spec)==s['env_spec']
    for group in spec['pip_phases']:
        for requirement in group:
            name,version=requirement.split('==');assert importlib.metadata.version(name)==version
    assert Path(operators.__file__).resolve()==root/'source/invariant_clocks.py'
    sys.path.insert(0,s['joint_source']);import joint_study as n;n.setup(root)
    assert Path(n.__file__).resolve()==Path(s['joint_source'])/'joint_study.py'
    assert Path(n.legacy.__file__).resolve()==Path(s['native_root'])/'scripts/run_recurrent_depth.py'
    torch.set_num_threads(1);return s,n
def entry(root,law):return next(v for v in load(root/'settings.json')['laws'] if v['law']==law)
def verify(root):
    for path,digest in load(root/'bindings.json').items():assert sha(path)==digest,path
    for seal in load(root/'settings.json')['prior_seals']:
        seal=Path(seal)
        for name,digest in load(seal).items():assert sha(seal.parent/name)==digest,(seal,name)
def train_bank(e,n):
    p=Path(e['train']);return torch.from_numpy(np.load(p) if p.suffix=='.npy' else n.legacy.load_targets(p,16))
def build(root,law,arm):
    _,n=setup(root);e=entry(root,law);model=n.old_build(16,'swiglu128','direct')
    cold=torch.load(Path(e['references']['swiglu128'])/'step_0000000.pt',weights_only=False);n.restore_model(model,cold)
    torch.manual_seed(0)
    if arm in ('rotation','euler','normalized_euler','silu_identity','fixed06_identity'):
        for block in model.raw.blocks:
            block.ffn=InvariantClockFFN(kind=arm) if arm in ARMS[:3] else IdentitySiLU(kind=arm)
    elif arm=='swiglu129':
        raw=n.VelocityNet(16,'swiglu',width=64,depth=3,ffn_hidden=129);n.copy_surrounding(raw,model.raw);model.raw=raw
    elif arm=='native_k1':
        raw=n.old_build(16,'ode_k2','direct').raw;n.copy_surrounding(raw,model.raw)
        for block in raw.blocks:block.ffn.ode1.orders=block.ffn.ode2.orders=1
        model.raw=raw
    else:raise ValueError(arm)
    torch.set_rng_state(cold['torch_rng']);return model
def historical(root,law,family):
    _,n=setup(root);model=n.build(family);folder=Path(entry(root,law)['references'][family])
    state=torch.load(folder/'step_0005000.pt',weights_only=False);n.restore_model(model,state);return model,folder,state
def prepare(root):
    repo=Path(__file__).resolve().parents[3];assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True)
    assert not root.exists();root.mkdir();shutil.copytree(Path(__file__).parent,root/'source');(root/'doc-validation').mkdir()
    s=load(PARENT/'settings.json');s['laws']=[v for v in s['laws'] if v['law'] in LAWS]
    cohort=load(Path(s['reference_root'])/'audit-cohort.json')
    for e in s['laws']:
        if e['law']==LAWS[0]:e['references']={r['family']:r['folder'] for r in cohort if r['dataset']==e['law'] and r['family'] in HISTORY}
        else:
            parent=Path(s['old_joint_root']) if e['law']=='sign_code' else BASE/'oct27-cycle02'
            e['references']={family:str(parent/'output/runs'/family) for family in HISTORY}
        assert set(e['references'])==set(HISTORY)
        for folder in e['references'].values():
            for name in ('result.json','diagnostics/metrics.json','diagnostics/field.npz'):assert (Path(folder)/name).exists(),(folder,name)
    s.update(source_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),arms=ARMS,model_seed=0,training_stream=100000,
        env_spec=env_hash(load(root/'source/synthetic-env-spec.json')),prior_seals=[*s['prior_seals'],str(PARENT/'sealed-runtime.json'),
        str(BASE/'oct28-image-clock-training-v2/sealed-runtime.json'),str(BASE/'oct28-image-gaussian-reference/sealed-runtime.json'),
        str(BASE/'oct28-posterior-state-eligibility/sealed-runtime.json'),str(BASE/'oct28-invariant-clock-operator/sealed-runtime.json')])
    write(root/'settings.json',s)
    paths=[*(root/'source').rglob('*'),root/'settings.json',*(Path(s['native_root']).rglob('*.py')),*(Path(s['joint_source']).glob('*.py')),PARENT/'source/study.py']
    for e in s['laws']:
        paths += [Path(e['train']),*Path(e['banks']).glob('*.npz')]
        for folder in e['references'].values():paths+=list(Path(folder).rglob('*'))
    paths += [Path(v) for v in s['prior_seals']]
    write(root/'bindings.json',{str(p.resolve()):sha(p) for p in sorted(set(paths)) if p.is_file() and p.suffix!='.pyc'})
    verify(root);print(json.dumps(dict(status='PREPARED',cells=21,historical_endpoints=15,source_commit=s['source_commit'])),flush=True)
def matrix_values(model):
    result={};kinds={}
    for j,block in enumerate(model.raw.blocks):
        f=block.ffn
        for k in (1,2):
            if isinstance(f,InvariantClockFFN):value=getattr(f,'ode'+str(k)).generator();kind='skew_generator_P'
            elif isinstance(f,IdentitySiLU):value=getattr(f,'matrix'+str(k));kind='shifted_P' if f.kind=='fixed06_identity' else 'ordinary_W'
            elif hasattr(f,'ode1'):value=getattr(f,'ode'+str(k)).proj.weight;kind='native_P'
            else:continue
            name=f'block{j}.matrix{k}';result[name]=value.detach().cpu().numpy().copy();kinds[name]=kind
    return result,kinds
@torch.no_grad()
def visits(model,banks,folder):
    step=load(folder/'status.json')['step'];assert step in STEPS
    matrices,kinds=matrix_values(model);np.savez_compressed(folder/f'matrices_{step:07d}.npz',**matrices)
    rows=[];hooks=[]
    for name,layer in model.named_modules():
        if not isinstance(layer,InvariantClockLayer):continue
        def capture(layer,args,name=name):
            x=args[0].detach().clone();before=layer.invariants(x);after=layer.invariants(InvariantClockLayer.forward(layer,x));rates=layer.rates(x);clocks=layer.clocks(before)
            expected=before*(1+rates.square()) if layer.kind=='euler' else before
            torch.testing.assert_close(after,expected,rtol=3e-5,atol=3e-6)
            promoted=copy.deepcopy(layer).double();gains=[]
            for anchor in x[:2].double():gains.append(float(torch.linalg.svdvals(promoted.complete_jacobian(anchor))[0]))
            rows.append(dict(layer=name,kind=layer.kind,input_count=len(x),action_max_absolute_change=float((after-before).abs().max()),
                clock_mean=float(clocks.mean()),clock_std=float(clocks.std(unbiased=False)),clock_across_example_std=float(clocks.std(0,unbiased=False).mean()),
                rate_max_absolute=float(rates.abs().max()),promoted_complete_map_gains=gains))
        hooks.append(layer.register_forward_pre_hook(capture))
    try:
        with np.load(banks/'probe.npz') as b:model(torch.from_numpy(b['x'][:32]),torch.from_numpy(b['t'][:32]))
    finally:
        for hook in hooks:hook.remove()
    summaries=[]
    for name,a in matrices.items():
        singular=np.linalg.svd(a.astype(np.float64),compute_uv=False);eigen=np.linalg.eigvals(a.astype(np.float64))
        summaries.append(dict(layer=name,kind=kinds[name],sigma_max=float(singular.max()),sigma_min=float(singular.min()),
            singular_CV=float(singular.std()/singular.mean()) if singular.mean()>0 else None,eigen_max_real=float(eigen.real.max()),eigen_max_absolute=float(np.abs(eigen).max())))
    return dict(matrices=summaries,flow_inputs=rows)
def shared(name):return not (name.startswith('raw.blocks.') and '.ffn.' in name)
def preflight(root):
    s,n=setup(root);verify(root);work=root/'preflight-work';work.mkdir();tick=time.monotonic()
    from verify_operator import verify as verify_operator
    from manual_jacobian import network_jacobian
    mathematics=verify_operator();rows=[];references=[];active_routes=[];paired=[];tangents=[];chain=[]
    for e in s['laws']:
        law=e['law'];bank=train_bank(e,n);assert bank.shape==(100000,16)
        with np.load(Path(e['banks'])/'field.npz') as b:x=torch.from_numpy(b['x'][:8]);t=torch.from_numpy(b['t'][:8]);assert b['x'].shape==(20000,16)
        baseline=build(root,law,'rotation');old=torch.load(Path(e['references']['swiglu128'])/'step_0000000.pt',weights_only=False)
        for name,value in baseline.state_dict().items():
            if shared(name):assert torch.equal(value,old['model'][name])
        head=torch.randn(baseline.raw.output.weight.shape,generator=torch.Generator().manual_seed(0))*.02
        assert torch.linalg.matrix_rank(head)==16
        with torch.no_grad():baseline.raw.output.weight.copy_(head)
        names=[k for k,_ in baseline.named_parameters() if shared(k)]
        base_output=baseline(x,t);base_grad=torch.autograd.grad(base_output.square().sum(),[dict(baseline.named_parameters())[k] for k in names])
        flow_names=[k for k,_ in baseline.named_parameters() if '.ffn.' in k]
        probe_weights=torch.cos(torch.arange(16)+.4)
        base_tangent=torch.autograd.grad((baseline(x,t)*probe_weights).sum(),[dict(baseline.named_parameters())[k] for k in flow_names])
        for arm in ARMS:
            model=build(root,law,arm)
            for name,value in model.state_dict().items():
                if shared(name):assert torch.equal(value,old['model'][name])
            for name,value in model.named_buffers():assert torch.equal(value,old['runtime_buffers'][name])
            if arm in ARMS[:5]:
                with torch.no_grad():model.raw.output.weight.copy_(head)
                torch.testing.assert_close(model(x,t),base_output,rtol=2e-5,atol=2e-6)
                gradients=torch.autograd.grad(model(x,t).square().sum(),[dict(model.named_parameters())[k] for k in names])
                for a,b in zip(gradients,base_grad):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
                paired.append(dict(law=law,arm=arm,shared_parameter_gradients=len(names),nonzero_head=True,trainable_tangent_spaces_equal=arm in ARMS[:3]))
            if arm in ARMS[:3]:
                actual_tangent=torch.autograd.grad((model(x,t)*probe_weights).sum(),[dict(model.named_parameters())[k] for k in flow_names])
                for name,a,b in zip(flow_names,actual_tangent,base_tangent):
                    torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
                    assert (float(a.norm())>1e-12) if name.endswith('.frequency') else torch.equal(a,torch.zeros_like(a)),name
                tangents.append(dict(law=law,arm=arm,parameter_tensors=len(flow_names),initial_zero_clock_gradients=True,frequency_routes_nonzero=True))
                for module in model.modules():
                    if isinstance(module,InvariantClockLayer):
                        with torch.no_grad():module.frequency.copy_(.2*torch.cos(torch.arange(32)+.3))
                probe=(model(x,t)*torch.cos(torch.arange(16)+.4)).sum()
                params=[(k,v) for k,v in model.named_parameters() if '.ffn.' in k];gradient=torch.autograd.grad(probe,[v for _,v in params])
                assert all(float(v.norm())>1e-12 for v in gradient)
                active_routes.append(dict(law=law,arm=arm,parameter_tensors=len(params),minimum_gradient_norm=min(float(v.norm()) for v in gradient)))
            with torch.no_grad():model.raw.output.weight.copy_(head)
            promoted=copy.deepcopy(model).double().eval();anchor=x[0].double();time_anchor=t[0].double()
            explicit=network_jacobian(promoted,anchor,time_anchor)
            automatic=torch.autograd.functional.jacobian(lambda value:promoted(value[None],time_anchor[None])[0],anchor,vectorize=True)
            assert float(automatic.norm())>1e-8
            torch.testing.assert_close(explicit,automatic,rtol=3e-9,atol=3e-9)
            chain.append(dict(law=law,arm=arm,nonzero_head=True,max_absolute_error=float((explicit-automatic).abs().max())))
            ends=[]
            for physical_resume in (False,True):
                model=build(root,law,arm);opt=n.optimizer(model);rng=torch.Generator().manual_seed(100000);digest=n.EMPTY
                for step in range(1,4):
                    xx,tt,a,b=n.draw(bank,rng);digest=n.legacy.fingerprint_batch(digest,a,b,tt);n.update(model,opt,xx,tt,a,b)
                    if physical_resume and step==2:
                        path=work/f'{law}-{arm}-boundary.pt';torch.save(n.checkpoint(model,opt,rng,step,digest),path)
                        model=build(root,law,arm);opt=n.optimizer(model);digest=n.restore(model,opt,rng,torch.load(path,weights_only=False))
                ends.append(n.checkpoint(model,opt,rng,3,digest))
            n.equal(*ends);cost=n.compute_budget(model)
            expected=(84240,81984) if arm in ARMS[:3] else (84180,82368) if arm=='swiglu129' else (83862,82368) if arm=='native_k1' else (58512,57408)
            assert (cost['parameters'],cost['dense_forward_macs'])==expected,(arm,cost)
            rows.append(dict(law=law,arm=arm,physical_resume=True,**cost))
        for family in HISTORY:
            model,folder,_=historical(root,law,family);model.eval()
            with np.load(Path(e['banks'])/'field.npz') as b, np.load(folder/'diagnostics/field.npz') as old_prediction:
                with torch.no_grad():actual=model(torch.from_numpy(b['x'][:1024]),torch.from_numpy(b['t'][:1024])).numpy()
                assert actual.tobytes()==old_prediction['prediction'][:1024].tobytes(),(law,family)
            references.append(dict(law=law,family=family,replayed_native_predictions_bitwise=1024))
    verify(root);write(root/'preflight.json',dict(status='PASS',mathematics=mathematics,cells=rows,paired_initial_functions=paired,nonzero_frequency_routes=active_routes,
        initial_flow_parameter_tangents=tangents,manual_complete_network_derivatives=chain,historical_reference_checks=references,
        env_spec=s['env_spec'],seconds=time.monotonic()-tick,formal_training_started=False))
    print(json.dumps(dict(status='NATIVE_INVARIANT_CLOCK_PREFLIGHT_PASS',cells=len(rows),historic_endpoints=len(references),physical_resumes=21)),flush=True)
def cell(root,law,arm):
    _,n=setup(root);verify(root);base=module('invariant_registered_native_cell',PARENT/'source/study.py');folder=root/'output/runs'/law/arm
    timing=dict(draw_calls=0,draw_seconds=0.,update_calls=0,update_seconds=0.);old_draw,old_update=n.draw,n.update
    def draw(*args):
        tick=time.monotonic();answer=old_draw(*args);timing['draw_seconds']+=time.monotonic()-tick;timing['draw_calls']+=1;return answer
    def update(*args):
        tick=time.monotonic();answer=old_update(*args);timing['update_seconds']+=time.monotonic()-tick;timing['update_calls']+=1;return answer
    n.draw,n.update=draw,update;base.setup=lambda _: (load(root/'settings.json'),n,None,None);base.build=build;base.generator_visits=lambda model,banks:visits(model,banks,folder)
    base.cell(root,law,arm);assert timing['draw_calls']==timing['update_calls']==5000
    write(folder/'draw-update-timing.json',timing)
def train(root):
    setup(root);verify(root);out=root/'output';out.mkdir();(out/'runs').mkdir();tick=time.monotonic();records=[];lock=threading.Lock();abort=threading.Event();children=set()
    write(out/'status.json',dict(state='RUNNING',completed=0,total=21))
    def launch(law,arm):
        command=[PYTHON,'-B',str(root/'source/twist_study.py'),'cell','--root',str(root),'--law',law,'--arm',arm]
        with (out/f'{law}-{arm}.log').open('w') as log:
            with lock:
                assert not abort.is_set();p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);children.add(p)
                record=dict(law=law,arm=arm,command=command,pid=p.pid,create_time=psutil.Process(p.pid).create_time());records.append(record);write(out/'launches.json',records)
            try:code=p.wait(timeout=600)
            except subprocess.TimeoutExpired:p.terminate();p.wait();raise
            finally:
                with lock:children.remove(p)
            record['exit_code']=code
            with lock:write(out/'launches.json',records)
            assert code==0,(law,arm,code)
        return law+'/'+arm
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(launch,law,arm) for law in LAWS for arm in ARMS]
            try:
                for count,future in enumerate(as_completed(futures),1):
                    name=future.result();write(out/'status.json',dict(state='RUNNING',completed=count,total=21,latest=name,seconds=time.monotonic()-tick));print(json.dumps(load(out/'status.json')),flush=True)
            except BaseException:
                abort.set()
                for f in futures:f.cancel()
                with lock:
                    for p in children:p.terminate()
                raise
        verify(root);write(out/'status.json',dict(state='COMPLETE',completed=21,total=21,seconds=time.monotonic()-tick))
    except BaseException:write(out/'status.json',dict(state='FAILED',error=traceback.format_exc()));raise
def run(root):
    setup(root);verify(root);receipt=load(root/'documented-pass.json')
    assert receipt['status']=='PASS' and receipt['doc_vs_reality_mismatches']==[] and receipt['preflight_sha256']==sha(root/'preflight.json')
    assert all(v['exit_code']==0 for v in receipt['records']);out=root/'orchestrator';out.mkdir();me=psutil.Process();write(out/'identity.json',dict(pid=me.pid,create_time=me.create_time()))
    phases=[]
    try:
        for phase,ceiling in (('train',7200),('audit',1200)):
            write(out/'status.json',dict(state='RUNNING',phase=phase))
            command=[PYTHON,'-B',str(root/'source/twist_study.py'),phase,'--root',str(root)]
            with (out/f'{phase}.log').open('w') as f:p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
            tracker=psutil.Process(p.pid);tracked={};tick=time.monotonic();r=dict(phase=phase,pid=p.pid,create_time=tracker.create_time(),command=command);phases.append(r);write(out/'launches.json',phases)
            while p.poll() is None:
                if time.monotonic()-tick>ceiling:
                    owned=[*tracker.children(recursive=True),tracker]
                    for proc in owned:
                        try:proc.terminate()
                        except psutil.NoSuchProcess:pass
                    _,alive=psutil.wait_procs(owned,timeout=5)
                    for proc in alive:
                        try:proc.kill()
                        except psutil.NoSuchProcess:pass
                    p.wait();r['exit_code']=p.returncode;write(out/'launches.json',phases);raise TimeoutError(phase)
                resources=[]
                try:processes=[tracker,*tracker.children(recursive=True)]
                except psutil.NoSuchProcess:processes=[]
                for proc in processes:
                    try:
                        identity=(proc.pid,proc.create_time())
                        if identity not in tracked:tracked[identity]=proc;proc.cpu_percent();cpu=None
                        else:cpu=tracked[identity].cpu_percent()
                        resources.append(dict(pid=proc.pid,create_time=identity[1],cpu_percent=cpu,RSS=proc.memory_info().rss))
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                append(out/'resources.jsonl',dict(phase=phase,seconds=time.monotonic()-tick,resources=resources));time.sleep(5)
            r.update(exit_code=p.returncode,seconds=time.monotonic()-tick);write(out/'launches.json',phases);assert p.returncode==0,(phase,p.returncode)
    except BaseException:
        write(out/'status.json',dict(state='FAILED',phases=phases,error=traceback.format_exc()));raise
    verify(root);write(out/'status.json',dict(state='COMPLETE',phases=2,quality_interpretation_requires_actual_tool_exit=True));print(json.dumps(load(out/'status.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','preflight','run','train','cell','audit'));p.add_argument('--root',type=Path,required=True);p.add_argument('--law',choices=LAWS);p.add_argument('--arm',choices=ARMS);a=p.parse_args();root=a.root.resolve()
    if a.action=='prepare':prepare(root)
    elif a.action=='cell':cell(root,a.law,a.arm)
    elif a.action=='audit':
        from audit_study import audit;audit(root)
    else:globals()[a.action](root)
