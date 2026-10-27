"""Fixed fifteen-cell gate-transport screen with immutable native references."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np
import torch

ARMS=('transport','diagonal','swiglu143')
STEPS=(0,1000,2500,5000)
TIMES=(.1,.3,.5,.7,.9)


def load(path: Path) -> dict:return json.loads(path.read_text())


def write(path: Path,value: object) -> None:
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)


def append(path: Path,value: object) -> None:
    with path.open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')


def setup(root: Path):
    settings=load(root/'settings.json');sys.path.insert(0,settings['joint_source'])
    import joint_study as native
    native.setup(root);torch.set_num_threads(1)
    sys.path.insert(0,str(root/'source'))
    from layers.gate_transport import GateTransportFFN,DiagonalGateScaleFFN
    assert Path(sys.modules['layers.gate_transport'].__file__).resolve()==root/'source/layers/gate_transport.py'
    return settings,native,GateTransportFFN,DiagonalGateScaleFFN


def entry_for(root: Path,law: str) -> dict:
    return next(row for row in load(root/'settings.json')['laws'] if row['law']==law)


def build(root: Path,law: str,arm: str):
    settings,native,transport,diagonal=setup(root);entry=entry_for(root,law)
    model=native.old_build(entry['dim'],'swiglu128','direct')
    cold=torch.load(Path(entry['references']['swiglu128'])/'step_0000000.pt',weights_only=False)
    native.restore_model(model,cold)
    if arm in ('transport','diagonal'):
        kind=transport if arm=='transport' else diagonal
        for block in model.raw.blocks:block.ffn=kind.from_swiglu(block.ffn)
    elif arm in ('swiglu143','swiglu160'):
        torch.manual_seed(0)
        raw=native.VelocityNet(entry['dim'],'swiglu',width=64,depth=3,ffn_hidden=int(arm[6:]))
        native.copy_surrounding(raw,model.raw);model.raw=raw
    elif arm!='swiglu128':raise ValueError(arm)
    torch.set_rng_state(cold['torch_rng'])
    return model


def training_bank(entry: dict,native) -> torch.Tensor:
    path=Path(entry['train'])
    return torch.from_numpy(np.load(path) if path.suffix=='.npy' else native.legacy.load_targets(path,entry['dim']))


def prepare(root: Path,repo: Path,base: Path) -> None:
    root.mkdir(parents=True,exist_ok=False);source=root/'source';source.mkdir();layers=source/'layers';layers.mkdir()
    (layers/'__init__.py').write_text('')
    for name in ('gated_shear.py','gate_transport.py'):shutil.copyfile(repo/'synthetic'/name,layers/name)
    for name in ('study.py','verify_operator.py','protocol.md','design.md'):shutil.copyfile(Path(__file__).parent/name,source/name)
    previous=root.parent;settings=load(previous/'oct27-cycle02/settings.json')
    settings.update(repo=str(repo),laws=[],prior_seals=[str(previous/name/'sealed-runtime.json') for name in ('oct27-cycle01','oct27-cycle02','oct27-cycle03')],
        registration_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip())
    old=base/'conservative_readout_20261026';joint=base/'joint_dependence_20261026';mixed=previous/'oct27-cycle02'
    config=load(Path(settings['native_training_config']));cubic=load(base/'cubic_split_20261026/audit-cohort.json')
    for law in ('gaussian_aniso_2d','gmm_ring8_2d','rotated_product_bimodal_16d','sign_code','half_mixture'):
        dim=2 if law in ('gaussian_aniso_2d','gmm_ring8_2d') else 16
        if law in ('sign_code','half_mixture'):
            parent=joint if law=='sign_code' else mixed
            banks=parent/'banks';train=parent/'data/train.npy'
            references={family:str(parent/'output/runs'/family) for family in ('swiglu128','swiglu160')}
        else:
            banks=old/'output/banks'/law;train=Path(config['data_roots'][law])/law/'train.npz'
            references=dict(swiglu128=str(old/'output/runs'/law/'swiglu128_direct'),
                swiglu160=next(r['folder'] for r in cubic if r['dataset']==law and r['family']=='swiglu160'))
        settings['laws'].append(dict(law=law,dim=dim,banks=str(banks),train=str(train),references=references))
    write(root/'settings.json',settings);_,native,_,_=setup(root)
    paths=list(source.rglob('*'))+[root/'settings.json',repo/'docs/autoresearch/program.md']
    paths+=list(Path(settings['native_root']).rglob('*.py'))+list(Path(settings['joint_source']).glob('*.py'))
    for entry in settings['laws']:
        paths+=[Path(entry['train'])]+list(Path(entry['banks']).glob('*.npz'))
        for folder in entry['references'].values():paths+=list(Path(folder).rglob('*'))
    paths+=[Path(p) for p in settings['prior_seals']]
    paths=sorted({p for p in paths if p.is_file() and p.suffix!='.pyc'})
    write(root/'bindings.json',{str(path):native.sha(path) for path in paths})
    print(dict(prepared=str(root),bindings=len(paths),cells=15),flush=True)


def verify_inputs(root: Path,native) -> None:
    for path,digest in load(root/'bindings.json').items():assert native.sha(path)==digest,path
    for seal in load(root/'settings.json')['prior_seals']:
        seal=Path(seal)
        for name,digest in load(seal).items():assert native.sha(seal.parent/name)==digest,name


def preflight(root: Path) -> None:
    settings,native,transport,diagonal=setup(root);verify_inputs(root,native)
    from verify_operator import verify
    mathematics=verify(transport,diagonal)
    rows=[]
    for entry in settings['laws']:
        bank=training_bank(entry,native);base=build(root,entry['law'],'swiglu128')
        for arm in ARMS:
            model=build(root,entry['law'],arm)
            if arm!='swiglu143':
                for name,value in base.state_dict().items():assert torch.equal(value,model.state_dict()[name]),name
                x=bank[:8];t=torch.full((len(x),1),.3)
                assert torch.equal(base(x,t),model(x,t))
                # The nonzero fitted probe head exposes hidden-parameter responses.
                with torch.no_grad():
                    head=torch.sin(torch.arange(base.raw.output.weight.numel())).reshape_as(base.raw.output.weight)*.01
                    base.raw.output.weight.copy_(head);model.raw.output.weight.copy_(head)
                aa=torch.autograd.grad(base(x,t).square().sum(),tuple(base.parameters()))
                names=dict(model.named_parameters());bb=torch.autograd.grad(model(x,t).square().sum(),[names[n] for n,_ in base.named_parameters()])
                for a,b in zip(aa,bb):torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-6)
                base=build(root,entry['law'],'swiglu128')
            ends=[]
            for resume in (False,True):
                model=build(root,entry['law'],arm);opt=native.optimizer(model);rng=torch.Generator().manual_seed(100000);digest=native.EMPTY
                for step in range(1,4):
                    x,t,a,b=native.draw(bank,rng);digest=native.legacy.fingerprint_batch(digest,a,b,t);native.update(model,opt,x,t,a,b)
                    if resume and step==2:
                        state=native.checkpoint(model,opt,rng,step,digest);model=build(root,entry['law'],arm);opt=native.optimizer(model)
                        digest=native.restore(model,opt,rng,state)
                ends.append(native.checkpoint(model,opt,rng,3,digest))
            native.equal(*ends);cost=native.compute_budget(model)
            if entry['dim']==16 and arm in ('transport','swiglu143'):
                expected=(89016,87210) if arm=='transport' else (89418,87552)
                assert (cost['parameters'],cost['dense_forward_macs'])==expected,(arm,cost)
            rows.append(dict(law=entry['law'],arm=arm,cold_pairing=arm!='swiglu143',exact_resume=True,**cost))
    verify_inputs(root,native)
    write(root/'preflight.json',dict(status='PASS',mathematics=mathematics,cells=rows))
    write(root/'ready.json',dict(bindings_sha256=native.sha(root/'bindings.json'),preflight_sha256=native.sha(root/'preflight.json')))
    print(dict(status='PASS',cells=len(rows),mathematics=mathematics),flush=True)


def generator_visits(model,banks: Path) -> list[dict]:
    visits=[];hooks=[]
    for index,block in enumerate(model.raw.blocks):
        if not hasattr(block.ffn,'transport'):continue
        def capture(module,args,index=index):
            with torch.no_grad():
                value=module.w12(args[0][:2]).chunk(2,-1)[1].double();c=module.transport.detach().double()
                r=module.w3.in_features;m=module.left_width
                p=c.new_zeros(r,r);p[:m,m:]=c
                singular=torch.linalg.svdvals(c);conditional=[]
                for v in value:
                    k=c.new_zeros(2*r,2*r);k[:r,:r]=p;k[r:,:r]=torch.diag(v)
                    assert torch.equal(k@k@k,torch.zeros_like(k))
                    conditional.append(torch.linalg.svdvals(k).tolist())
                visits.append(dict(layer=f'block{index}',rectangular_C_singular_values=singular.tolist(),
                    gate_operator_eigenvalues_zero=True,joint_generator_cube_zero=True,
                    conditional_K_singular_values=conditional,gate_operator_frobenius=float(c.norm())))
        hooks.append(block.ffn.register_forward_pre_hook(capture))
    try:
        with np.load(banks/'probe.npz') as data:
            with torch.no_grad():model(torch.from_numpy(data['x'][:32]),torch.from_numpy(data['t'][:32]))
    finally:
        for hook in hooks:hook.remove()
    return visits


def cell(root: Path,law: str,arm: str) -> None:
    _,native,_,_=setup(root);entry=entry_for(root,law);folder=root/'output/runs'/law/arm;folder.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    def status(state='RUNNING',step=0,**extra):write(folder/'status.json',dict(state=state,step=step,pid=os.getpid(),elapsed_seconds=time.monotonic()-started,**extra))
    status()
    try:
        model=build(root,law,arm);opt=native.optimizer(model);rng=torch.Generator().manual_seed(100000);digest=native.EMPTY
        bank=training_bank(entry,native);banks=Path(entry['banks']);cost=native.compute_budget(model)
        write(folder/'provenance.json',dict(law=law,arm=arm,model_seed=0,training_stream=100000,steps=5000,batch=512,**cost))
        def save(step):
            torch.save(native.checkpoint(model,opt,rng,step,digest),folder/f'step_{step:07d}.pt')
            write(folder/f'generator_{step:07d}.json',generator_visits(model,banks))
            return native.legacy.field_evaluation(model,banks,save=folder/f'field_{step:07d}.npz')
        append(folder/'metrics.jsonl',dict(step=0,**save(0)))
        ls=gs=0.;n=clipped=0
        for step in range(1,5001):
            x,t,a,b=native.draw(bank,rng);digest=native.legacy.fingerprint_batch(digest,a,b,t)
            loss,gradient=native.update(model,opt,x,t,a,b);ls+=loss;gs+=gradient;n+=1;clipped+=int(gradient>1)
            if step%100==0:status(step=step);assert time.monotonic()-started<600,'registered cell timeout'
            if step%500==0:
                row=dict(step=step,train_loss=ls/n,grad_norm=gs/n,clip_fraction=clipped/n)
                if step in STEPS:row.update(save(step))
                append(folder/'metrics.jsonl',row);ls=gs=0.;n=clipped=0
        result=native.legacy.diagnostics(model,banks,folder/'diagnostics',rollout=False)
        write(folder/'result.json',dict(law=law,arm=arm,input_stream_digest=digest,**cost,**result));status('COMPLETE',5000)
    except BaseException:status('FAILED',error=traceback.format_exc());raise


def run(root: Path) -> None:
    settings,native,_,_=setup(root);ready=load(root/'ready.json')
    assert native.sha(root/'bindings.json')==ready['bindings_sha256'] and native.sha(root/'preflight.json')==ready['preflight_sha256']
    assert load(root/'preflight.json')['status']=='PASS';verify_inputs(root,native)
    out=root/'output';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    started=time.monotonic();stop=threading.Event();abort=threading.Event();lock=threading.Lock();children=set()
    status=dict(state='RUNNING',completed=0,total=15,pid=os.getpid())
    def progress(**extra):status.update(extra,elapsed_seconds=time.monotonic()-started);write(out/'status.json',status);print(status,flush=True)
    def monitor():
        import psutil
        while not stop.is_set():
            parent=psutil.Process();rows=[]
            for process in [parent]+parent.children(recursive=True):
                try:rows.append(dict(pid=process.pid,rss_bytes=process.memory_info().rss,cpu_seconds=sum(process.cpu_times()[:2])))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            append(out/'monitor.jsonl',dict(elapsed=time.monotonic()-started,processes=rows));stop.wait(5)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();progress()
    environment=os.environ.copy();environment.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    def launch(law,arm):
        command=[sys.executable,str(Path(__file__)),'cell','--root',str(root),'--law',law,'--arm',arm]
        with (out/f'{law}-{arm}.log').open('w') as log:
            with lock:
                if abort.is_set():raise RuntimeError('study aborted')
                process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT);children.add(process)
                append(out/'launches.jsonl',dict(law=law,arm=arm,pid=process.pid,command=command))
            try:code=process.wait(timeout=600)
            except subprocess.TimeoutExpired:process.terminate();process.wait();raise
            finally:
                with lock:children.remove(process)
        assert code==0,(law,arm,code)
        return law+'/'+arm
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(launch,entry['law'],arm) for entry in settings['laws'] for arm in ARMS]
            try:
                for future in as_completed(futures):progress(completed=status['completed']+1,latest=future.result())
            except BaseException:
                abort.set()
                for future in futures:future.cancel()
                with lock:
                    for process in children:process.terminate()
                raise
        verify_inputs(root,native);progress(state='COMPLETE')
    except BaseException:progress(state='FAILED',error=traceback.format_exc());raise
    finally:stop.set();watcher.join(timeout=6)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','preflight','run','cell'));p.add_argument('--root',type=Path,required=True)
    p.add_argument('--repo',type=Path);p.add_argument('--base',type=Path);p.add_argument('--law');p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    if a.action=='prepare':prepare(a.root,a.repo,a.base)
    elif a.action=='preflight':preflight(a.root)
    elif a.action=='run':run(a.root)
    else:cell(a.root,a.law,a.arm)
