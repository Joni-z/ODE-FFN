"""Registered native-checkpoint/optimizer compatibility, not a quality search."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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


def write(path: Path, value: object) -> None:
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(8*2**20):h.update(chunk)
    return h.hexdigest()


def append(path: Path,value: dict) -> None:
    with path.open('a') as stream:stream.write(json.dumps(value,allow_nan=False)+'\n')


def table(path: Path,rows: list[dict]) -> None:
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def setup(root: Path):
    settings=load(root/'settings.json');sys.path.insert(0,settings['joint_source'])
    import joint_study as native
    native.setup(root);torch.set_num_threads(1)
    sys.path.insert(0,str(root/'source'))
    from gated_shear import GatedShearFFN
    assert Path(sys.modules['gated_shear'].__file__).resolve()==root/'source/gated_shear.py'
    return settings,native,GatedShearFFN


def build(native,entry: dict):
    return native.build(entry['family']) if entry['dim']==16 else native.old_build(2,entry['family'],'direct')


def converted(model,component):
    rng=torch.get_rng_state().clone();result=copy.deepcopy(model)
    for block in result.raw.blocks:block.ffn=component.from_swiglu(block.ffn)
    assert torch.equal(rng,torch.get_rng_state()),'conversion consumed RNG'
    assert list(model.state_dict())==list(result.state_dict())
    assert [n for n,_ in model.named_parameters()]==[n for n,_ in result.named_parameters()]
    for a,b in zip(model.parameters(),result.parameters()):
        assert torch.equal(a,b) and a.data_ptr()!=b.data_ptr()
    return result


def prepare(root: Path,repo: Path,base: Path) -> None:
    root.mkdir(parents=True,exist_ok=False);(root/'source').mkdir()
    for name in ('study.py','verify_math.py','protocol.md','theory.md'):
        shutil.copyfile(Path(__file__).parent/name,root/'source'/name)
    shutil.copyfile(repo/'synthetic/gated_shear.py',root/'source/gated_shear.py')
    joint=base/'joint_dependence_20261026';mixed=root.parent/'oct27-cycle02'
    settings=load(mixed/'settings.json');settings.update(repo=str(repo),cohort=[],
        prior_seals=[str(root.parent/name/'sealed-runtime.json') for name in ('oct27-cycle01','oct27-cycle02')],
        registration_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip())
    native=base/'conservative_readout_20261026'
    references=load(base/'cubic_split_20261026/audit-cohort.json')
    for law in ('product','sign_code','half_mixture'):
        for family in ('swiglu64','swiglu128','swiglu160'):
            if law=='product':
                row=next(r for r in references if r['family']==family and r['dataset']=='rotated_product_bimodal_16d')
                folder=Path(row['folder']);banks=native/'output/banks/rotated_product_bimodal_16d'
            elif law=='sign_code':folder=joint/'output/runs'/family;banks=joint/'banks'
            else:folder=mixed/'output/runs'/family;banks=mixed/'banks'
            settings['cohort'].append(dict(law=law,family=family,dim=16,folder=str(folder),banks=str(banks)))
    for dataset in ('gaussian_aniso_2d','gmm_ring8_2d'):
        settings['cohort'].append(dict(law=dataset,family='swiglu128',dim=2,
            folder=str(native/'output/runs'/dataset/'swiglu128_direct'),banks=str(native/'output/banks'/dataset)))
    write(root/'settings.json',settings)
    paths=list((root/'source').iterdir())+[root/'settings.json',repo/'docs/autoresearch/program.md']
    paths+=list(Path(settings['native_root']).rglob('*.py'))+list(Path(settings['joint_source']).glob('*.py'))
    for row in settings['cohort']:
        paths+=list(Path(row['folder']).rglob('*'))+list(Path(row['banks']).glob('*.npz'))
    paths += [Path(p) for p in settings['prior_seals']]
    paths=sorted({p for p in paths if p.is_file()})
    write(root/'bindings.json',{str(p):sha(p) for p in paths})
    print(dict(prepared=str(root),cohort=len(settings['cohort']),bindings=len(paths)),flush=True)


def verify_inputs(root: Path) -> int:
    bindings=load(root/'bindings.json')
    for path,digest in bindings.items():assert sha(Path(path))==digest,path
    for seal in load(root/'settings.json')['prior_seals']:
        seal=Path(seal)
        for name,digest in load(seal).items():assert sha(seal.parent/name)==digest,name
    return len(bindings)


def preflight(root: Path) -> None:
    settings,native,component=setup(root);verify_inputs(root)
    from verify_math import verify
    math=verify(component)
    entry=next(r for r in settings['cohort'] if r['law']=='half_mixture' and r['family']=='swiglu128')
    cold=torch.load(Path(entry['folder'])/'step_0000000.pt',weights_only=False)
    bank=torch.from_numpy(np.load(root.parent/'oct27-cycle02/data/train.npy'))
    runs=[]
    for resume in (False,True):
        model=build(native,entry);native.restore_model(model,cold);model=converted(model,component)
        opt=native.optimizer(model);opt.load_state_dict(cold['optimizer']);rng=torch.Generator().manual_seed(100000)
        torch.set_rng_state(cold['torch_rng']);digest=native.EMPTY
        for step in range(1,4):
            x,t,a,b=native.draw(bank,rng);digest=native.legacy.fingerprint_batch(digest,a,b,t);native.update(model,opt,x,t,a,b)
            if resume and step==2:
                state=native.checkpoint(model,opt,rng,step,digest);model=converted(build(native,entry),component)
                opt=native.optimizer(model);digest=native.restore(model,opt,rng,state)
        runs.append(native.checkpoint(model,opt,rng,3,digest))
    native.equal(*runs);verify_inputs(root)
    write(root/'preflight.json',dict(status='PASS',mathematics=math,exact_resume=True))
    print(dict(status='PASS',mathematics=math,exact_resume=True),flush=True)


def run(root: Path) -> None:
    settings,native,component=setup(root);assert load(root/'preflight.json')['status']=='PASS'
    bindings=verify_inputs(root);out=root/'output';out.mkdir(exist_ok=False)
    (out/'predictions').mkdir();(out/'paired-training').mkdir();started=time.monotonic();stop=threading.Event()
    counts=dict(checkpoints=0,panels=0,sampling=0,gradient_probes=0,paired_updates=0)
    def status(state='RUNNING',**extra):
        elapsed=time.monotonic()-started;assert elapsed<600,'registered wall ceiling'
        write(out/'status.json',dict(state=state,pid=os.getpid(),elapsed_seconds=elapsed,**counts,**extra))
    def monitor():
        import psutil
        process=psutil.Process()
        while not stop.is_set():
            append(out/'monitor.jsonl',dict(elapsed=time.monotonic()-started,pid=os.getpid(),
                rss_bytes=process.memory_info().rss,cpu_seconds=sum(process.cpu_times()[:2])));stop.wait(5)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();status()
    endpoints=[];trajectory=[];spectra=[];costs=[]
    try:
        for entry in settings['cohort']:
            folder,banks=Path(entry['folder']),Path(entry['banks'])
            records=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()]
            for step in (0,1000,2500,5000):
                parent=build(native,entry);checkpoint=torch.load(folder/f'step_{step:07d}.pt',weights_only=False)
                native.restore_model(parent,checkpoint);parent.eval();model=converted(parent,component)
                native.equal(parent.state_dict(),model.state_dict());native.equal(dict(parent.named_buffers()),dict(model.named_buffers()))
                tag=f"{entry['law']}-{entry['family']}-{step}"
                metrics=native.legacy.field_evaluation(model,banks,save=out/'predictions'/(tag+'.npz'))
                record=next(r for r in records if r['step']==step)
                for key in ('oracle_mse','pair_mse','prediction_rms'):
                    np.testing.assert_allclose(metrics[key],record[key],rtol=2e-6,atol=1e-9)
                with np.load(banks/'field.npz') as data:xx,tt=data['x'],data['t']
                with np.load(out/'predictions'/(tag+'.npz')) as data:saved=data['prediction']
                with torch.no_grad():
                    for k in range(0,len(xx),1024):
                        actual=parent(torch.from_numpy(xx[k:k+1024]),torch.from_numpy(tt[k:k+1024])).numpy()
                        np.testing.assert_array_equal(actual,saved[k:k+1024])
                endpoints.append(dict(law=entry['law'],family=entry['family'],step=step,**metrics,
                    native_prediction_bitwise_equal=True,parameter_state_equal=True))
                # Capture gates on actual post-norm/FiLM inputs, never on invented unnormalized states.
                hooks=[]
                def capture(name,module,args):
                    with torch.no_grad():
                        gate=F_silu(module.w12(args[0]).chunk(2,-1)[0]);absolute=gate.abs().double();maximum=absolute.amax(-1)
                        flow_norm=.5*(torch.sqrt(4+maximum.square())+maximum)
                        spectra.append(dict(law=entry['law'],family=entry['family'],step=step,layer=name,
                            width=gate.shape[-1],gate_sigma_mean=float(absolute.mean()),gate_sigma_max=float(absolute.max()),
                            conditional_flow_norm_mean=float(flow_norm.mean()),conditional_flow_norm_max=float(flow_norm.max()),
                            generator_eigenvalue_max_abs=0.,input_rows=len(gate)))
                for i,block in enumerate(model.raw.blocks):
                    hooks.append(block.ffn.register_forward_pre_hook(lambda module,args,name=f'block{i}':capture(name,module,args)))
                with np.load(banks/'probe.npz') as probe:
                    with torch.no_grad():model(torch.from_numpy(probe['x']),torch.from_numpy(probe['t']))
                for hook in hooks:hook.remove()
                counts['checkpoints']+=1
                if step==5000:
                    native_cost=native.compute_budget(parent);shear_cost=native.compute_budget(model);native.equal(native_cost,shear_cost)
                    costs.append(dict(law=entry['law'],family=entry['family'],**shear_cost,identical_to_native=True))
                    # A fitted nonzero head exposes the full backbone parameter response.
                    assert float(parent.raw.output.weight.norm())>1e-5
                    gradients=[]
                    for m in (parent,model):
                        x=torch.from_numpy(xx[:8]).clone().requires_grad_(True);t=torch.from_numpy(tt[:8])
                        value=m(x,t);coeff=torch.cos(torch.arange(value.numel(),dtype=value.dtype)).reshape_as(value)
                        loss=(value*coeff).sum()+.01*value.square().sum()
                        gradients.append(torch.autograd.grad(loss,(x,*m.parameters())))
                    native.equal(*gradients);counts['gradient_probes']+=1
                    for clock in (.1,.3,.5,.7,.9):
                        with np.load(banks/f'panel_{clock:.1f}.npz') as panel:x,t=torch.from_numpy(panel['x']),torch.from_numpy(panel['t'])
                        a,ja=native.legacy.jacobian(parent,x,t);b,jb=native.legacy.jacobian(model,x,t)
                        assert torch.equal(a,b) and torch.equal(ja,jb)
                        counts['panels']+=1
                    with np.load(banks/'sampling.npz') as bank:source=torch.from_numpy(bank['source'])
                    a=native.legacy.generated_samples(parent,source,32);b=native.legacy.generated_samples(model,source,32)
                    assert torch.equal(a,b)
                    with np.load(folder/'diagnostics/samples.npz') as saved_sample:np.testing.assert_array_equal(a.numpy(),saved_sample['generated'])
                    counts['sampling']+=1
                status(current=tag)
            print(dict(law=entry['law'],family=entry['family'],**counts,elapsed=time.monotonic()-started),flush=True)
        # One paired compatibility trajectory, using the unchanged historical minibatches.
        entry=next(r for r in settings['cohort'] if r['law']=='half_mixture' and r['family']=='swiglu128')
        folder=Path(entry['folder']);cold=torch.load(folder/'step_0000000.pt',weights_only=False)
        parent=build(native,entry);native.restore_model(parent,cold);model=converted(parent,component)
        optimizers=[native.optimizer(m) for m in (parent,model)]
        for opt in optimizers:opt.load_state_dict(copy.deepcopy(cold['optimizer']))
        generators=[torch.Generator().manual_seed(100000) for _ in range(2)]
        torch.set_rng_state(cold['torch_rng']);digests=[native.EMPTY,native.EMPTY]
        bank=torch.from_numpy(np.load(root.parent/'oct27-cycle02/data/train.npy'))
        for step in range(5001):
            if step:
                outcomes=[]
                for index,m in enumerate((parent,model)):
                    x,t,a,b=native.draw(bank,generators[index]);digests[index]=native.legacy.fingerprint_batch(digests[index],a,b,t)
                    outcomes.append(native.update(m,optimizers[index],x,t,a,b))
                assert outcomes[0]==outcomes[1]
                counts['paired_updates']=step
                if step%500==0:trajectory.append(dict(step=step,loss=outcomes[0][0],gradient_norm=outcomes[0][1],paired_equal=True))
                if step%100==0:status(current='paired-training')
            if step in (0,1000,2500,5000):
                states=[native.checkpoint(m,opt,rng,step,digest) for m,opt,rng,digest in zip((parent,model),optimizers,generators,digests)]
                native.equal(*states)
                historical=torch.load(folder/f'step_{step:07d}.pt',weights_only=False);native.equal(states[0],historical)
                torch.save(states[1],out/'paired-training'/f'step_{step:07d}.pt')
        assert counts==dict(checkpoints=44,panels=55,sampling=11,gradient_probes=11,paired_updates=5000),counts
        verify_inputs(root)
        table(out/'compatibility.csv',endpoints);table(out/'generator-stats.csv',spectra)
        table(out/'costs.csv',costs);table(out/'paired-training.csv',trajectory)
        write(out/'decision.json',dict(status='keep_as_exact_baseline',quality_gain=False,new_hypothesis_class=False,
            next_feedback_or_quality_training_selected=False,**counts,bindings_unchanged=bindings))
        status('COMPLETE')
    except BaseException:
        write(out/'failure.json',dict(error=traceback.format_exc(),counts=counts))
        write(out/'status.json',dict(state='FAILED',pid=os.getpid(),elapsed_seconds=time.monotonic()-started,**counts))
        raise
    finally:stop.set();watcher.join(timeout=6)


def F_silu(value: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(value)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','preflight','run'));parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--repo',type=Path);parser.add_argument('--base',type=Path);args=parser.parse_args()
    if args.action=='prepare':prepare(args.root,args.repo,args.base)
    elif args.action=='preflight':preflight(args.root)
    else:run(args.root)
