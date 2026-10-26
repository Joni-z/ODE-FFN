"""Five unchanged architectures on a controlled dependent-sign target law."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time as timer
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from joint_core import SignCodeMixture, target_geometry

FAMILIES = ('ode_k2', 'free_isru', 'swiglu64', 'swiglu128', 'swiglu160')
SNAPSHOTS = (0, 1000, 2500, 5000)
TIMES = (.1, .3, .5, .7, .9)
EMPTY = hashlib.sha256(b'').hexdigest()
load = lambda p: json.loads(Path(p).read_text())


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def append(path: Path, value) -> None:
    with path.open('a') as stream: stream.write(json.dumps(value, allow_nan=False) + '\n')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(8 * 2**20): h.update(data)
    return h.hexdigest()


def setup(root: Path) -> dict:
    settings = load(root / 'settings.json')
    native = Path(settings['native_root'])
    sys.path.insert(0, str(native))
    global legacy, old_build, compute_budget, copy_surrounding, VelocityNet, restore_model
    from scripts import run_recurrent_depth as legacy
    from synthetic.conservative_readout import build as old_build, compute_budget
    from synthetic.recurrent_depth import copy_surrounding
    from synthetic.models import VelocityNet
    from synthetic.train import restore_model
    assert Path(sys.modules['synthetic.models'].__file__).resolve() == (native / 'synthetic/models.py').resolve()
    return settings


class FreeISRU(nn.Module):
    def __init__(self, original: nn.Module):
        super().__init__()
        with torch.no_grad():
            p = original.proj.weight
            initial = torch.eye(len(p), dtype=p.dtype, device=p.device) + .3 * p + .045 * (p @ p)
        self.matrix1 = nn.Parameter(initial.clone())
        self.matrix2 = nn.Parameter(initial.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = F.linear(x, self.matrix1)
        return F.linear(u * torch.rsqrt(1 + 1.2 * u.square()), self.matrix2)


def build(family: str) -> nn.Module:
    if family in ('ode_k2', 'swiglu128'):
        return old_build(16, family, 'direct')
    model = old_build(16, 'ode_k2', 'direct')
    if family == 'free_isru':
        for block in model.raw.blocks: block.ffn = FreeISRU(block.ffn.ode1)
    else:
        torch.manual_seed(0)
        raw = VelocityNet(16, 'swiglu', width=64, depth=3, ffn_hidden=int(family[6:]))
        copy_surrounding(raw, model.raw); model.raw = raw
    return model


def optimizer(model: nn.Module):
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.)


def equal(a, b) -> None:
    if isinstance(a, torch.Tensor): assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else: assert a == b, (a, b)


def reference_rows(root: Path) -> list:
    rows = load(Path(load(root/'settings.json')['reference_root'])/'audit-cohort.json')
    return [r for r in rows if r['dataset'] == 'rotated_product_bimodal_16d' and r['family'] in FAMILIES]


def distribution(root: Path) -> SignCodeMixture:
    info = load(root/'data/distribution.json')
    return SignCodeMixture(np.array(info['locations']), np.array(info['rotation']))


def prepare_data(root: Path) -> None:
    settings = setup(root)
    old = Path(settings['native_root']).parent
    # Obtain the exact original Product16 amplitudes/rotation from its data manifest.
    native_config = load(Path(settings['native_training_config']))
    product_root = Path(native_config['data_roots']['rotated_product_bimodal_16d'])
    specification = load(product_root/'manifest.json')['datasets']['rotated_product_bimodal_16d']
    product = legacy.load_distribution(specification)
    target = SignCodeMixture(product.locations, product.rotation)
    folder = root/'data'; folder.mkdir(exist_ok=False)
    banks = root/'banks'; banks.mkdir(exist_ok=False)
    write(folder/'distribution.json', target.to_dict())
    rng = np.random.default_rng(0)
    train = target.sample(rng, 100000); validation = target.sample(rng, 20000)
    np.save(folder/'train.npy', train); np.save(folder/'validation.npy', validation)
    probe_target = target.sample(rng, 2048)
    from synthetic.data import fm_batch
    probe_x, probe_t, _ = fm_batch(probe_target, rng)
    np.savez_compressed(banks/'probe.npz', x=probe_x[:256], t=probe_t[:256])
    x, t, pair = fm_batch(validation, np.random.default_rng(7761))
    np.savez_compressed(banks/'field.npz', x=x, t=t, pair=pair, oracle=target.oracle(x,t).astype(np.float32))
    sampler = np.random.default_rng(14427)
    source = sampler.standard_normal((2048,16)).astype(np.float32)
    directions = sampler.standard_normal((128,16)).astype(np.float32)
    directions /= np.linalg.norm(directions,axis=1,keepdims=True).clip(min=1e-12)
    np.savez_compressed(banks/'sampling.npz', source=source,target=validation[:2048],
                        target2=validation[2048:4096],directions=directions)
    panel_rng = np.random.default_rng(4427)
    for clock in TIMES:
        tt = np.full((512,1),clock,dtype=np.float32)
        xx = tt*validation[:512]+(1-tt)*panel_rng.standard_normal((512,16)).astype(np.float32)
        velocity,jacobian,_ = target.oracle(xx,tt,True)
        np.savez_compressed(banks/f'panel_{clock:.1f}.npz',x=xx,t=tt,
                            oracle=velocity,oracle_jacobian=jacobian)
    paths = [p for directory in [folder,banks] for p in directory.iterdir() if p.is_file()]
    write(root/'data-manifest.json',dict(data_seed=0,training_count=100000,validation_count=20000,
        probe_count=2048,source='standard_normal',coupling='independent',components=32,
        files={str(p.relative_to(root)):sha(p) for p in paths},original_specification=specification,
        original_manifest_sha256=sha(product_root/'manifest.json')))


def draw(bank: torch.Tensor, rng: torch.Generator):
    target = bank[torch.randint(len(bank),(512,),generator=rng)]
    source = torch.randn(target.shape,generator=rng); t=torch.rand((512,1),generator=rng)
    return (1-t)*source+t*target,t,target,source


def update(model,opt,x,t,target,source):
    opt.zero_grad(set_to_none=True); loss=(model(x,t)-(target-source)).square().mean()
    assert torch.isfinite(loss);loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    assert torch.isfinite(gradient);opt.step();return float(loss.detach()),float(gradient)


def checkpoint(model,opt,rng,step,digest):
    return copy.deepcopy(dict(model=model.state_dict(),optimizer=opt.state_dict(),data_rng=rng.get_state(),
        torch_rng=torch.get_rng_state(),step=step,input_stream_digest=digest,
        runtime_buffers={k:v.detach().clone() for k,v in model.named_buffers()}))


def restore(model,opt,rng,state):
    restore_model(model,state);opt.load_state_dict(state['optimizer']);rng.set_state(state['data_rng'])
    torch.set_rng_state(state['torch_rng']);return state['input_stream_digest']


def preflight(root: Path) -> None:
    settings=setup(root);torch.set_num_threads(1);target=distribution(root)
    s=target.signs;mu=target.locations;r=target.rotation
    np.testing.assert_array_equal(s.mean(0),np.zeros(16))
    np.testing.assert_allclose(s.T@s/len(s),np.eye(16),atol=1e-14)
    for count in (1,2,3):
        for index in itertools.combinations(range(16),count):
            _,counts=np.unique(s[:,index],axis=0,return_counts=True)
            np.testing.assert_array_equal(counts,np.full(2**count,len(s)//2**count))
    np.testing.assert_allclose(np.sum(target.means**2,axis=1),np.full(32,np.sum(mu**2)),rtol=1e-14)
    assert float(np.mean(np.prod(s[:,:4],axis=1)))==1.
    # Compare against the pre-existing enumerated Gaussian-mixture conditional formula.
    from synthetic.data import GaussianMixture
    generic=GaussianMixture(target.means,np.full((32,16),.04),np.ones(32))
    x=np.sin(np.arange(11*16).reshape(11,16))*.7;times=np.linspace(.05,.95,11)[:,None]
    velocity,jacobian,_=target.oracle(x,times,True)
    np.testing.assert_allclose(velocity,generic.velocity(x@r,times)@r.T,rtol=1e-11,atol=1e-11)
    xx=torch.tensor(x,dtype=torch.float64,requires_grad=True);tt=torch.tensor(times)
    means=torch.tensor(target.means);rotation=torch.tensor(r);q=xx@rotation
    variance=(1-tt)**2+.04*tt**2
    residual=q[:,None]-tt[:,:,None]*means
    logits=-.5*(residual.square()/variance[:,:,None]).sum(-1)
    probability=torch.softmax(logits,dim=1)
    coefficient=(.04*tt-(1-tt))/variance
    component_velocity=means+coefficient[:,:,None]*residual
    native=(probability[:,:,None]*component_velocity).sum(1)@rotation.T
    derivative=torch.stack([torch.autograd.grad(native[:,j].sum(),xx,retain_graph=True)[0] for j in range(16)],1)
    np.testing.assert_allclose(native.detach().numpy(),velocity,rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(derivative.detach().numpy(),jacobian,rtol=1e-9,atol=1e-10)
    # Fixed analytic witness: the zero-state Jacobian has distinct latent eigenvalues.
    witness=np.stack([np.zeros(16),.3*np.sin(np.arange(16)+1)])@r.T
    _,jj,_=target.oracle(witness,.3,True)
    witness_norm=float(np.linalg.norm(jj[0]@jj[1]-jj[1]@jj[0]))
    assert witness_norm>1e-6,witness_norm
    bank=torch.from_numpy(np.load(root/'data/train.npy'));references=reference_rows(root);assert len(references)==5
    history=[]
    for row in references:
        family=row['family'];model=build(family);state=torch.load(Path(row['folder'])/'step_0000000.pt',weights_only=False)
        equal(model.state_dict(),state['model']);equal(dict(model.named_buffers()),state['runtime_buffers'])
        equal(optimizer(model).state_dict(),state['optimizer'])
        runs=[]
        for resume in (False,True):
            model=build(family);opt=optimizer(model);rng=torch.Generator().manual_seed(100000);digest=EMPTY
            for step in range(1,4):
                x,t,a,b=draw(bank,rng);digest=legacy.fingerprint_batch(digest,a,b,t);update(model,opt,x,t,a,b)
                if resume and step==2:
                    state=checkpoint(model,opt,rng,step,digest);model=build(family);opt=optimizer(model)
                    digest=restore(model,opt,rng,state)
            runs.append(checkpoint(model,opt,rng,3,digest))
        equal(*runs);history.append(dict(family=family,exact_native_reference=True,exact_resume=True,costs=compute_budget(build(family))))
    write(root/'preflight.json',dict(status='PASS',native_oracle_max=float(np.max(np.abs(native.detach().numpy()-velocity))),
        native_jacobian_max=float(np.max(np.abs(derivative.detach().numpy()-jacobian))),
        matched_sign_marginals_through_order=3,all_component_norms_equal=True,four_sign_product=1.,
        noncommuting_witness_norm=witness_norm,references_and_resume=history))


def spectrum(model):
    from synthetic.models import OriginalODELayer
    rows=[]
    for name,module in model.named_modules():
        if isinstance(module,OriginalODELayer):
            p=module.proj.weight.detach().double()
            rows.append(dict(layer=name,singular_values=torch.linalg.svdvals(p).tolist(),
                             symmetric_eigenvalues=torch.linalg.eigvalsh(.5*(p+p.T)).tolist()))
    return rows


def cell(root: Path,family: str) -> None:
    setup(root);torch.set_num_threads(1)
    folder=root/'output/runs'/family;folder.mkdir(exist_ok=False);started=timer.monotonic()
    def status(state='RUNNING',step=0,**extra):write(folder/'status.json',dict(state=state,step=step,pid=os.getpid(),elapsed_seconds=timer.monotonic()-started,**extra))
    status()
    try:
        model=build(family);opt=optimizer(model);rng=torch.Generator().manual_seed(100000);digest=EMPTY
        bank=torch.from_numpy(np.load(root/'data/train.npy'));banks=root/'banks';costs=compute_budget(model)
        write(folder/'provenance.json',dict(family=family,model_seed=0,stream_initialization=100000,steps=5000,batch=512,costs=costs))
        def save(step):
            torch.save(checkpoint(model,opt,rng,step,digest),folder/f'step_{step:07d}.pt')
            write(folder/f'spectrum_{step:07d}.json',spectrum(model))
            write(folder/f'visits_{step:07d}.json',legacy.visit_diagnostics(model,banks))
            return legacy.field_evaluation(model,banks,save=folder/f'field_{step:07d}.npz')
        append(folder/'metrics.jsonl',dict(step=0,**save(0)))
        loss_sum=grad_sum=0.;count=clipped=0
        for step in range(1,5001):
            x,t,a,b=draw(bank,rng);digest=legacy.fingerprint_batch(digest,a,b,t);loss,grad=update(model,opt,x,t,a,b)
            loss_sum+=loss;grad_sum+=grad;count+=1;clipped+=int(grad>1)
            if step%100==0:status(step=step)
            if step%500==0:
                row=dict(step=step,train_loss=loss_sum/count,grad_norm=grad_sum/count,clip_fraction=clipped/count)
                if step in SNAPSHOTS:row.update(save(step))
                append(folder/'metrics.jsonl',row);print(row,flush=True);loss_sum=grad_sum=0.;count=clipped=0
        result=legacy.diagnostics(model,banks,folder/'diagnostics',rollout=False)
        write(folder/'result.json',dict(family=family,law='sign_code',input_stream_digest=digest,**costs,**result))
        status('COMPLETE',5000)
    except BaseException:status('FAILED',error=traceback.format_exc());raise


def run(root: Path) -> None:
    setup(root);torch.set_num_threads(1);ready=load(root/'ready.json');assert load(root/'preflight.json')['status']=='PASS'
    for path,digest in ready['bindings'].items():assert sha(path)==digest,path
    out=root/'output';out.mkdir(exist_ok=False);(out/'runs').mkdir();(out/'reused').mkdir()
    for row in reference_rows(root):
        folder=Path(row['folder']);write(out/'reused'/f"{row['family']}.json",dict(**row,
            files_sha256={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.is_file()}))
    stop=threading.Event();abort=threading.Event();lock=threading.Lock();children=set();started=timer.monotonic()
    progress=dict(state='RUNNING',completed=0,total_new=5,historical=5,pid=os.getpid())
    def status(**extra):progress.update(extra,elapsed_seconds=timer.monotonic()-started);write(out/'status.json',progress);print(progress,flush=True)
    def monitor():
        import psutil
        while not stop.is_set():
            row=dict(time=legacy.now(),processes=[]);parent=psutil.Process()
            for p in [parent]+parent.children(recursive=True):
                try:row['processes'].append(dict(pid=p.pid,rss_bytes=p.memory_info().rss,cpu_seconds=sum(p.cpu_times()[:2])))
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            append(out/'monitor.jsonl',row);write(out/'monitor.json',row);stop.wait(5)
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();status()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    def launch(family):
        command=[sys.executable,str(Path(__file__)),'cell','--root',str(root),'--family',family]
        with (out/(family+'.log')).open('w') as log:
            with lock:
                if abort.is_set():raise RuntimeError('study aborted')
                p=subprocess.Popen(command,cwd=Path(__file__).parent,env=env,stdout=log,stderr=subprocess.STDOUT);children.add(p)
                append(out/'launches.jsonl',dict(family=family,pid=p.pid,command=command))
            code=p.wait()
            with lock:children.remove(p)
        if code:raise RuntimeError(f'{family} exited {code}')
        return family
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures=[executor.submit(launch,family) for family in FAMILIES]
            try:
                for future in as_completed(futures):status(completed=progress['completed']+1,latest=future.result())
            except BaseException:
                abort.set()
                for future in futures:future.cancel()
                with lock:
                    for p in children:p.terminate()
                raise
        for path,digest in ready['bindings'].items():assert sha(path)==digest,path
        status(state='COMPLETE')
    except BaseException:status(state='FAILED',error=traceback.format_exc());raise
    finally:stop.set();watcher.join(timeout=6)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['data','preflight','cell','run']);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--family',choices=FAMILIES);args=parser.parse_args()
    if args.action=='data':prepare_data(args.root)
    elif args.action=='preflight':preflight(args.root)
    elif args.action=='run':run(args.root)
    else:cell(args.root,args.family)
