"""One independent second-coefficient screen with unchanged native geometry training/eval."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
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

ARMS = ('jet_c_clean','affine_c_clean','free_c_clean','direct_b_clean','swiglu258_clean')


def load(path):return json.loads(Path(path).read_text())
def write(path, value):
    path = Path(path);temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n');os.replace(temporary, path)
def append(path, value):
    with Path(path).open('a') as stream:stream.write(json.dumps(value, allow_nan=False)+'\n')


def import_file(name, path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec);sys.modules[name] = module;spec.loader.exec_module(module)
    module = sys.modules[name]
    assert Path(module.__file__).resolve()==Path(path).resolve()
    return module


def setup(root):
    settings = load(root/'settings.json')
    geometry = import_file('geometry_study', root/'source/geometry_study.py')
    _, native, core = geometry.setup(root)
    sys.path.insert(0, str(root/'source'))
    operator = import_file('coefficient_operator', root/'source/coefficient_operator.py')
    geometry.build = build;geometry.generator_visits = generator_visits
    return settings, geometry, native, core, operator


def verify_inputs(root):
    settings, _, native, _, _ = setup(root)
    for name, digest in load(root/'bindings.json').items():assert native.sha(Path(name))==digest, name
    # Verify sealed lineage and its external bindings without importing prior
    # benchmark modules into the new benchmark's Python module namespace.
    for manifest in settings['prior_seals']:
        manifest = Path(manifest)
        for name, digest in load(manifest).items():assert native.sha(manifest.parent/name)==digest, name
        bindings = manifest.parent/'bindings.json'
        if bindings.exists():
            for name, digest in load(bindings).items():
                path = Path(name)
                if not path.is_absolute():path = manifest.parent/path
                assert native.sha(path)==digest, str(path)


def build(root, law, arm):
    _, _, native, core, operator = setup(root)
    family, head = arm.rsplit('_', 1);assert head=='clean'
    reference = native.old_build(128, 'ode_k2', 'direct').raw
    if family=='swiglu258':
        from synthetic.models import VelocityNet
        from synthetic.recurrent_depth import copy_surrounding
        torch.manual_seed(0)
        raw = VelocityNet(128, 'swiglu', width=64, depth=3, ffn_hidden=258)
        copy_surrounding(raw, reference)
    else:
        mode={'jet_c':'quadratic','affine_c':'affine','free_c':'free','direct_b':'direct'}[family]
        raw=reference
        for block in raw.blocks:
            for name in ('ode1','ode2'):
                setattr(block.ffn,name,operator.CoefficientJetLayer(getattr(block.ffn,name),mode))
    return core.CappedHead(raw,head)


def generator_visits(model, banks):
    visits=[]
    for name, layer in model.ode_layers():
        p=layer.proj.weight.detach().double().numpy();parameter=layer.coefficient.detach().double().numpy()
        b=parameter if layer.coefficient_mode=='direct' else p@p+parameter
        c=b-p@p if layer.coefficient_mode=='direct' else parameter
        eig=np.linalg.eigvals(p)
        visits.append(dict(layer=name,coefficient_mode=layer.coefficient_mode,
            P_singular_values=np.linalg.svd(p,compute_uv=False).tolist(),
            P_eigenvalues_real=eig.real.tolist(),P_eigenvalues_imag=eig.imag.tolist(),
            reference_B_singular_values=np.linalg.svd(b,compute_uv=False).tolist(),
            C_singular_values=np.linalg.svd(c,compute_uv=False).tolist(),
            P_frobenius=float(np.linalg.norm(p)),reference_B_frobenius=float(np.linalg.norm(b)),
            reference_B_scope="P^2+C; free/affine controls use input-dependent effective coefficients",
            C_frobenius=float(np.linalg.norm(c)),P_C_commutator_frobenius=float(np.linalg.norm(p@c-c@p))))
    return visits


def prepare(root,parent):
    root.mkdir(exist_ok=False);source=root/'source';source.mkdir()
    repo=Path(__file__).resolve().parents[3]
    for name in ('jet_study.py','verify_jet.py','protocol.md','theory.md','research-question-card.md','calibrate.py'):
        shutil.copyfile(Path(__file__).parent/name,source/name)
    shutil.copyfile(repo/'synthetic/coefficient_jet.py',source/'coefficient_operator.py')
    cycle12=Path(load(parent/'settings.json')['parent'])
    cycle11=Path(load(cycle12/'settings.json')['parent'])
    settings=load(cycle11/'settings.json');reference=Path(settings['geometry_reference'])
    shutil.copyfile(reference/'source/study.py',source/'geometry_study.py')
    shutil.copyfile(reference/'source/benchmark_core.py',source/'benchmark_core.py')
    shutil.copyfile(reference/'basis.npy',root/'basis.npy')
    shutil.copyfile(root.parent/'oct28-cycle14-calibration.json',source/'calibration.json')
    settings['prior_seals'] += [str(p/'sealed-runtime.json') for p in (cycle11,cycle12,parent)]
    settings.update(parent_cycle13=str(parent.resolve()),parent_cycle11=str(cycle11),
        registration_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip())
    write(root/'settings.json',settings);_,_,native,_,_=setup(root)
    paths=[p for p in source.iterdir() if p.is_file()]+[root/'settings.json',root/'basis.npy',parent/'sealed-runtime.json',repo/'synthetic/coefficient_jet.py']
    write(root/'bindings.json',{str(p):native.sha(p) for p in paths});verify_inputs(root)
    print(dict(prepared=str(root),new_cells=10,reused_cells=4,bindings=len(paths)),flush=True)


def preflight(root):
    import copy
    settings,geometry,native,core,operator=setup(root);verify_inputs(root)
    from verify_jet import verify
    mathematics=verify(operator,operator.OriginalODELayer);rows=[];common_C={}
    calibration={r['family']:r for r in load(root/'source/calibration.json')['costs']}
    for entry in settings['laws']:
        law=entry['law'];bank=geometry.training_bank(entry,native)
        for arm in ARMS:
            model=build(root,law,arm)
            parent=core.CappedHead(native.old_build(128,'ode_k2','direct').raw,'clean')
            for name,value in parent.state_dict().items():
                if arm=='swiglu258_clean' and '.ffn.' in name: continue
                assert torch.equal(value,model.state_dict()[name]),name
            for name,value in parent.named_buffers(): assert torch.equal(value,dict(model.named_buffers())[name]),name
            initial={}
            if arm!='swiglu258_clean':
                with torch.no_grad():
                    w=torch.linspace(-.02,.03,parent.raw.output.weight.numel()).reshape_as(parent.raw.output.weight)
                    parent.raw.output.weight.copy_(w);model.raw.output.weight.copy_(w)
                with np.load(Path(entry['banks'])/'probe.npz') as data: x,t=data['x'][:3],data['t'][:3]
                x=torch.from_numpy(x).requires_grad_(True);t=torch.from_numpy(t)
                a,b=parent(x,t),model(x,t)
                difference=float((a-b).detach().abs().max())
                if arm!='direct_b_clean': torch.testing.assert_close(a,b,atol=0,rtol=0)
                else: torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
                original=list(parent.named_parameters());new=dict(model.named_parameters())
                ga=torch.autograd.grad(a.square().mean(),(x,*(p for _,p in original)))
                gb=torch.autograd.grad(b.square().mean(),(x,*new.values()))
                gradients=dict(zip(new,gb[1:]));head_difference=float((ga[1+next(i for i,(n,_) in enumerate(original) if n=='raw.output.weight')]-gradients['raw.output.weight']).abs().max())
                if arm!='direct_b_clean':
                    torch.testing.assert_close(ga[0],gb[0],atol=0,rtol=0)
                    for (name,_),g in zip(original,ga[1:]): torch.testing.assert_close(g,gradients[name],atol=0,rtol=0)
                if arm=='jet_c_clean': common_C[law]={n:g.clone() for n,g in gradients.items() if n.endswith('.coefficient')}
                if arm=='free_c_clean':
                    for n,g in common_C[law].items(): torch.testing.assert_close(g,gradients[n],atol=0,rtol=0)
                    for n,g in gradients.items():
                        if n.endswith(('free_weight','free_bias')): assert torch.count_nonzero(g)==0
                initial=dict(initial_prediction_max=difference,initial_output_head_gradient_max=head_difference,
                             inherited_gradients_bitwise=arm!='direct_b_clean')
            ends=[]
            for resume in (False,True):
                model=build(root,law,arm);opt=native.optimizer(model);rng=torch.Generator().manual_seed(100000);digest=native.EMPTY
                for step in range(1,4):
                    x,t,target,noise=native.draw(bank,rng);digest=native.legacy.fingerprint_batch(digest,target,noise,t)
                    geometry.update(model,opt,x,t,target,noise,core)
                    if resume and step==2:
                        state=native.checkpoint(model,opt,rng,step,digest);model=build(root,law,arm);opt=native.optimizer(model)
                        digest=native.restore(model,opt,rng,state)
                ends.append(native.checkpoint(model,opt,rng,3,digest))
            native.equal(*ends);cost=native.compute_budget(model);expected=calibration[arm.rsplit('_',1)[0]]
            for k in ('parameters','dense_forward_macs','dense_training_forward_backward_macs'): assert cost[k]==expected[k],(arm,cost)
            rows.append(dict(law=law,arm=arm,exact_resume=True,**initial,**cost))
    verify_inputs(root);write(root/'preflight.json',dict(status='PASS',mathematics=mathematics,cells=rows))
    write(root/'ready.json',dict(bindings=native.sha(root/'bindings.json'),preflight=native.sha(root/'preflight.json')))
    print(dict(status='PREFLIGHT_PASS',cells=len(rows),mathematics=mathematics),flush=True)


def run(root):
    _, _, native, _, _ = setup(root);verify_inputs(root)
    assert load(root/'ready.json')==dict(bindings=native.sha(root/'bindings.json'), preflight=native.sha(root/'preflight.json'))
    out = root/'output';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    started = time.monotonic();stop = threading.Event();abort = threading.Event();lock = threading.Lock();children = set()
    status = dict(state='RUNNING', completed=0, total=10, pid=os.getpid())
    def progress(**extra):
        status.update(extra, elapsed_seconds=time.monotonic()-started);write(out/'status.json', status);print(status, flush=True)
    def monitor():
        import psutil
        while not stop.is_set():
            parent = psutil.Process();records = []
            for process in [parent]+parent.children(recursive=True):
                try:records.append(dict(pid=process.pid, rss_bytes=process.memory_info().rss, cpu_seconds=sum(process.cpu_times()[:2])))
                except (psutil.NoSuchProcess, psutil.AccessDenied):pass
            append(out/'monitor.jsonl', dict(elapsed=time.monotonic()-started, processes=records));stop.wait(5)
    watcher = threading.Thread(target=monitor, daemon=True);watcher.start();progress()
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    def launch(law, arm):
        command = [sys.executable, str(root/'source/jet_study.py'), 'cell', '--root', str(root), '--law', law, '--arm', arm]
        with (out/f'{law}-{arm}.log').open('w') as stream:
            with lock:
                if abort.is_set():raise RuntimeError('screen aborted')
                process = subprocess.Popen(command, env=environment, stdout=stream, stderr=subprocess.STDOUT);children.add(process)
                append(out/'launches.jsonl', dict(law=law, arm=arm, pid=process.pid, command=command))
            try:code = process.wait(timeout=600)
            except subprocess.TimeoutExpired:process.terminate();process.wait();raise
            finally:
                with lock:children.remove(process)
        assert code==0, (law, arm, code)
        return law+'/'+arm
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(launch, law, arm) for law in ('thin', 'thick') for arm in ARMS]
            try:
                for future in as_completed(futures):progress(completed=status['completed']+1, latest=future.result())
            except BaseException:
                abort.set()
                for future in futures:future.cancel()
                with lock:
                    for child in children:child.terminate()
                raise
        verify_inputs(root);progress(state='COMPLETE')
    except BaseException:progress(state='FAILED', error=traceback.format_exc());raise
    finally:stop.set();watcher.join(timeout=6)


if __name__=='__main__':
    parser = argparse.ArgumentParser();parser.add_argument('command', choices=('prepare', 'preflight', 'run', 'cell'))
    parser.add_argument('--root', type=Path, required=True);parser.add_argument('--parent', type=Path)
    parser.add_argument('--law');parser.add_argument('--arm');args = parser.parse_args();root = args.root.resolve()
    if args.command=='prepare':
        assert args.parent is not None
        prepare(root, args.parent)
    elif args.command=='cell':
        assert args.law in ('thin', 'thick') and args.arm in ARMS
        _, geometry, *_ = setup(root);geometry.cell(root, args.law, args.arm)
    else:globals()[args.command](root)
