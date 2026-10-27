"""One conditional-frame screen with unchanged native geometry training/eval."""
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

ARMS = ('frame_k2_clean', 'global_k2_clean', 'frame_k1_clean', 'swiglu257_clean')


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
    operator = import_file('frame_operator', root/'source/frame_operator.py')
    geometry.build = build;geometry.generator_visits = generator_visits
    return settings, geometry, native, core, operator


def verify_inputs(root):
    settings, _, native, _, _ = setup(root)
    for name, digest in load(root/'bindings.json').items():assert native.sha(Path(name))==digest, name
    parent = Path(settings['parent_cycle10'])
    prior = import_file('derivative_study', parent/'source/derivative_study.py')
    prior.verify_inputs(parent)
    for name, digest in load(parent/'sealed-runtime.json').items():assert native.sha(parent/name)==digest, name


def build(root, law, arm):
    _, _, native, core, operator = setup(root)
    family, head = arm.rsplit('_', 1);assert head=='clean'
    reference = native.old_build(128, 'ode_k2', 'direct').raw
    if family=='swiglu257':
        from synthetic.models import VelocityNet
        from synthetic.recurrent_depth import copy_surrounding
        torch.manual_seed(0)
        raw = VelocityNet(128, 'swiglu', width=64, depth=3, ffn_hidden=257)
        copy_surrounding(raw, reference)
    elif family in ('frame_k2', 'global_k2', 'frame_k1'):
        raw = reference
        for block in raw.blocks:
            for name in ('ode1', 'ode2'):
                layer = getattr(block.ffn, name)
                setattr(block.ffn, name, operator.ConditionalFrameLayer(layer,
                    'global' if family=='global_k2' else 'state', 1 if family=='frame_k1' else 2))
    else:
        raise ValueError(f'unsupported arm {arm}')
    return core.CappedHead(raw, head)


def generator_visits(model, banks):
    visits = []
    for name, layer in model.ode_layers():
        p = layer.proj.weight.detach().double().numpy();eig = np.linalg.eigvals(p)
        visits.append(dict(layer=name, P_singular_values=np.linalg.svd(p, compute_uv=False).tolist(),
            P_eigenvalues_real=eig.real.tolist(), P_eigenvalues_imag=eig.imag.tolist(),
            P_frobenius=float(np.linalg.norm(p)), frame_mode=layer.frame_mode,
            frame_weight_frobenius=float(layer.frame_weight.detach().norm()), frame_bias_norm=float(layer.frame_bias.detach().norm())))
    return visits


def prepare(root, parent):
    root.mkdir(exist_ok=False);source = root/'source';source.mkdir()
    repo = Path(__file__).resolve().parents[3]
    for name in ('frame_study.py', 'verify_frame.py', 'protocol.md', 'theory.md', 'research-question-card.md'):
        shutil.copyfile(Path(__file__).parent/name, source/name)
    shutil.copyfile(repo/'synthetic/conditional_frame.py', source/'frame_operator.py')
    cycle09 = Path(load(parent/'settings.json')['cycle09'])
    cycle08 = Path(load(cycle09/'settings.json')['cycle08'])
    reference = Path(load(cycle08/'settings.json')['cycle07'])
    shutil.copyfile(reference/'source/study.py', source/'geometry_study.py')
    shutil.copyfile(reference/'source/benchmark_core.py', source/'benchmark_core.py')
    shutil.copyfile(reference/'basis.npy', root/'basis.npy')
    settings = load(reference/'settings.json')
    settings.update(parent_cycle10=str(parent.resolve()), geometry_reference=str(reference),
        registration_commit=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip())
    write(root/'settings.json', settings)
    _, _, native, _, _ = setup(root)
    paths = [p for p in source.iterdir() if p.is_file()]+[root/'settings.json', root/'basis.npy', parent/'sealed-runtime.json', repo/'synthetic/conditional_frame.py']
    write(root/'bindings.json', {str(p): native.sha(p) for p in paths});verify_inputs(root)
    print(dict(prepared=str(root), new_cells=8, reused_cells=4, bindings=len(paths)), flush=True)


def preflight(root):
    import copy
    settings, geometry, native, core, operator = setup(root);verify_inputs(root)
    from verify_frame import verify
    mathematics = verify(operator, operator.OriginalODELayer);rows = []
    for entry in settings['laws']:
        law = entry['law'];bank = geometry.training_bank(entry, native)
        for arm in ARMS:
            model = build(root, law, arm)
            parent = core.CappedHead(native.old_build(128, 'ode_k2', 'direct').raw, 'clean')
            if arm!='swiglu257_clean':
                for name, value in parent.state_dict().items():assert torch.equal(value, model.state_dict()[name]), name
                for name, value in parent.named_buffers():assert torch.equal(value, dict(model.named_buffers())[name]), name
                if arm!='frame_k1_clean':
                    with torch.no_grad():
                        parent.raw.output.weight.copy_(torch.linspace(-.02, .03, parent.raw.output.weight.numel()).reshape_as(parent.raw.output.weight))
                        model.raw.output.weight.copy_(parent.raw.output.weight)
                    with np.load(Path(entry['banks'])/'probe.npz') as data:x, t = data['x'][:3], data['t'][:3]
                    torch.testing.assert_close(model(torch.from_numpy(x), torch.from_numpy(t)),
                        parent(torch.from_numpy(x), torch.from_numpy(t)), atol=0, rtol=0)
                    active_x = torch.from_numpy(x).requires_grad_(True);active_t = torch.from_numpy(t)
                    common = dict(model.named_parameters())
                    inherited = list(parent.named_parameters())
                    old_gradients = torch.autograd.grad(parent(active_x, active_t).square().mean(),
                        (active_x, *(p for _, p in inherited)))
                    new_gradients = torch.autograd.grad(model(active_x, active_t).square().mean(),
                        (active_x, *(common[name] for name, _ in inherited)))
                    for left, right in zip(old_gradients, new_gradients):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
            ends = []
            for resume in (False, True):
                model = build(root, law, arm);opt = native.optimizer(model)
                rng = torch.Generator().manual_seed(100000);digest = native.EMPTY
                for step in range(1, 4):
                    x, t, target, source = native.draw(bank, rng)
                    digest = native.legacy.fingerprint_batch(digest, target, source, t)
                    geometry.update(model, opt, x, t, target, source, core)
                    if resume and step==2:
                        state = native.checkpoint(model, opt, rng, step, digest)
                        model = build(root, law, arm);opt = native.optimizer(model)
                        digest = native.restore(model, opt, rng, state)
                ends.append(native.checkpoint(model, opt, rng, 3, digest))
            native.equal(*ends)
            cost = native.compute_budget(model)
            assert cost['parameters']==({'frame_k2_clean':122880, 'global_k2_clean':122880,
                'frame_k1_clean':122880, 'swiglu257_clean':148098}[arm]), (arm, cost)
            expected = {'frame_k2_clean':(145472,426112), 'global_k2_clean':(145472,401920),
                'frame_k1_clean':(120896,352384), 'swiglu257_clean':(145664,426688)}[arm]
            assert (cost['dense_forward_macs'],cost['dense_training_forward_backward_macs'])==expected
            rows.append(dict(law=law, arm=arm, exact_resume=True, **cost))
    verify_inputs(root)
    write(root/'preflight.json', dict(status='PASS', mathematics=mathematics, cells=rows))
    write(root/'ready.json', dict(bindings=native.sha(root/'bindings.json'), preflight=native.sha(root/'preflight.json')))
    print(dict(status='PASS', mathematics=mathematics, cells=rows), flush=True)


def run(root):
    _, _, native, _, _ = setup(root);verify_inputs(root)
    assert load(root/'ready.json')==dict(bindings=native.sha(root/'bindings.json'), preflight=native.sha(root/'preflight.json'))
    out = root/'output';out.mkdir(exist_ok=False);(out/'runs').mkdir()
    started = time.monotonic();stop = threading.Event();abort = threading.Event();lock = threading.Lock();children = set()
    status = dict(state='RUNNING', completed=0, total=8, pid=os.getpid())
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
        command = [sys.executable, str(root/'source/frame_study.py'), 'cell', '--root', str(root), '--law', law, '--arm', arm]
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
