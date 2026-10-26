"""Monitored native-FM screen, importing immutable historical data and models."""
from __future__ import annotations

import argparse
import copy
import hashlib
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

BASE = Path('/Users/mr.z/ode-workspace-prep/architecture-search')
OLD = BASE / 'conservative_readout_20261026'
NATIVE = OLD / 'source-v2'
DOC = Path('/Users/mr.z/ODE-FFN-report-work/docs/research-cubic-split-2026-10-26')
sys.path.insert(0, str(NATIVE))
from scripts import run_recurrent_depth as legacy
from synthetic.conservative_readout import build as old_build, compute_budget
from synthetic.recurrent_depth import copy_surrounding
from synthetic.models import VelocityNet
from synthetic.train import restore_model
from cubic_core import SplitCore, FreeISRU, explicit_split, input_jacobian, cubic_flow

NEW = ('cubic_split', 'silu_split', 'free_isru', 'swiglu64', 'swiglu160')
REFERENCES = ('ode_k2', 'swiglu128')
TIMES = (.1, .3, .5, .7, .9)
SNAPSHOTS = (0, 1000, 2500, 5000)
EMPTY = hashlib.sha256(b'').hexdigest()
load = lambda p: json.loads(Path(p).read_text())
write, append, sha = legacy.write, legacy.append, legacy.sha256


def build(dim: int, family: str) -> torch.nn.Module:
    if family in REFERENCES:
        return old_build(dim, family, 'direct')
    model = old_build(dim, 'ode_k2', 'direct')
    if family.startswith('swiglu'):
        torch.manual_seed(0)
        raw = VelocityNet(dim, 'swiglu', width=64, depth=3, ffn_hidden=int(family[6:]))
        copy_surrounding(raw, model.raw)
        model.raw = raw
    elif family in ('cubic_split', 'silu_split', 'free_isru'):
        for block in model.raw.blocks:
            original = block.ffn.ode1
            block.ffn = (FreeISRU(original) if family == 'free_isru' else
                         SplitCore(original, 'cubic' if family == 'cubic_split' else 'silu'))
    else:
        raise ValueError(family)
    return model


def optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.)


def reference(dataset: str, family: str) -> Path:
    return OLD / 'output/runs' / dataset / (family + '_direct')


def targets(dataset: str) -> torch.Tensor:
    root, spec, _ = legacy.load_data(load(OLD / 'output/study.json'), dataset)
    return torch.from_numpy(legacy.load_targets(root / dataset / 'train.npz', spec['dim']))


def draw(bank: torch.Tensor, generator: torch.Generator) -> tuple:
    target = bank[torch.randint(len(bank), (512,), generator=generator)]
    source = torch.randn(target.shape, generator=generator)
    clock = torch.rand((512, 1), generator=generator)
    return (1 - clock) * source + clock * target, clock, target, source


def update(model, opt, x, clock, target, source) -> tuple[float, float]:
    opt.zero_grad(set_to_none=True)
    loss = (model(x, clock) - (target - source)).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('nonfinite loss')
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(grad):
        raise FloatingPointError('nonfinite gradient')
    opt.step()
    return float(loss.detach()), float(grad)


def state(model, opt, generator, step: int, digest: str) -> dict:
    return copy.deepcopy(dict(step=step, model=model.state_dict(), optimizer=opt.state_dict(),
                runtime_buffers={k: v.detach().clone() for k, v in model.named_buffers()},
                torch_rng=torch.get_rng_state(), data_rng=generator.get_state(), input_stream_digest=digest))


def restore(model, opt, generator, checkpoint: dict) -> str:
    restore_model(model, checkpoint)
    opt.load_state_dict(checkpoint['optimizer'])
    generator.set_state(checkpoint['data_rng'])
    torch.set_rng_state(checkpoint['torch_rng'])
    return checkpoint['input_stream_digest']


def equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right): equal(a, b)
    else:
        assert left == right, (left, right)


def statistics(model, banks: Path) -> list[dict]:
    visits, hooks = {}, []
    for j, block in enumerate(model.raw.blocks):
        def capture(module, args, result, j=j):
            visits[j] = args[0].detach().clone()
        hooks.append(block.ffn.register_forward_hook(capture))
    try:
        with np.load(banks / 'probe.npz') as data, torch.no_grad():
            model(torch.from_numpy(data['x'][:32]), torch.from_numpy(data['t'][:32]))
    finally:
        for hook in hooks: hook.remove()
    rows = []
    for j, block in enumerate(model.raw.blocks):
        module = block.ffn
        row = dict(position=j)
        if isinstance(module, SplitCore):
            p = module.proj.weight.detach().double()
            with torch.no_grad():
                h = visits[j]
                a = module.duration(h)
                u = module.half(h, a)
                v = cubic_flow(u, a) if module.response == 'cubic' else torch.nn.functional.silu(u)
            row.update(P_singular_values=torch.linalg.svdvals(p).tolist(),
                       P_symmetric_eigenvalues=torch.linalg.eigvalsh(.5 * (p + p.T)).tolist(),
                       duration_min=float(a.min()), duration_max=float(a.max()), duration_mean=float(a.mean()),
                       nonlinear_change_rms_over_input=float((v - u).square().mean().sqrt() / h.square().mean().sqrt()),
                       output_rms_over_input=float(module(h).detach().square().mean().sqrt() / h.square().mean().sqrt()))
        rows.append(row)
    return rows


def preflight(root: Path) -> None:
    torch.set_num_threads(1)
    checks, costs, derivative_checks = [], {}, []
    for dataset in legacy.DATASETS:
        bank = targets(dataset)
        dim = bank.shape[1]
        parent = build(dim, 'ode_k2')
        for family in (*NEW, *REFERENCES):
            model = build(dim, family)
            for getter in ('named_parameters', 'named_buffers'):
                a = {k: v for k, v in getattr(model, getter)() if '.ffn.' not in k}
                b = {k: v for k, v in getattr(parent, getter)() if '.ffn.' not in k}
                equal(a, b)
            costs[f'{dataset}/{family}'] = compute_budget(model)
            if family in REFERENCES:
                saved = torch.load(reference(dataset, family) / 'step_0000000.pt', weights_only=False)
                equal(model.state_dict(), saved['model'])
                equal(dict(model.named_buffers()), saved['runtime_buffers'])
                equal(optimizer(model).state_dict(), saved['optimizer'])
                checks.append(dict(dataset=dataset, family=family, historical_initial_state=True))
            else:
                histories = []
                for resume in (False, True):
                    model = build(dim, family)
                    opt = optimizer(model)
                    rng = torch.Generator().manual_seed(100000)
                    digest = EMPTY
                    for step in range(1, 4):
                        x, clock, target, source = draw(bank, rng)
                        digest = legacy.fingerprint_batch(digest, target, source, clock)
                        update(model, opt, x, clock, target, source)
                        if resume and step == 2:
                            saved = state(model, opt, rng, step, digest)
                            model = build(dim, family); opt = optimizer(model)
                            digest = restore(model, opt, rng, saved)
                    histories.append(state(model, opt, rng, 3, digest))
                equal(*histories)
                checks.append(dict(dataset=dataset, family=family, exact_resume=True))
        for family in ('cubic_split', 'silu_split'):
            model = build(dim, family)
            for j, block in enumerate(model.raw.blocks):
                module = copy.deepcopy(block.ffn).double()
                # Deliberately nonconstant clock, beyond the nearly constant initialization.
                with torch.no_grad(): module.t_func[-1].weight.mul_(9.)
                x = torch.linspace(-.8, .9, 3 * 64, dtype=torch.float64).reshape(3, 64).requires_grad_(True)
                native = module(x); explicit = explicit_split(module, x)
                torch.testing.assert_close(native, explicit, rtol=1e-11, atol=1e-12)
                jac = torch.stack([torch.autograd.grad(native[:, k].sum(), x, retain_graph=True)[0]
                                   for k in range(64)], dim=1)
                manual = input_jacobian(module, x)
                torch.testing.assert_close(jac, manual, rtol=1e-10, atol=1e-11)
                parameters = tuple(module.parameters())
                gn = torch.autograd.grad(native.square().mean(), parameters)
                ge = torch.autograd.grad(explicit.square().mean(), parameters)
                for a, b in zip(gn, ge): torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)
                derivative_checks.append(dict(dataset=dataset, family=family, position=j,
                    forward_max=float((native - explicit).detach().abs().max()),
                    jacobian_max=float((jac - manual).detach().abs().max()),
                    adjoint_max=max(float((a - b).abs().max()) for a, b in zip(gn, ge))))
    # Independent RK4 integration checks second-order consistency of the specified fixed field.
    p = torch.tensor([[.2, -.1], [.3, -.2]], dtype=torch.float64)
    h = torch.tensor([.4, -.6], dtype=torch.float64)
    errors = []
    for duration in (.2, .1, .05):
        b = torch.eye(2, dtype=h.dtype) + .5 * duration * p + .125 * duration**2 * (p @ p)
        approximation = b @ cubic_flow(b @ h, duration)
        y = h.clone(); dt = duration / 256
        field = lambda z: p @ z - z.pow(3)
        for _ in range(256):
            k1 = field(y); k2 = field(y + .5 * dt * k1)
            k3 = field(y + .5 * dt * k2); k4 = field(y + dt * k3)
            y = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        errors.append(float((approximation - y).norm()))
    ratios = [errors[k] / errors[k + 1] for k in (0, 1)]
    assert all(6 < ratio < 10 for ratio in ratios), ratios
    z = torch.linspace(-2, 2, 17, dtype=torch.float64)
    torch.testing.assert_close(cubic_flow(cubic_flow(z, .2), .3), cubic_flow(z, .5), rtol=1e-13, atol=1e-14)
    write(root / 'preflight.json', dict(status='PASS', paired_surrounding_and_resume=checks,
          derivatives=derivative_checks, rk4_errors=errors, halving_ratios=ratios,
          cubic_subflow_semigroup=True, costs=costs))


def cell(root: Path, dataset: str, family: str) -> None:
    torch.set_num_threads(1)
    folder = root / 'output/runs' / dataset / family
    folder.mkdir(parents=True, exist_ok=False)
    started = timer.monotonic()
    def status(state_name='RUNNING', step=0, **extra):
        write(folder / 'status.json', dict(state=state_name, step=step, pid=os.getpid(),
              elapsed_seconds=timer.monotonic() - started, **extra))
    status()
    try:
        bank = targets(dataset); model = build(bank.shape[1], family); opt = optimizer(model)
        rng = torch.Generator().manual_seed(100000); digest = EMPTY
        banks = OLD / 'output/banks' / dataset
        costs = compute_budget(model)
        write(folder / 'provenance.json', dict(dataset=dataset, family=family, seed=0,
            stream_generator_initialization=100000, steps=5000, batch=512, costs=costs,
            actual_device='cpu', source=str(Path(__file__).resolve())))
        def save(step):
            torch.save(state(model, opt, rng, step, digest), folder / f'step_{step:07d}.pt')
            write(folder / f'statistics_{step:07d}.json', statistics(model, banks))
            return legacy.field_evaluation(model, banks, save=folder / f'field_{step:07d}.npz')
        append(folder / 'metrics.jsonl', dict(step=0, **save(0)))
        loss_sum = grad_sum = 0.; count = clipped = 0
        for step in range(1, 5001):
            x, clock, target, source = draw(bank, rng)
            digest = legacy.fingerprint_batch(digest, target, source, clock)
            loss, grad = update(model, opt, x, clock, target, source)
            loss_sum += loss; grad_sum += grad; count += 1; clipped += int(grad > 1)
            if step % 100 == 0: status(step=step)
            if step % 500 == 0:
                row = dict(step=step, train_loss=loss_sum/count, grad_norm=grad_sum/count,
                           clip_fraction=clipped/count, wall_seconds=timer.monotonic()-started)
                if step in SNAPSHOTS: row.update(save(step))
                append(folder / 'metrics.jsonl', row); print(row, flush=True)
                loss_sum = grad_sum = 0.; count = clipped = 0
        diagnostics = legacy.diagnostics(model, banks, folder / 'diagnostics', rollout=False)
        write(folder / 'result.json', dict(dataset=dataset, family=family, reused=False,
              input_stream_digest=digest, **costs, **diagnostics, elapsed_seconds=timer.monotonic()-started))
        status('COMPLETE', 5000)
    except BaseException:
        status('FAILED', error=traceback.format_exc()); raise


def run(root: Path) -> None:
    torch.set_num_threads(1)
    ready = load(root / 'ready.json'); assert load(root / 'preflight.json')['status'] == 'PASS'
    for path, digest in ready['bindings'].items(): assert sha(path) == digest, path
    out = root / 'output'; out.mkdir(exist_ok=False)
    (out / 'runs').mkdir(); (out / 'reused').mkdir()
    for dataset in legacy.DATASETS:
        for family in REFERENCES:
            folder = reference(dataset, family)
            write(out / 'reused' / f'{dataset}--{family}.json', dict(folder=str(folder),
                  files_sha256={str(p.relative_to(folder)): sha(p) for p in folder.rglob('*') if p.is_file()}))
    started = timer.monotonic(); stop = threading.Event(); abort = threading.Event(); lock = threading.Lock(); children = set()
    progress = dict(state='RUNNING', completed=0, total_new=15, historical=6, pid=os.getpid())
    def status(**extra):
        progress.update(extra, elapsed_seconds=timer.monotonic()-started)
        write(out / 'status.json', progress); print(progress, flush=True)
    def monitor():
        import psutil
        while not stop.is_set():
            row = dict(time=legacy.now(), processes=[])
            parent = psutil.Process()
            for p in [parent] + parent.children(recursive=True):
                try:
                    row['processes'].append(dict(pid=p.pid, rss_bytes=p.memory_info().rss,
                        cpu_seconds=sum(p.cpu_times()[:2]), command=p.cmdline()))
                except (psutil.NoSuchProcess, psutil.AccessDenied): pass
            append(out / 'monitor.jsonl', row); write(out / 'monitor.json', row); stop.wait(5)
    watcher = threading.Thread(target=monitor, daemon=True); watcher.start(); status()
    env = os.environ.copy(); env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    def launch(job):
        dataset, family = job
        command = [sys.executable, str(Path(__file__)), 'cell', '--root', str(root), '--dataset', dataset, '--family', family]
        with (out / f'{dataset}--{family}.log').open('w') as log:
            with lock:
                if abort.is_set(): raise RuntimeError('study aborted')
                p = subprocess.Popen(command, env=env, cwd=Path(__file__).parent, stdout=log, stderr=subprocess.STDOUT)
                children.add(p); append(out / 'launches.jsonl', dict(dataset=dataset, family=family, pid=p.pid, command=command))
            code = p.wait()
            with lock: children.remove(p)
        if code: raise RuntimeError(f'{dataset}/{family} exited {code}')
        return job
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(launch, (d, f)) for d in legacy.DATASETS for f in NEW]
            try:
                for future in as_completed(futures): status(completed=progress['completed']+1, latest=future.result())
            except BaseException:
                abort.set()
                for future in futures: future.cancel()
                with lock:
                    for p in children: p.terminate()
                raise
        for path, digest in ready['bindings'].items(): assert sha(path) == digest, path
        status(state='COMPLETE')
    except BaseException:
        status(state='FAILED', error=traceback.format_exc()); raise
    finally:
        stop.set(); watcher.join(timeout=6)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preflight', 'cell', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset', choices=legacy.DATASETS)
    parser.add_argument('--family', choices=NEW)
    args = parser.parse_args()
    if args.action == 'preflight': preflight(args.root)
    elif args.action == 'run': run(args.root)
    else: cell(args.root, args.dataset, args.family)
