"""Read-only, registered complete-hidden-field diagnostic; no training."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

METHODS = ('euler', 'taylor', 'frozen_gate', 'midpoint', 'half_euler')
STEPS = (1000, 2500, 5000)
TIMES = (.1, .3, .5, .7, .9)
POSITIONS = ('1', '2', '3', 'all')


def write(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 2**20):
            h.update(chunk)
    return h.hexdigest()


def append(path, data):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(data, allow_nan=False) + '\n')


def table(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setup(root):
    settings = load(root / 'settings.json')
    sys.path.insert(0, settings['native_root'])
    from synthetic.conservative_readout import build, compute_budget
    from synthetic.train import restore_model
    from scripts import run_recurrent_depth as legacy
    import synthetic.models
    assert Path(synthetic.models.__file__).resolve() == Path(settings['native_root']) / 'synthetic/models.py'
    torch.set_num_threads(1)
    return settings, build, compute_budget, restore_model, legacy


def field(block, h, scale, shift):
    return block.ffn(block.norm(h) * (1 + scale) + shift)


def field_tangent(block, h, scale, shift, direction=None, frozen_gate=False):
    """Primal and analytic J_A v, including LayerNorm and both gate paths."""
    z = block.norm(h) * (1 + scale) + shift
    gate, value = block.ffn.w12(z).chunk(2, -1)
    activation = F.silu(gate)
    primal = block.ffn.w3(activation * value)
    v = primal if direction is None else direction
    centered = h - h.mean(-1, keepdim=True)
    variance = centered.square().mean(-1, keepdim=True) + block.norm.eps
    dz = (v - v.mean(-1, keepdim=True)
          - centered * (centered * v).mean(-1, keepdim=True) / variance)
    dz = dz * torch.rsqrt(variance) * block.norm.weight * (1 + scale)
    wg, wv = block.ffn.w12.weight.chunk(2, 0)
    if frozen_gate:
        product_tangent = activation * F.linear(dz, wv)
    else:
        dgate, dvalue = F.linear(dz, block.ffn.w12.weight).chunk(2, -1)
        sigmoid = torch.sigmoid(gate)
        derivative = sigmoid * (1 + gate * (1 - sigmoid))
        product_tangent = derivative * dgate * value + activation * dvalue
    tangent = F.linear(product_tangent, block.ffn.w3.weight)
    return primal, tangent


class Intervention(nn.Module):
    def __init__(self, native, method, position):
        super().__init__()
        self.raw, self.dim = native.raw, native.dim
        self.method, self.position = method, position

    def forward(self, x, t, *, updates=3, step_size=None):
        assert updates == 3 and step_size is None
        raw = self.raw
        temb = raw.time(t)
        h = raw.input(x)
        for i, block in enumerate(raw.blocks):
            selected = self.method != 'euler' and (self.position == 'all' or self.position == str(i + 1))
            if not selected:
                h = block(h, temb)
                continue
            scale, shift = block.condition(temb).chunk(2, -1)
            c = block.residual_scale
            if self.method in ('taylor', 'frozen_gate'):
                a, tangent = field_tangent(block, h, scale, shift, frozen_gate=self.method == 'frozen_gate')
                h = h + c * a + .5 * c * c * tangent
            else:
                a = field(block, h, scale, shift)
                midpoint = h + .5 * c * a
                b = field(block, midpoint, scale, shift)
                h = h + c * b if self.method == 'midpoint' else midpoint + .5 * c * b
        return raw.output(raw.norm(h))


def prepare(root, repo, base):
    root.mkdir(parents=True, exist_ok=False)
    source = root / 'source'
    source.mkdir()
    for name in ('run_cycle.py', 'protocol.md'):
        shutil.copyfile(Path(__file__).parent / name, source / name)
    native = base / 'conservative_readout_20261026'
    joint = base / 'joint_dependence_20261026'
    settings = dict(native_root=str(native / 'source-v2'), repo=str(repo), model_seed=0,
                    upstream_revision='228791fb499afffb54b46200aca536f79142f117',
                    base_commit='2d2259a8e34fb15d48e57f9492f18f4d026e57c8', wall_limit_seconds=600,
                    methods=METHODS, steps=STEPS, positions=POSITIONS, cohort=[])
    for dataset in ('gaussian_aniso_2d', 'gmm_ring8_2d', 'rotated_product_bimodal_16d', 'sign_code'):
        if dataset == 'sign_code':
            folder, banks = joint / 'output/runs/swiglu128', joint / 'banks'
        else:
            folder = native / 'output/runs' / dataset / 'swiglu128_direct'
            banks = native / 'output/banks' / dataset
        settings['cohort'].append(dict(dataset=dataset, dim=16 if dataset in ('sign_code', 'rotated_product_bimodal_16d') else 2,
                                       folder=str(folder), banks=str(banks)))
    write(root / 'settings.json', settings)
    paths = list((native / 'source-v2').rglob('*.py')) + list(source.iterdir())
    paths += [root / 'settings.json']
    for entry in settings['cohort']:
        folder, banks = Path(entry['folder']), Path(entry['banks'])
        paths.extend(folder / f'step_{step:07d}.pt' for step in STEPS)
        paths.extend(folder / name for name in ('result.json', 'metrics.jsonl', 'status.json'))
        paths.extend(banks.glob('*.npz'))
    paths.extend(Path(settings['repo']) / p for p in (
        'docs/autoresearch/program.md',
        'docs/research-joint-dependence-2026-10-26/next-question.md',
        'docs/research-architecture-search/coupled-generator-prior-review.md',
        'docs/research-architecture-search/protocol-state-clock.md',
        'docs/research-recurrent-depth-2026-10-26/protocol.md'))
    write(root / 'bindings.json', {str(p): sha(p) for p in sorted(set(paths))})
    print(dict(prepared=str(root), bindings=len(set(paths))), flush=True)


def verify_bindings(root):
    bindings = load(root / 'bindings.json')
    for path, expected in bindings.items():
        assert sha(path) == expected, path
    return len(bindings)


def restored(build, restore, entry, step):
    model = build(entry['dim'], 'swiglu128', 'direct').eval()
    state = torch.load(Path(entry['folder']) / f'step_{step:07d}.pt', map_location='cpu', weights_only=False)
    restore(model, state)
    return model


def preflight(root):
    _, build, costs, restore, legacy = setup(root)
    verify_bindings(root)
    checks = []
    try:
        for entry in load(root / 'settings.json')['cohort']:
            with np.load(Path(entry['banks']) / 'probe.npz') as bank:
                x, t = torch.from_numpy(bank['x'][:8]), torch.from_numpy(bank['t'][:8])
            records = [json.loads(line) for line in (Path(entry['folder']) / 'metrics.jsonl').read_text().splitlines()]
            for step in STEPS:
                model = restored(build, restore, entry, step)
                with torch.no_grad():
                    native, wrapped = model(x, t), Intervention(model, 'euler', 'all')(x, t)
                assert torch.equal(native, wrapped)
                observed = legacy.field_evaluation(model, Path(entry['banks']))
                record = next(r for r in records if r['step'] == step)
                for key in ('oracle_mse', 'pair_mse', 'prediction_rms'):
                    np.testing.assert_allclose(observed[key], record[key], rtol=2e-6, atol=1e-9)
                h, temb = model.raw.input(x).detach(), model.raw.time(t).detach()
                tangents = []
                for block in model.raw.blocks:
                    b = copy.deepcopy(block).double()
                    hh, tt = h.double(), temb.double()
                    scale, shift = b.condition(tt).chunk(2, -1)
                    function = lambda state: field(b, state, scale, shift)
                    primal = function(hh)
                    _, independent = torch.autograd.functional.jvp(function, hh, primal, create_graph=False)
                    actual_primal, actual = field_tangent(b, hh, scale, shift)
                    _, frozen = field_tangent(b, hh, scale, shift, frozen_gate=True)
                    torch.testing.assert_close(primal, actual_primal, rtol=1e-12, atol=1e-12)
                    torch.testing.assert_close(actual, independent, rtol=1e-9, atol=1e-10)
                    # Independent gate-path decomposition using AD on the gate alone.
                    z = b.norm(hh) * (1 + scale) + shift
                    _, dz = torch.autograd.functional.jvp(lambda state: b.norm(state) * (1 + scale) + shift, hh, primal)
                    gate, value = b.ffn.w12(z).chunk(2, -1)
                    wg = b.ffn.w12.weight.chunk(2, 0)[0]
                    _, dactivation = torch.autograd.functional.jvp(F.silu, gate, F.linear(dz, wg))
                    gate_path = F.linear(dactivation * value, b.ffn.w3.weight)
                    torch.testing.assert_close(actual - frozen, gate_path, rtol=1e-8, atol=1e-10)
                    # A tangent receives no constant output bias of its own.
                    arbitrary = torch.cos(hh)
                    _, before = field_tangent(b, hh, scale, shift, direction=arbitrary)
                    with torch.no_grad():
                        b.ffn.w3.bias.add_(.75)
                    _, after = field_tangent(b, hh, scale, shift, direction=arbitrary)
                    assert torch.equal(before, after)
                    tangents.append(dict(max_absolute_error=float((actual-independent).abs().max().detach()),
                                         gate_path_rms=float(gate_path.square().mean().sqrt().detach())))
                    with torch.no_grad():
                        h = block(h, temb)
                # A zero duration is exactly the identity for each intervention.
                zero = copy.deepcopy(model)
                for b in zero.raw.blocks:
                    b.residual_scale = 0.
                with torch.no_grad():
                    for method in METHODS:
                        assert torch.equal(zero(x,t), Intervention(zero, method, 'all')(x,t))
                checks.append(dict(dataset=entry['dataset'], step=step, baseline=observed, tangents=tangents))
        assert any(x['gate_path_rms'] > 1e-6 for row in checks for x in row['tangents'])
        verify_bindings(root)
        write(root / 'preflight.json', dict(status='PASS', checks=checks, checkpoints=len(checks),
                                           derivative_checks=sum(len(x['tangents']) for x in checks)))
        print(dict(status='PASS', checkpoints=len(checks), derivative_checks=36), flush=True)
    except BaseException:
        write(root / 'preflight-failure.json', dict(error=traceback.format_exc(), completed=checks))
        raise


def run(root):
    settings, build, compute_budget, restore, legacy = setup(root)
    assert load(root / 'preflight.json')['status'] == 'PASS'
    bindings = verify_bindings(root)
    out = root / 'output'
    out.mkdir(exist_ok=False)
    (out / 'predictions').mkdir()
    started = time.monotonic()
    stop = threading.Event()
    status = dict(state='RUNNING', pid=os.getpid(), field_cases=0, panels=0, sampling=0)
    metrics, panels, sampling, costs = [], [], [], []

    def heartbeat():
        import psutil
        process = psutil.Process()
        while not stop.is_set():
            append(out / 'monitor.jsonl', dict(elapsed=time.monotonic()-started, pid=os.getpid(),
                    rss_bytes=process.memory_info().rss, cpu_seconds=sum(process.cpu_times()[:2])))
            stop.wait(2)

    def progress(**kwargs):
        status.update(kwargs, elapsed_seconds=time.monotonic()-started)
        write(out / 'status.json', status)
        assert status['elapsed_seconds'] < settings['wall_limit_seconds'], 'registered timeout'

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    progress()
    try:
        for entry in settings['cohort']:
            banks = Path(entry['banks'])
            for step in STEPS:
                model = restored(build, restore, entry, step)
                immutable = {k: v.detach().clone() for k,v in model.state_dict().items()}
                buffers = {k: v.detach().clone() for k,v in model.named_buffers()}
                for method in METHODS:
                    positions = ('all',) if method == 'euler' else POSITIONS
                    for position in positions:
                        wrapper = Intervention(model, method, position)
                        tag = f"{entry['dataset']}-{step}-{method}-{position}"
                        row = dict(dataset=entry['dataset'], step=step, method=method, position=position)
                        row.update(legacy.field_evaluation(wrapper, banks, save=out/'predictions'/(tag+'.npz')))
                        metrics.append(row)
                        append(out / 'field.jsonl', row)
                        progress(field_cases=len(metrics), current=tag)
                        if step == 5000 and position == 'all':
                            costs.append(dict(dataset=entry['dataset'], method=method, **compute_budget(wrapper)))
                            for clock in TIMES:
                                with np.load(banks / f'panel_{clock:.1f}.npz') as bank:
                                    x,t,oracle,jj = [bank[k] for k in ('x','t','oracle','oracle_jacobian')]
                                value, jac = legacy.jacobian(wrapper, torch.from_numpy(x), torch.from_numpy(t))
                                panel = dict(dataset=entry['dataset'], method=method, time=clock,
                                    oracle_mse=float(np.mean((value.numpy()-oracle)**2)),
                                    jacobian_error=float(np.mean(np.sum((jac.numpy()-jj)**2,axis=(1,2)))))
                                panels.append(panel)
                                np.savez_compressed(out/'predictions'/f'{tag}-panel-{clock:.1f}.npz', prediction=value.numpy(),jacobian=jac.numpy())
                                append(out / 'panels.jsonl', panel)
                                progress(panels=len(panels))
                            with np.load(banks / 'sampling.npz') as bank:
                                source,target,target2,directions = [torch.from_numpy(bank[k]) for k in ('source','target','target2','directions')]
                            generated = legacy.generated_samples(wrapper, source, 32)
                            assert torch.isfinite(generated).all()
                            sample = dict(dataset=entry['dataset'], method=method,
                                sw2=legacy.sliced_w2(generated,target,directions),
                                reference_floor=legacy.sliced_w2(target,target2,directions))
                            sampling.append(sample)
                            np.savez_compressed(out/'predictions'/f'{tag}-samples.npz', generated=generated.numpy())
                            append(out / 'sampling.jsonl', sample)
                            progress(sampling=len(sampling))
                for key, value in model.state_dict().items():
                    assert torch.equal(value,immutable[key]),key
                for key, value in model.named_buffers():
                    assert torch.equal(value,buffers[key]),key
                print(dict(dataset=entry['dataset'], step=step, field_cases=len(metrics), elapsed=time.monotonic()-started),flush=True)
        assert len(metrics)==204 and len(panels)==100 and len(sampling)==20
        for name,rows in [('field',metrics),('panels',panels),('sampling',sampling),('costs',costs)]:
            table(out / (name+'.csv'), rows)
        verify_bindings(root)
        progress(state='COMPLETE', bindings_unchanged=bindings)
    except BaseException:
        write(out / 'failure.json', dict(error=traceback.format_exc()))
        status.update(state='FAILED', elapsed_seconds=time.monotonic()-started)
        write(out / 'status.json', status)
        raise
    finally:
        stop.set()
        thread.join(timeout=3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare','preflight','run'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--repo',type=Path)
    parser.add_argument('--base',type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        assert args.repo and args.base
        prepare(args.root,args.repo,args.base)
    elif args.action == 'preflight':
        preflight(args.root)
    else:
        run(args.root)
