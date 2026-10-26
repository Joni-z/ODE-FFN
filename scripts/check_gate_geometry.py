"""Anchored input-gate decomposition of complete synthetic velocity Jacobians.

Read-only checkpoints, analytic Gaussian-mixture oracles and persistent telemetry.
This measures local derivatives, not generation quality or a retrained ablation.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime
import json
import os
from pathlib import Path
import platform
try:
    import resource
except ImportError:  # Native Windows has no resource module.
    resource = None
import sys
import threading
import time
from typing import Iterator

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synthetic.data import GaussianMixture,RotatedGaussianMixture,RotatedProductMixture
from synthetic.models import VelocityNet
from synthetic.train import load_distribution, load_targets, sha256

KINDS = ('ode_original', 'ode_fixed', 'ode_scalar', 'ffn', 'swiglu')
DATASETS = ('gaussian_aniso_2d', 'gmm_ring8_2d')
TIMES = (.1, .3, .5, .7, .9)


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@contextmanager
def gate_context(model: VelocityNet, values: dict[str, torch.Tensor],
                 capture: bool = False) -> Iterator[None]:
    """Capture native durations, or hold them fixed at the original anchor."""
    restore = []
    try:
        for name, layer in model.ode_layers():
            prior = layer.__dict__.get('step_size')
            original = layer.step_size
            restore.append((layer, prior))
            if capture:
                def wrapped(x, context=None, key=name, original=original):
                    s = original(x, context)
                    values[key] = s.detach().clone()
                    return s
            else:
                def wrapped(x, context=None, key=name):
                    s = values[key]
                    if len(s) != len(x):
                        raise ValueError('anchored gate batch mismatch')
                    return s
            layer.step_size = wrapped
        yield
    finally:
        for layer, prior in restore:
            if prior is None:
                del layer.step_size
            else:
                layer.step_size = prior


def jacobian(model: VelocityNet, x: torch.Tensor,
             t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    xx = x.detach().clone().requires_grad_(True)
    pred = model(xx, t)
    jac = torch.stack([torch.autograd.grad(pred[:, j].sum(), xx,
                       retain_graph=j + 1 < pred.shape[1])[0]
                       for j in range(pred.shape[1])], dim=1)
    return pred.detach(), jac.detach()


def oracle_geometry(dist: GaussianMixture, x: np.ndarray,
                    t: float) -> dict[str, np.ndarray]:
    """Exact derivative, including responsibility derivatives for unequal diagonals."""
    if isinstance(dist,RotatedProductMixture):
        velocity,diagonal,entropy,density=dist._terms(x,t)
        order=np.argsort(diagonal,axis=1)
        return dict(oracle=velocity@dist.rotation.T,oracle_jacobian=dist.jacobian(x,t),
            eigenvalues=np.take_along_axis(diagonal,order,axis=1),
            eigenvectors=np.stack([dist.rotation[:,indices] for indices in order]),
            entropy=entropy,log_density=density)
    if isinstance(dist,RotatedGaussianMixture):
        rotation=dist.rotation
        values=oracle_geometry(dist.base,np.asarray(x)@rotation,t)
        values['oracle']=values['oracle']@rotation.T
        values['oracle_jacobian']=np.einsum('ij,bjk,lk->bil',rotation,values['oracle_jacobian'],rotation)
        values['eigenvectors']=np.einsum('ij,bjk->bik',rotation,values['eigenvectors'])
        np.testing.assert_allclose(values['oracle'],dist.velocity(x,t),rtol=1e-11,atol=1e-11)
        return values
    tt, variance, residual, weights, log_density = dist._terms(x, t)
    coefficient = (tt * dist.variances[None] - (1 - tt)) / variance
    components = dist.means[None] + coefficient * residual
    velocity = np.einsum('bk,bki->bi', weights, components)
    scores = -residual / variance
    centered_scores = scores - np.einsum('bk,bki->bi', weights, scores)[:, None]
    j = np.einsum('bk,bki,bkj->bij', weights, components, centered_scores)
    diagonal = np.einsum('bk,bki->bi', weights, coefficient)
    j[:, np.arange(dist.dim), np.arange(dist.dim)] += diagonal
    np.testing.assert_allclose(j, j.transpose(0, 2, 1), atol=2e-10, rtol=1e-10)
    eigenvalues, eigenvectors = np.linalg.eigh((j + j.transpose(0, 2, 1)) / 2)
    entropy = -np.sum(weights * np.log(np.maximum(weights, np.finfo(float).tiny)), axis=1)
    if len(dist.weights) > 1:
        entropy /= np.log(len(dist.weights))
    reference = dist.velocity(x, t)
    np.testing.assert_allclose(velocity, reference, atol=2e-11, rtol=1e-11)
    return dict(oracle=velocity, oracle_jacobian=j, eigenvalues=eigenvalues,
                eigenvectors=eigenvectors, entropy=entropy, log_density=log_density)


def diagnostic_batch(model: VelocityNet, x: torch.Tensor, t: torch.Tensor,
                     finite_difference: bool = False) -> dict[str, np.ndarray | float]:
    values: dict[str, torch.Tensor] = {}
    with gate_context(model, values, capture=True):
        pred, full = jacobian(model, x, t)
    with gate_context(model, values):
        anchored_pred, frozen = jacobian(model, x, t)
    anchor_error = float((pred - anchored_pred).abs().max())
    if anchor_error > 1e-11:
        raise AssertionError(f'anchored forward mismatch: {anchor_error}')
    result = dict(prediction=pred.cpu().numpy(), jacobian=full.cpu().numpy(),
                  anchored_jacobian=frozen.cpu().numpy(), anchor_max_error=anchor_error)
    if finite_difference:
        epsilon = 1e-4
        finite = []
        for anchored, expected in ((False, full), (True, frozen)):
            columns = []
            with torch.no_grad():
                for axis in range(x.shape[1]):
                    delta = torch.zeros_like(x)
                    delta[:, axis] = epsilon
                    if anchored:
                        with gate_context(model, values):
                            diff = (model(x + delta, t) - model(x - delta, t)) / (2 * epsilon)
                    else:
                        diff = (model(x + delta, t) - model(x - delta, t)) / (2 * epsilon)
                    columns.append(diff)
            fd = torch.stack(columns, dim=-1)
            relative = float((fd - expected).abs().max() / expected.abs().max().clamp_min(1))
            finite.append(relative)
            if relative > 1e-5:
                raise AssertionError(f'finite difference failure: {relative}')
        result['native_fd_relative_max_error'] = finite[0]
        result['anchored_fd_relative_max_error'] = finite[1]
    return result


@torch.no_grad()
def trajectory_states(model: VelocityNet, source: torch.Tensor,
                      times: tuple[float, ...], steps: int = 100) -> dict[float, torch.Tensor]:
    requested = {round(t * steps): t for t in times}
    if any(abs(index / steps - t) > 1e-10 for index, t in requested.items()):
        raise ValueError('times must lie on the Heun grid')
    states = {}
    x = source.clone()
    dt = 1 / steps
    for i in range(steps):
        t = x.new_full((len(x), 1), i * dt)
        velocity = model(x, t)
        predictor = x + dt * velocity
        x = x + dt / 2 * (velocity + model(predictor, t + dt))
        if i + 1 in requested:
            states[requested[i + 1]] = x.clone()
    return states


def panel_metrics(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    native = arrays['jacobian'] - arrays['oracle_jacobian']
    frozen = arrays['anchored_jacobian'] - arrays['oracle_jacobian']
    correction = arrays['jacobian'] - arrays['anchored_jacobian']
    error_to_fix = -frozen
    norm2 = lambda a: np.sum(a * a, axis=(-2, -1))
    n2, f2, c2 = norm2(native), norm2(frozen), norm2(correction)
    gain = f2 - n2
    np.testing.assert_allclose(gain, 2 * np.sum(error_to_fix * correction, axis=(-2, -1)) - c2,
                               atol=1e-10, rtol=1e-9)
    directions = arrays['eigenvectors']
    directional = np.einsum('bij,bjk->bik', native, directions)
    skew = (arrays['jacobian'] - arrays['jacobian'].transpose(0, 2, 1)) / 2
    return dict(velocity_mse=np.mean((arrays['prediction'] - arrays['oracle'])**2, axis=1),
                native_jacobian_error=n2, anchored_jacobian_error=f2,
                gain=gain, correction_squared_norm=c2,
                correction_alignment=np.sum(error_to_fix * correction, axis=(-2, -1)) /
                np.maximum(np.sqrt(f2 * c2), 1e-15),
                directional_error_min=np.sum(directional[:, :, 0]**2, axis=1),
                directional_error_max=np.sum(directional[:, :, -1]**2, axis=1),
                skew_squared_norm=norm2(skew))


def summarize(arrays: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict:
    if mask is None:
        mask = np.ones(len(arrays['x']), dtype=bool)
    count = int(mask.sum())
    row = dict(count=count, panel_fraction=float(mask.mean()))
    if not count:
        return row
    metrics = panel_metrics(arrays)
    row.update({key: float(value[mask].mean()) for key, value in metrics.items()})
    row['positive_gain_fraction'] = float(np.mean(metrics['gain'][mask] > 1e-10))
    row['relative_gain'] = row['gain'] / max(row['anchored_jacobian_error'], 1e-15)
    row['mean_entropy'] = float(arrays['entropy'][mask].mean())
    row['log_density_min'] = float(arrays['log_density'][mask].min())
    row['log_density_median'] = float(np.median(arrays['log_density'][mask]))
    return row


def strata(arrays: dict[str, np.ndarray], mixture: bool) -> dict:
    eigen = arrays['eigenvalues']
    rows = {'all': summarize(arrays)}
    for name, mask in [('contracting', eigen[:, -1] < -1e-8),
                       ('expanding', eigen[:, 0] > 1e-8),
                       ('mixed', (eigen[:, 0] < -1e-8) & (eigen[:, -1] > 1e-8)),
                       ('near_zero', (np.abs(eigen) <= 1e-8).any(axis=1))]:
        rows[name] = summarize(arrays, mask)
    if mixture:
        entropy = arrays['entropy']
        for name, mask in [('entropy_low', entropy < .25),
                           ('entropy_middle', (entropy >= .25) & (entropy <= .75)),
                           ('entropy_high', entropy > .75)]:
            rows[name] = summarize(arrays, mask)
    return rows


def monitor(output: Path, stop: threading.Event, progress: dict) -> None:
    started = time.monotonic()
    previous_time, previous_cpu = started, time.process_time()
    while not stop.is_set():
        current, cpu = time.monotonic(), time.process_time()
        if resource is None:
            import psutil
            rss_mb = psutil.Process().memory_info().rss / 1024**2
        else:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            rss_mb = usage.ru_maxrss / (1024**2 if sys.platform == 'darwin' else 1024)
        row = dict(time=now(), pid=os.getpid(), elapsed_seconds=current-started,
                   cpu_percent=100*(cpu-previous_cpu)/max(current-previous_time, 1e-6),
                   peak_rss_mb=rss_mb, **progress.copy())
        write_json(output / 'monitor.json', row)
        with (output / 'telemetry.jsonl').open('a') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        previous_time, previous_cpu = current, cpu
        stop.wait(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=False)
    progress = dict(state='RUNNING', completed_cells=0, current='verifying inputs')
    write_json(args.output / 'status.json', dict(time=now(), **progress))
    stop = threading.Event()
    watcher = threading.Thread(target=monitor, args=(args.output, stop, progress), daemon=True)
    watcher.start()
    started = time.monotonic()
    try:
        assert json.loads((args.study/'status.json').read_text())['state'] == 'COMPLETE'
        frozen = json.loads((args.study/'frozen-inputs.json').read_text())['source_and_data_sha256']
        manifest_path = args.data_root/'manifest.json'
        assert sha256(manifest_path) == frozen['data/synthetic/v1/manifest.json']
        for path, digest in frozen.items():
            if path.startswith('synthetic/'):
                assert sha256(ROOT/path) == digest, path
        manifest = json.loads(manifest_path.read_text())
        selected = [r for r in json.loads((args.study/'summary.json').read_text())['results']
                    if r['primary'] and r['dataset'] in DATASETS]
        assert len(selected) == 10
        n_inter, n_path = (16, 8) if args.smoke else (512, 128)
        times = (.5,) if args.smoke else TIMES
        datasets = DATASETS[:1] if args.smoke else DATASETS
        kinds = KINDS[:1] if args.smoke else KINDS
        provenance = dict(time=now(), host=platform.node(), device='cpu', dtype='float64',
                          torch=torch.__version__, threads=args.threads, smoke=args.smoke,
                          checkpoint_step=5000, training_seed=0, evaluation_seed=0,
                          times=times, interpolation_points=n_inter, trajectory_points=n_path,
                          heun_steps=100, source_sha256={str(p.relative_to(ROOT)):sha256(p)
                              for p in [Path(__file__), *sorted((ROOT/'synthetic').glob('*.py'))]},
                          protocol_sha256=sha256(ROOT/'docs/research-2026-10-21-gate-geometry/protocol.md'),
                          study_summary_sha256=sha256(args.study/'summary.json'),
                          dataset_manifest_sha256=sha256(manifest_path), inputs=[])
        results = []
        write_json(args.output/'provenance.json', provenance)
        for dataset in datasets:
            spec = manifest['datasets'][dataset]
            dist = load_distribution(spec)
            validation_path = args.data_root/dataset/'validation.npz'
            assert sha256(validation_path) == frozen[f'data/synthetic/v1/{dataset}/validation.npz']
            targets = load_targets(validation_path, 2)[:n_inter].astype(np.float64)
            noise = np.random.default_rng(0).standard_normal((n_inter, 2))
            source = torch.from_numpy(np.random.default_rng(0).standard_normal((n_path, 2)))
            input_path = args.output/f'{dataset}-inputs.npz'
            np.savez_compressed(input_path, validation_indices=np.arange(n_inter), targets=targets,
                                interpolation_source=noise, trajectory_source=source.numpy())
            for kind in kinds:
                progress['current'] = f'{dataset}/{kind}: loading'
                folder = args.study/'runs'/f'gates_20261021_{kind}'/dataset/kind/'seed000'
                cfg_path = folder/'config.json'
                cfg = json.loads(cfg_path.read_text())
                checkpoint = folder/'checkpoints/step_0005000.pt'
                reference = next(r for r in selected if r['dataset'] == dataset and r['model'] == kind)
                checkpoint_hash = sha256(checkpoint)
                assert checkpoint_hash == reference['checkpoint_sha256']
                state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                assert state['step'] == 5000 and state['identity'] == dict(dataset=dataset, model=kind, seed=0)
                assert state['config_sha256'] == sha256(cfg_path)
                assert state['dataset_manifest_sha256'] == sha256(manifest_path)
                model = VelocityNet(2, kind, **cfg['model_options']).eval()
                model.load_state_dict(state['model'], strict=True)
                with torch.no_grad():
                    cast_x = torch.from_numpy((.5*noise+.5*targets)[:16])
                    cast_t = torch.full((len(cast_x), 1), .5, dtype=torch.float32)
                    fp32 = model(cast_x.float(), cast_t).double()
                model.double()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                with torch.no_grad():
                    cast_drift = float((fp32-model(cast_x, cast_t.double())).abs().max())
                item = dict(dataset=dataset, model=kind, checkpoint=str(checkpoint),
                            checkpoint_sha256=checkpoint_hash, config_sha256=sha256(cfg_path),
                            inputs_sha256=sha256(input_path), validation_sha256=sha256(validation_path),
                            forward_float32_to_float64_max_drift=cast_drift)
                provenance['inputs'].append(item)
                write_json(args.output/'provenance.json', provenance)
                paths = trajectory_states(model, source, times)
                for panel in ('interpolation', 'trajectory'):
                    for tt in times:
                        x = torch.from_numpy((1-tt)*noise+tt*targets) if panel == 'interpolation' else paths[tt]
                        progress['current'] = f'{dataset}/{kind}/{panel}/t={tt}'
                        parts, checks = [], []
                        for left in range(0, len(x), 64):
                            xx = x[left:left+64]
                            t = torch.full((len(xx), 1), tt, dtype=torch.float64)
                            batch = diagnostic_batch(model, xx, t, finite_difference=left == 0)
                            checks.append({k:v for k, v in batch.items() if not isinstance(v, np.ndarray)})
                            parts.append({k:v for k, v in batch.items() if isinstance(v, np.ndarray)})
                        arrays = {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
                        arrays.update(x=x.numpy(), t=np.full(len(x), tt),
                                      **oracle_geometry(dist, x.numpy(), tt))
                        if kind != 'ode_original':
                            np.testing.assert_array_equal(arrays['jacobian'], arrays['anchored_jacobian'])
                        sample = args.output/f'{dataset}-{kind}-{panel}-{tt:.1f}.npz'
                        np.savez_compressed(sample, **arrays)
                        results.append(dict(dataset=dataset, model=kind, panel=panel, time=tt,
                                            checkpoint_sha256=checkpoint_hash, samples_sha256=sha256(sample),
                                            samples=sample.name, checks=checks,
                                            strata=strata(arrays, dataset == 'gmm_ring8_2d')))
                        progress['completed_cells'] = len(results)
                        write_json(args.output/'summary.json', dict(results=results))
                        write_json(args.output/'status.json', dict(time=now(), **progress))
                        print(json.dumps({k:results[-1][k] for k in ('dataset','model','panel','time')}), flush=True)
                assert sha256(checkpoint) == checkpoint_hash
        progress.update(state='COMPLETE', current='finished')
        write_json(args.output/'status.json', dict(time=now(), seconds=time.monotonic()-started, **progress))
    except BaseException as exc:
        progress.update(state='FAILED', error=repr(exc))
        write_json(args.output/'status.json', dict(time=now(), **progress))
        raise
    finally:
        stop.set()
        watcher.join()
        write_json(args.output/'monitor.json', dict(time=now(), pid=os.getpid(),
                   seconds=time.monotonic()-started, **progress))


if __name__ == '__main__':
    main()
