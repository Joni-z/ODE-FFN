"""Finite, label-free clock collapse of three completed original ODE checkpoints."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

DATASETS = ('gaussian_aniso_2d', 'gmm_ring8_2d', 'rotated_product_bimodal_16d')
TITLES = ('Gaussian2D', 'Ring8', 'Product16')
LAYERS = tuple(f'blocks.{i}.ffn.ode{j}' for i in range(3) for j in (1, 2))
TIMES = (.1, .3, .5, .7, .9)
PROTOCOL_SHA = 'df6eb527a0f6bd82ab5660251b878f834e9027c38e099e654686f9c1dc221411'
RUNTIME_SHA = '47e833e60e63c5fffb820ce5939f15ee9f595ba5c5fdc6e5a906ea39fa0dbb5c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


@contextmanager
def clocks(model, *, capture=None, means=None):
    """Capture actual clocks, or install one global constant vector per layer."""
    assert (capture is None) != (means is None)
    restore = []
    try:
        for name, layer in model.ode_layers():
            assert 'step_size' not in layer.__dict__, 'Refuse nested or pre-existing clock patches'
            original = layer.step_size
            if capture is not None:
                def patched(x, context=None, name=name, original=original):
                    assert name not in capture, 'A clock was evaluated twice during calibration'
                    duration = original(x, context)
                    capture[name] = duration.detach().clone()
                    return duration
            else:
                value = means[name].detach()
                assert value.ndim == 1
                def patched(x, context=None, value=value):
                    assert x.shape[-1] == len(value)
                    return value.to(device=x.device, dtype=x.dtype).expand(*x.shape[:-1], len(value))
            layer.step_size = patched; restore.append(layer)
        yield
    finally:
        for layer in restore:del layer.step_size


def snapshot(model):
    return dict(tensors={k: v.detach().clone() for k, v in model.state_dict().items()},
                buffers={k: v.detach().clone() for k, v in model.named_buffers()},
                training={k: m.training for k, m in model.named_modules()},
                forcing={k: m.forcing_mode for k, m in model.ode_layers()})


def unchanged(model, before):
    after = snapshot(model)
    assert before['training'] == after['training'] and before['forcing'] == after['forcing']
    for group in ('tensors', 'buffers'):
        assert set(before[group]) == set(after[group])
        for name, value in before[group].items():torch.testing.assert_close(after[group][name], value, atol=0, rtol=0)
    assert all('step_size' not in layer.__dict__ for _, layer in model.ode_layers())


def risk_terms(native, collapsed, oracle, *, matrix=False):
    """Exact finite risk polarization; matrix errors use Frobenius units."""
    assert native.shape == collapsed.shape == oracle.shape and native.ndim == (3 if matrix else 2)
    assert all(np.isfinite(v).all() and v.dtype == np.float64 for v in (native, collapsed, oracle))
    average = (lambda value: float(np.mean(np.sum(value, axis=(1, 2))))) if matrix else (lambda value: float(np.mean(value)))
    residual, delta = native-oracle, collapsed-native
    old, new = average(residual**2), average((collapsed-oracle)**2)
    alignment, energy = 2*average(residual*delta), average(delta**2)
    change = new-old; remainder = change-alignment-energy
    tolerance = 64*np.finfo(np.float64).eps*max(1., old, new, abs(alignment), energy)
    assert abs(remainder) <= tolerance, (remainder, tolerance)
    return dict(native=old, collapsed=new, risk_change=change, alignment_term=alignment,
                change_energy=energy, polarization_residual=remainder, polarization_tolerance=tolerance)


def figures(output, rows):
    directory = output/'figures'; directory.mkdir(exist_ok=True)
    for polarized in (False, True):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout='constrained')
        for col, (dataset, title) in enumerate(zip(DATASETS, TITLES)):
            selected = [r for r in rows if r['dataset'] == dataset and r['time'] != 'equal_time_mean']
            for row, (prefix, metric) in enumerate((('field', 'field_mse'), ('jacobian', 'jacobian_error'))):
                ax = axes[row, col]
                lines = ((prefix+'_risk_change', 'Risk change', '#333333', '-'),
                         (prefix+'_alignment_term', 'Error alignment', '#b44843', '--'),
                         (prefix+'_change_energy', 'Finite change energy', '#326b93', ':')) if polarized else (
                         ('native_'+metric, 'Native', '#326b93', '-'), ('collapsed_'+metric, 'Collapsed clocks', '#b44843', '--'))
                for key, label, color, style in lines:
                    ax.plot([r['time'] for r in selected], [r[key] for r in selected], label=label, color=color, ls=style, marker='o')
                if polarized:ax.axhline(0, color='black', lw=.6)
                ax.set(title=title if row == 0 else '', xlabel='Time', ylabel=('Field MSE' if row == 0 else 'Mean squared Frobenius error'))
                ax.grid(alpha=.15)
        axes[0, -1].legend(fontsize=8, frameon=False)
        fig.suptitle('Frozen checkpoints; constants from the existing probe only. No retraining, sampling or fresh holdout.', fontsize=9)
        name = 'clock-collapse-polarization' if polarized else 'clock-collapse-errors'
        for extension in ('png', 'pdf'):fig.savefig(directory/f'{name}.{extension}', dpi=180)
        plt.close(fig)


def run(study, output):
    study, output = study.resolve(), output.resolve(); source = study/'source'
    assert output != study and output not in study.parents
    for protected in (source, study/'output', study/'preflight'):
        assert output != protected and protected not in output.parents
    assert not output.exists(), 'Use a new analysis directory; never overwrite an earlier diagnostic'
    input_hashes, output_hashes = {}, {}
    def read(path):
        input_hashes[str(path)] = sha(path); return json.loads(path.read_text())
    def verify():
        path = study/'source-manifest.json'; assert sha(path) == RUNTIME_SHA
        manifest = read(path); assert len(manifest['all_files']) == 595
        for name, digest in manifest['all_files'].items():assert sha(source/name) == digest, name
        return manifest
    manifest = verify()
    assert read(study/'output/affine_forcing_queue_20261025_m1/status.json')['state'] == 'COMPLETE'
    assert read(study/'preflight/summary.json')['state'] == 'PASS'
    protocol = Path(__file__).resolve().with_name('protocol-clock-collapse.md')
    if not protocol.exists():protocol = Path(__file__).resolve().parents[1]/'docs/research-architecture-search/protocol-clock-collapse.md'
    assert sha(protocol) == PROTOCOL_SHA; input_hashes[str(protocol)] = sha(protocol)
    output.mkdir(parents=True); archive = output/'source'; archive.mkdir()
    for path in (Path(__file__).resolve(), protocol):
        destination = archive/path.name; shutil.copy2(path, destination); output_hashes[str(destination)] = sha(destination)
    sys.path.insert(0, str(source)); torch.set_num_threads(1)
    from synthetic.models import VelocityNet
    from synthetic.train import restore_model, load_distribution
    from scripts.check_gate_geometry import jacobian, oracle_geometry
    for module in ('synthetic.models', 'synthetic.train', 'scripts.check_gate_geometry'):
        assert Path(sys.modules[module].__file__).resolve().is_relative_to(source), 'Imported source is not the frozen runtime'
    rows, receipts, mean_rows = [], [], []
    for dataset in DATASETS:
        suffix, dataversion = ('product16', 'product16_v1') if dataset == DATASETS[2] else ('v1', 'v1')
        current = study/'output'/f'affine_forcing_{suffix}_20261025_m1'
        assert read(current/'status.json')['state'] == 'COMPLETE'
        provenance = read(current/'provenance.json')
        config_path = current/'configs/vector_parent.json'; cfg = read(config_path)
        options = cfg['model_options']
        assert cfg['models'] == ['ode_vector'] and cfg['seeds'] == [0] and cfg['steps'] == 5000
        assert options['orders'] == 1 and options['gate_hidden'] == 32 and options['ode_forcing_mode'] == 'none'
        run_dir = current/'runs'/cfg['experiment_id']/dataset/'ode_vector/seed000'
        checkpoint = run_dir/'checkpoints/step_0005000.pt'; input_hashes[str(checkpoint)] = sha(checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['step'] == 5000 and state['identity'] == dict(dataset=dataset, model='ode_vector', seed=0)
        assert state['config_sha256'] == sha(config_path)
        data = Path('/Users/mr.z/ode-workspace-prep/data/synthetic')/dataversion
        data_manifest = read(data/'manifest.json')
        assert state['dataset_manifest_sha256'] == provenance['dataset_manifest_sha256'] == sha(data/'manifest.json')
        for filename in ('probe.npz', 'validation.npz'):
            path = data/dataset/filename; input_hashes[str(path)] = sha(path)
            assert input_hashes[str(path)] == data_manifest['files'][dataset+'/'+filename]['sha256']
        # Only these two arrays are accessed for calibration, with native dtype.
        with np.load(data/dataset/'probe.npz', allow_pickle=False) as probe:
            px, pt = probe['xt'][:256].copy(), probe['t'][:256].copy()
        dim = data_manifest['datasets'][dataset]['dim']
        assert px.shape == (256, dim) and pt.shape == (256, 1)
        assert px.dtype == pt.dtype == np.float32 and np.isfinite(px).all() and np.isfinite(pt).all()
        model = VelocityNet(dim, 'ode_vector', **options).float().eval(); restore_model(model, state)
        layers = dict(model.ode_layers()); assert tuple(layers) == LAYERS
        for layer in layers.values():
            assert (layer.orders, layer.gate_source, layer.rate_placement, layer.solve_return, layer.operator_mode,
                    layer.forcing_mode, layer.feature_mode, layer.gate_transform, layer.spectral_cap) == (
                    1, 'input', 'after_projection', 'map', 'multiplicative', 'none', 'independent', 'sigmoid', None)
        before = snapshot(model); capture = {}
        with torch.no_grad(), clocks(model, capture=capture):model(torch.from_numpy(px), torch.from_numpy(pt))
        unchanged(model, before); assert tuple(capture) == LAYERS
        means = {}
        metrics_path = run_dir/'metrics.jsonl'; input_hashes[str(metrics_path)] = sha(metrics_path)
        final = json.loads(metrics_path.read_text().splitlines()[-1]); assert final['step'] == 5000
        entries = {entry['layer']: entry for entry in final['ode_layers']}; assert set(entries) == set(LAYERS)
        maximum_log_difference = 0.
        for name, duration in capture.items():
            assert duration.shape == (256, 64) and duration.dtype == torch.float32 and torch.isfinite(duration).all()
            means[name] = duration.mean(0)
            logged = entries[name]['activation']['s_mean']; measured = float(duration.mean())
            np.testing.assert_allclose(measured, logged, atol=5e-7, rtol=1e-6)
            maximum_log_difference = max(maximum_log_difference, abs(measured-logged))
            mean_rows.append(dict(dataset=dataset, layer=name, calibration_dtype='float32', samples=256,
                                  native_all_clock_mean=measured, mean_of_channel_means=float(means[name].mean()),
                                  logged_all_clock_mean=logged, channel_mean_min=float(means[name].min()), channel_mean_max=float(means[name].max())))
        mean_path = output/'clock-means'/f'{dataset}.npz'; mean_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(mean_path, layer_names=np.array(LAYERS), means=np.stack([means[name].numpy() for name in LAYERS]),
                            durations=np.stack([capture[name].numpy() for name in LAYERS]), probe_x=px, probe_t=pt)
        output_hashes[str(mean_path)] = sha(mean_path)
        # Restore in native precision before conversion, including nonpersistent buffers.
        native = VelocityNet(dim, 'ode_vector', **options).float().eval(); restore_model(native, state); native.double()
        for parameter in native.parameters():parameter.requires_grad_(False)
        for name, value in state['runtime_buffers'].items():torch.testing.assert_close(dict(native.named_buffers())[name], value.double(), atol=0, rtol=0)
        before = snapshot(native)
        diagnostic = current/'diagnostics'/dataset/'vector_parent'; summary = read(diagnostic/'summary.json')
        assert summary['checkpoint_sha256'] == sha(checkpoint) and summary['config_sha256'] == sha(config_path)
        assert summary['manifest_sha256'] == sha(data/'manifest.json') and summary['targets_sha256'] == sha(data/dataset/'validation.npz')
        assert summary['split'] == 'validation' and set(summary['arrays_sha256']) == {f'common-{t:.1f}.npz' for t in TIMES}
        distribution = load_distribution(data_manifest['datasets'][dataset])
        max_prediction, max_derivative, selected = 0., 0., []
        for time in TIMES:
            path = diagnostic/f'common-{time:.1f}.npz'; input_hashes[str(path)] = sha(path)
            assert input_hashes[str(path)] == summary['arrays_sha256'][path.name]
            with np.load(path, allow_pickle=False) as stored:
                values = {key: stored[key] for key in ('x', 'prediction', 'jacobian', 'oracle', 'oracle_jacobian')}
            assert values['x'].shape == (512, dim) and all(v.dtype == np.float64 and np.isfinite(v).all() for v in values.values())
            truth = oracle_geometry(distribution, values['x'], time)
            for key in ('oracle', 'oracle_jacobian'):np.testing.assert_allclose(values[key], truth[key], atol=1e-10, rtol=1e-11)
            x = torch.from_numpy(values['x']); t = torch.full((512, 1), time, dtype=torch.float64)
            pred, derivative = jacobian(native, x, t); pred, derivative = pred.numpy(), derivative.numpy()
            np.testing.assert_allclose(pred, values['prediction'], atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(derivative, values['jacobian'], atol=1e-9, rtol=1e-10)
            max_prediction = max(max_prediction, float(np.abs(pred-values['prediction']).max()))
            max_derivative = max(max_derivative, float(np.abs(derivative-values['jacobian']).max()))
            with clocks(native, means=means):collapsed, collapsed_j = jacobian(native, x, t)
            collapsed, collapsed_j = collapsed.numpy(), collapsed_j.numpy(); unchanged(native, before)
            row = dict(dataset=dataset, time=time, anchors=512, dim=dim)
            for prefix, metric, terms in (
                ('field', 'field_mse', risk_terms(pred, collapsed, values['oracle'])),
                ('jacobian', 'jacobian_error', risk_terms(derivative, collapsed_j, values['oracle_jacobian'], matrix=True))):
                row['native_'+metric], row['collapsed_'+metric] = terms.pop('native'), terms.pop('collapsed')
                row.update({prefix+'_'+key: value for key, value in terms.items()})
            rows.append(row); selected.append(row)
            artifact = output/'arrays'/dataset/path.name; artifact.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(artifact, x=values['x'], time=np.array(time), prediction=pred, jacobian=derivative,
                                collapsed_prediction=collapsed, collapsed_jacobian=collapsed_j,
                                oracle=values['oracle'], oracle_jacobian=values['oracle_jacobian'])
            output_hashes[str(artifact)] = sha(artifact)
        aggregate = dict(dataset=dataset, time='equal_time_mean', anchors=2560, dim=dim)
        aggregate.update({key: float(np.mean([row[key] for row in selected])) for key in selected[0] if key not in ('dataset', 'time', 'anchors', 'dim')})
        np.testing.assert_allclose([aggregate['native_field_mse'], aggregate['native_jacobian_error']],
                                   [summary['field_mse'], summary['jacobian_error']], atol=1e-10, rtol=1e-10)
        rows.append(aggregate)
        receipts.append(dict(dataset=dataset, state='PASS', panels=5, checkpoint_sha256=sha(checkpoint), config_sha256=sha(config_path),
                             means_sha256=sha(mean_path), maximum_clock_mean_log_difference=maximum_log_difference,
                             maximum_native_prediction_replay_difference=max_prediction, maximum_native_jacobian_replay_difference=max_derivative,
                             parameters_buffers_modes_unchanged=True, runtime_buffers_restored=True))
    assert len(rows) == 18 and len(receipts) == 3 and len(mean_rows) == 18
    for filename, records in (('clock-collapse.csv', rows), ('clock-mean-summary.csv', mean_rows)):
        path = output/filename
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
        output_hashes[str(path)] = sha(path)
    figures(output, rows); verify()
    for path in sorted((output/'figures').glob('*')):output_hashes[str(path)] = sha(path)
    write(output/'provenance.json', dict(state='PASS', script_sha256=sha(__file__), protocol_sha256=PROTOCOL_SHA,
          source_manifest_sha256=RUNTIME_SHA, source_files_verified=len(manifest['all_files']), input_sha256=input_hashes, output_sha256=output_hashes,
          scope='Native float32 probe-only channel means; float64 complete-model replay and finite clock collapse. Existing exposed validation only; no oracle fitting, retraining, primary validation, sampling, timing or architecture-goal claim.'))
    result = dict(state='PASS', checkpoints=3, native_panels_replayed=15, collapsed_panels=15, risk_rows=18, clock_mean_rows=18,
                  calibration_vectors=18, calibration_dtype='float32', evaluation_dtype='float64', receipts=receipts,
                  architecture_goal_complete=False, interpretation='Frozen function dependence only; independent folded-matrix verification is still required.')
    write(output/'summary.json', result)
    return result


def self_test():
    """No checkpoint or target data: known matrix folding and exact risk units."""
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(1)
    class Layer(nn.Module):
        def __init__(self, p, gate):
            super().__init__(); self.register_buffer('p', p); self.register_buffer('gate', gate); self.forcing_mode = 'none'
        def step_size(self, x, context=None):return .2+.8*torch.sigmoid(F.linear(x, self.gate))
        def forward(self, x):return x+self.step_size(x)*F.linear(x, self.p)
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.first = Layer(torch.tensor([[.3, -.2], [.1, .4]]), torch.tensor([[.7, -.3], [.4, .8]]))
            self.second = Layer(torch.tensor([[-.2, .1], [.3, .2]]), torch.tensor([[.4, .6], [-.8, .2]]))
        def ode_layers(self):return iter((('first', self.first), ('second', self.second)))
        def forward(self, x):return self.second(F.silu(self.first(x)))
    model = Model().eval(); px = torch.linspace(-1., 1., 512).reshape(256, 2)
    baseline = model(px); capture = {}; before = snapshot(model)
    with clocks(model, capture=capture):captured = model(px)
    torch.testing.assert_close(captured, baseline, atol=0, rtol=0); unchanged(model, before)
    means = {name: value.mean(0) for name, value in capture.items()}
    assert all(value.dtype == torch.float32 and value.shape == (2,) for value in means.values())
    model.double(); before = snapshot(model)
    x = torch.tensor([[1.3, -.5], [-.8, .6]], dtype=torch.float64)
    a = torch.eye(2, dtype=x.dtype)+means['first'].double()[:, None]*model.first.p
    b = torch.eye(2, dtype=x.dtype)+means['second'].double()[:, None]*model.second.p
    folded = lambda z: F.linear(F.silu(F.linear(z, a)), b)
    with clocks(model, means=means):
        actual = model(x)
        actual_j = torch.autograd.functional.jacobian(model, x)
    unchanged(model, before)
    torch.testing.assert_close(actual, folded(x), atol=2e-16, rtol=2e-14)
    torch.testing.assert_close(actual_j, torch.autograd.functional.jacobian(folded, x), atol=2e-16, rtol=2e-14)
    assert float((actual-model(x)).norm()) > 1e-4
    try:
        with clocks(model, means=means):raise RuntimeError('contract exception')
    except RuntimeError:pass
    unchanged(model, before)
    for shape, matrix in (((3, 2), False), ((3, 2, 2), True)):
        native = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)/7
        oracle = np.zeros_like(native)
        for collapsed, sign in ((native*.5, -1), (native*1.5, 1), (native.copy(), 0)):
            values = risk_terms(native, collapsed, oracle, matrix=matrix)
            assert np.sign(values['risk_change']) == sign
            expected = float(np.mean(np.sum(native**2, axis=(1, 2)))) if matrix else float(np.mean(native**2))
            assert values['native'] == expected
            assert abs(values['polarization_residual']) <= values['polarization_tolerance']
    return dict(state='PASS', checks=['native float32 capture', 'one channel vector per layer', 'independent matrix folding/input Jacobian',
                                      'finite downstream change', 'exception-safe restoration', 'risk signs and field/Frobenius units'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path); parser.add_argument('--output', type=Path); parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert args.study is None and args.output is None; result = self_test()
    else:
        assert args.study is not None and args.output is not None; result = run(args.study, args.output)
    print(json.dumps(result, indent=2))
