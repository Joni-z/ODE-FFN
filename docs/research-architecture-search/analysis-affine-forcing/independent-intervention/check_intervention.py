"""Independent, read-only all-layer bias intervention replay after terminal training.

The analyzed implementation switches forcing_mode. This checker never does:
six forward hooks subtract the live (s(h)-c)*b from native conditional outputs.
Jacobians use vmap(jacrev(single_example)), not the analyzer's batch-sum helper.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


DATASETS = ('gaussian_aniso_2d', 'gmm_ring8_2d', 'rotated_product_bimodal_16d')
TIMES = (.1, .3, .5, .7, .9)
LAYERS = tuple(f'blocks.{i}.ffn.ode{j}' for i in range(3) for j in (1, 2))
MODEL_SHA = '9fe326f400862597ac28a392b797f51b406955fdfaf5d373eed3907e71caf337'
F_TOL, J_TOL, METRIC_TOL = 1e-10, 1e-9, 1e-12


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_tensors(expected, actual):
    assert set(expected) == set(actual)
    for name, value in expected.items():
        torch.testing.assert_close(actual[name], value, atol=0, rtol=0)


def calculate(model, x, t):
    # Each example is independent in this LayerNorm/dense backbone.
    def one(z, time):
        return model(z.unsqueeze(0), time.unsqueeze(0)).squeeze(0)
    with torch.no_grad():
        prediction = model(x, t)
    derivative = torch.vmap(torch.func.jacrev(one, argnums=0), chunk_size=64)(x, t)
    return prediction.detach().cpu().numpy(), derivative.detach().cpu().numpy()


@contextmanager
def subtract_conditional_correction(model):
    layers = list(model.ode_layers())
    assert tuple(name for name, _ in layers) == LAYERS
    counts = dict.fromkeys(LAYERS, 0)
    handles = []
    old_hook_counts = {name: len(layer._forward_hooks) for name, layer in layers}
    try:
        for name, layer in layers:
            assert layer.forcing_mode == 'conditional' and layer.orders == 1
            def subtract(module, args, output, key=name):
                assert module.forcing_mode == 'conditional'
                assert isinstance(output, torch.Tensor)
                h = args[0]
                context = args[1] if len(args) > 1 else None
                duration = module.step_size(h, context)
                center = module.shift + module.scale/2
                counts[key] += 1
                # No detach: the input-dependent correction's derivative is
                # subtracted as well. Later layers receive this changed output.
                return output - (duration-center)*module.forcing
            handles.append(layer.register_forward_hook(subtract))
        yield counts
    finally:
        for handle in handles:
            handle.remove()
        assert all(layer.forcing_mode == 'conditional' for _, layer in layers)
        assert all(len(layer._forward_hooks) == old_hook_counts[name] for name, layer in layers)


def squared_average(a, b, denominator):
    residual = np.subtract(a, b).reshape(-1)
    return float(np.dot(residual, residual)/denominator)


def panel_numbers(dataset, time, values):
    original, changed = values['prediction'], values['staticized_prediction']
    original_j, changed_j = values['jacobian'], values['staticized_jacobian']
    n = len(original)
    row = dict(dataset=dataset, time=time, anchors=n,
        original_field_mse=squared_average(original, values['oracle'], original.size),
        staticized_field_mse=squared_average(changed, values['oracle'], changed.size),
        original_jacobian_error=squared_average(original_j, values['oracle_jacobian'], n),
        staticized_jacobian_error=squared_average(changed_j, values['oracle_jacobian'], n),
        prediction_change_mse=squared_average(changed, original, original.size),
        jacobian_change_mean_squared_frobenius=squared_average(changed_j, original_j, n))
    row['field_mse_change'] = row['staticized_field_mse']-row['original_field_mse']
    row['jacobian_error_change'] = row['staticized_jacobian_error']-row['original_jacobian_error']
    return row


def check(study, analysis, destination):
    study, analysis, destination = (Path(p).resolve() for p in (study, analysis, destination))
    source = study/'source'
    assert destination == study/'independent-check'
    observed = {}
    def remember(path):
        path = Path(path)
        observed[str(path)] = digest(path)
        return path
    def read(path):
        return json.loads(remember(path).read_text())
    def archive(path):
        with np.load(remember(path), allow_pickle=False) as values:
            return {key: values[key].copy() for key in values.files}
    # This gate precedes any checkpoint, diagnostic array or outcome CSV read.
    queue = read(study/'output/affine_forcing_queue_20261025_m1/status.json')
    assert queue['state'] == 'COMPLETE' and queue['completed'] == queue['total'] == 2
    terminals = {}
    for suffix, count in (('v1', 12), ('product16', 6)):
        status = read(study/'output'/f'affine_forcing_{suffix}_20261025_m1/status.json')
        assert status['state'] == 'COMPLETE' and status['completed'] == status['total'] == count
        terminals[suffix] = status
    manifest = read(study/'source-manifest.json')
    def verify_source():
        files = {str(p.relative_to(source)) for p in source.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts}
        assert files == set(manifest['all_files']) and len(files) == 595
        for name, expected in manifest['all_files'].items():
            assert digest(source/name) == expected, name
    verify_source()
    assert digest(source/'synthetic/models.py') == MODEL_SHA
    sys.path.insert(0, str(source))
    from synthetic.models import VelocityNet, OriginalODELayer
    from synthetic.train import restore_model
    assert Path(inspect.getfile(VelocityNet)).resolve() == source/'synthetic/models.py'
    assert Path(inspect.getfile(restore_model)).resolve() == source/'synthetic/train.py'
    torch.set_num_threads(1)
    csv_path = remember(analysis/'affine-to-static.csv')
    with csv_path.open(newline='') as handle:
        reported = list(csv.DictReader(handle))
    assert len(reported) == 18
    lookup = {(r['dataset'], r['time']): r for r in reported}
    assert len(lookup) == 18
    intervention_audit = read(analysis/'affine-to-static-audit.json')
    assert intervention_audit['state'] == 'PASS' and intervention_audit['total_panels'] == 15
    audit_rows = {r['dataset']: r for r in intervention_audit['rows']}
    assert set(audit_rows) == set(DATASETS)
    cases, recomputed, checkpoints = [], [], []
    maxima = dict(native_prediction=0., native_jacobian=0., hook_prediction=0., hook_jacobian=0., csv_arithmetic=0.)

    def compare_numbers(row):
        saved = lookup[row['dataset'], str(row['time'])]
        assert int(saved['anchors']) == row['anchors']
        for key, value in row.items():
            if key in ('dataset', 'time', 'anchors'):
                continue
            difference = abs(float(saved[key])-value)
            maxima['csv_arithmetic'] = max(maxima['csv_arithmetic'], difference)
            assert math.isclose(float(saved[key]), value, abs_tol=METRIC_TOL, rel_tol=METRIC_TOL), (row['dataset'], row['time'], key, difference)

    for dataset in DATASETS:
        dim, suffix = (16, 'product16') if dataset == DATASETS[-1] else (2, 'v1')
        runroot = study/'output'/f'affine_forcing_{suffix}_20261025_m1'
        config_path = runroot/'configs/affine_forcing.json'
        config = read(config_path)
        opts = config['model_options']
        assert opts['gate_hidden'] == 32 and opts['orders'] == 1 and opts['ode_forcing_mode'] == 'conditional'
        assert opts['ode_solve_init'] == 'constant_paired'
        checkpoint = runroot/'runs'/config['experiment_id']/dataset/'ode_vector/seed000/checkpoints/step_0005000.pt'
        state = torch.load(remember(checkpoint), map_location='cpu', weights_only=False)
        assert state['step'] == 5000 and state['identity'] == dict(dataset=dataset, model='ode_vector', seed=0)
        native_dir = runroot/'diagnostics'/dataset/'affine_forcing'
        native_summary = read(native_dir/'summary.json')
        assert state['config_sha256'] == native_summary['config_sha256'] == digest(config_path)
        assert native_summary['checkpoint_sha256'] == digest(checkpoint) == audit_rows[dataset]['checkpoint_sha256']
        assert audit_rows[dataset]['config_sha256'] == digest(config_path)
        data_root = Path(config['data_root'])
        data_manifest = read(data_root/'manifest.json')
        assert state['dataset_manifest_sha256'] == native_summary['manifest_sha256'] == digest(data_root/'manifest.json')
        assert native_summary['split'] == 'validation'
        target_path = remember(data_root/dataset/'validation.npz')
        assert digest(target_path) == native_summary['targets_sha256'] == data_manifest['files'][dataset+'/validation.npz']['sha256']
        targets_file = archive(target_path)
        assert set(targets_file) == {'x1'} and targets_file['x1'].dtype == np.float32
        targets = targets_file['x1'][:512].astype(np.float64)
        assert targets.shape == (512, dim)
        gaussian = np.random.default_rng(0).standard_normal(targets.shape)

        # A fresh, cold model is restored once for each of the three final
        # checkpoints using the actual immutable trainer's buffer helper.
        model = VelocityNet(dim, 'ode_vector', **opts).double().eval()
        restore_model(model, state)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        exact_tensors({k: v.double() for k, v in state['model'].items()}, model.state_dict())
        exact_tensors({k: v.double() for k, v in state['runtime_buffers'].items()}, dict(model.named_buffers()))
        assert tuple(name for name, _ in model.ode_layers()) == LAYERS
        assert all(isinstance(layer, OriginalODELayer) and layer.forcing_mode == 'conditional' for _, layer in model.ode_layers())
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        original_buffers = {k: v.clone() for k, v in model.named_buffers()}
        rows = []
        for time in TIMES:
            native_path = native_dir/f'common-{time:.1f}.npz'
            native = archive(native_path)
            assert digest(native_path) == native_summary['arrays_sha256'][native_path.name]
            intervention_path = analysis/'affine-to-static-arrays'/dataset/native_path.name
            values = archive(intervention_path)
            assert digest(intervention_path) == audit_rows[dataset]['arrays_sha256'][str(intervention_path)]
            assert float(values['time']) == time
            for name in ('x', 'oracle', 'oracle_jacobian'):
                np.testing.assert_array_equal(values[name], native[name])
            np.testing.assert_array_equal(native['x'], (1-time)*gaussian+time*targets)
            assert all(v.dtype == np.float64 and np.isfinite(v).all() for v in values.values())
            x = torch.from_numpy(values['x']); t = torch.full((len(x), 1), time, dtype=torch.float64)
            original, original_j = calculate(model, x, t)
            with subtract_conditional_correction(model) as counts:
                counter, counter_j = calculate(model, x, t)
            assert all(n > 0 for n in counts.values())
            for actual, expected, tolerance, name in (
                (original, native['prediction'], F_TOL, 'native_prediction'),
                (original_j, native['jacobian'], J_TOL, 'native_jacobian'),
                (counter, values['staticized_prediction'], F_TOL, 'hook_prediction'),
                (counter_j, values['staticized_jacobian'], J_TOL, 'hook_jacobian')):
                np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)
                maxima[name] = max(maxima[name], float(np.max(np.abs(actual-expected))))
            np.testing.assert_allclose(values['prediction'], native['prediction'], atol=F_TOL, rtol=0)
            np.testing.assert_allclose(values['jacobian'], native['jacobian'], atol=J_TOL, rtol=0)
            row = panel_numbers(dataset, time, values)
            compare_numbers(row); rows.append(row); recomputed.append(row)
            cases.append(dict(dataset=dataset, time=time, anchors=512, hook_invocations=counts,
                checkpoint_sha256=digest(checkpoint), native_array_sha256=digest(native_path),
                intervention_array_sha256=digest(intervention_path),
                maximum_hook_prediction_difference=float(np.max(np.abs(counter-values['staticized_prediction']))),
                maximum_hook_jacobian_difference=float(np.max(np.abs(counter_j-values['staticized_jacobian'])))))
        aggregate = dict(dataset=dataset, time='equal_time_mean', anchors=2560)
        for key in rows[0]:
            if key not in ('dataset', 'time', 'anchors'):
                aggregate[key] = math.fsum(row[key] for row in rows)/5
        compare_numbers(aggregate); recomputed.append(aggregate)
        for key, target in (('original_field_mse', 'field_mse'), ('original_jacobian_error', 'jacobian_error')):
            assert math.isclose(aggregate[key], native_summary[target], abs_tol=METRIC_TOL, rel_tol=METRIC_TOL)
        exact_tensors(original_state, model.state_dict())
        exact_tensors(original_buffers, dict(model.named_buffers()))
        assert all(layer.forcing_mode == 'conditional' and not layer._forward_hooks for _, layer in model.ode_layers())
        checkpoints.append(dict(dataset=dataset, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
            config_sha256=digest(config_path), runtime_buffers_restored=True,
            parameters_buffers_and_modes_unchanged=True, cold_model=True, device='cpu', dtype='float64'))
    assert len(cases) == 15 and len(recomputed) == 18 and len(checkpoints) == 3
    verify_source()
    for path, expected in observed.items():
        assert digest(path) == expected, path
    output_csv = destination/'recomputed-panel-metrics.csv'
    with output_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(recomputed[0]))
        writer.writeheader(); writer.writerows(recomputed)
    receipt = dict(state='PASS', checked_at_utc=datetime.now(timezone.utc).isoformat(),
        checker_sha256=digest(__file__), source_manifest_sha256=digest(study/'source-manifest.json'),
        source_files_verified=595, model_source_sha256=MODEL_SHA, terminal_statuses=terminals,
        intervention='All six native conditional outputs minus live (s(h)-c)*b; downstream controllers recomputed.',
        derivative='torch.vmap(torch.func.jacrev(single_example)), chunk_size64; no analyzer Jacobian helper imported.',
        tolerances=dict(forward_absolute=F_TOL, jacobian_absolute=J_TOL, array_relative=0., metrics_absolute_relative=METRIC_TOL),
        cold_checkpoints_restored=3, panels=15, anchors=7680, csv_rows_checked=18, equal_time_aggregates_checked=3,
        native_inputs_regenerated=True, forcing_modes_never_changed=True, all_parameters_buffers_unchanged=True,
        maximum_differences=maxima, checkpoints=checkpoints, panel_checks=cases,
        recomputed_csv_sha256=digest(output_csv), input_sha256=observed,
        scope='Fixed-checkpoint validation-panel output dependence only. No retraining, primary uniform-time loss, sampling, causality or approximation lower-bound claim.')
    receipt_path = destination/'receipt.json'
    temporary = receipt_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    temporary.replace(receipt_path)
    print(json.dumps({k: receipt[k] for k in ('state', 'panels', 'csv_rows_checked', 'maximum_differences')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    check(args.study, args.analysis, args.study/'independent-check')
