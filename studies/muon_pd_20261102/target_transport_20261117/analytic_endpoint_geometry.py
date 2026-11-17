"""Cached actual-endpoint loss geometry; no model imports or new samples."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from verify_analytic_target import reconstruct, sha


def close(value, reference, tolerance=1e-11):
    error = abs(value-reference)
    assert error <= tolerance*(1+abs(reference)), (value, reference, error)
    return float(error)


def geometry(x, crossed, target, directions):
    assert x.shape == crossed.shape == target.shape
    assert directions.ndim == 2 and directions.shape[1] == x.shape[1]
    assert all(a.dtype == np.float64 and np.isfinite(a).all()
               for a in (x, crossed, target, directions))
    gradient = np.zeros_like(x)
    rows = []
    displacement = crossed-x
    for index, direction in enumerate(directions):
        initial, final = x@direction, crossed@direction
        order = np.argsort(initial, kind='stable')
        if np.any(np.diff(initial[order]) == 0):
            raise ValueError('Exact projection ties: a common gradient is not established')
        wanted = np.sort(target@direction)
        residual = initial[order]-wanted
        projected_displacement = displacement@direction
        fixed_residual = final[order]-wanted
        sorted_residual = np.sort(final)-wanted
        # Scatter the one-dimensional loss gradient to original source indices.
        gradient[order] += (2*residual/(len(x)*len(directions)))[:, None]*direction
        native_loss = float(np.mean(residual**2))
        crossed_loss = float(np.mean(sorted_residual**2))
        linear = float(2*np.mean(residual*projected_displacement[order]))
        quadratic = float(np.mean(projected_displacement**2))
        sorting = float(np.mean((sorted_residual-fixed_residual)*
                                (sorted_residual+fixed_residual)))
        assert sorting <= 1e-12*(1+native_loss)
        error = close(crossed_loss-native_loss, linear+quadratic+sorting)
        rows.append(dict(direction=index, native_loss=native_loss,
                         crossed_loss=crossed_loss, linear=linear,
                         quadratic=quadratic, sorting=sorting,
                         changed_rank_slots=int(np.count_nonzero(
                             np.argsort(final, kind='stable') != order)),
                         identity_error=error))
    result = {key: float(np.mean([r[key] for r in rows]))
              for key in ('native_loss', 'crossed_loss', 'linear', 'quadratic', 'sorting')}
    result['actual_delta_loss'] = result['crossed_loss']-result['native_loss']
    result['nonlinear_remainder'] = result['quadratic']+result['sorting']
    result['identity_error'] = close(result['actual_delta_loss'],
                                     result['linear']+result['nonlinear_remainder'])
    result['gradient_projection_error'] = close(float(np.sum(gradient*displacement)),
                                                result['linear'])
    result['directions'] = rows
    return gradient, result


def independent_ad(x, target, directions, gradient):
    import torch
    torch.set_num_threads(1)
    point = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    theta = torch.tensor(directions, dtype=torch.float64)
    wanted = torch.sort(torch.tensor(target)@theta.T, dim=0).values
    loss = (torch.sort(point@theta.T, dim=0).values-wanted).square().mean()
    derivative, = torch.autograd.grad(loss, point)
    error = float(np.max(np.abs(derivative.numpy()-gradient)))
    assert error <= 1e-12*(1+float(np.max(np.abs(gradient))))
    return dict(loss=float(loss.detach()), gradient_max_error=error)


def tangent_accounting(x, crossed, tangent, gradient, target_geometry):
    displacement = crossed-x
    residual = tangent-displacement
    energy = np.sum(residual**2, axis=1)
    gradient_energy = np.sum(gradient**2, axis=1)
    contribution = np.sum(gradient*residual, axis=1)
    absolute = np.abs(contribution)
    prediction = float(np.sum(gradient*tangent))
    weighted_residual = float(np.sum(contribution))
    forecast_error = prediction-target_geometry['actual_delta_loss']
    identity_error = close(forecast_error,
                           weighted_residual-target_geometry['nonlinear_remainder'])
    order = np.lexsort((np.arange(len(x)), -energy))

    def fraction(part, total):
        return None if total == 0 else float(part/total)

    tails = []
    for label, count in [('top1', 1), ('top10', min(10, len(x))),
                         ('top1percent', max(1, math.ceil(.01*len(x))))]:
        selected = order[:count]
        tails.append(dict(selection=label, count=count, source_indices=selected.tolist(),
            residual_energy_share=fraction(energy[selected].sum(), energy.sum()),
            gradient_energy_share=fraction(gradient_energy[selected].sum(), gradient_energy.sum()),
            absolute_gradient_contribution_share=fraction(absolute[selected].sum(), absolute.sum()),
            signed_gradient_contribution=float(contribution[selected].sum())))
    signal = float(np.mean(displacement**2))
    squared_error = float(np.mean(residual**2))
    return dict(predicted_delta_loss=prediction, forecast_error=forecast_error,
        gradient_weighted_residual=weighted_residual,
        nonlinear_remainder=target_geometry['nonlinear_remainder'],
        accounting_error=identity_error, endpoint_mean_squared_error=squared_error,
        endpoint_score=None if signal == 0 else 1-squared_error/signal,
        absolute_gradient_contributions=float(absolute.sum()),
        signed_absolute_cancellation_ratio=fraction(abs(weighted_residual), absolute.sum()),
        tails=tails)


def inspect(values, originals=None):
    x, crossed, target, directions = (values[k] for k in
        ('native', 'crossed_reference', 'target', 'directions'))
    gradient, result = geometry(x, crossed, target, directions)
    result['independent_ad'] = independent_ad(x, target, directions, gradient)
    close(result['independent_ad']['loss'], result['native_loss'])
    old_loss, _ = reconstruct(values['native_reference'], target, directions)
    result['native_reference_loss'] = old_loss
    result['native_replay_loss_change'] = result['native_loss']-old_loss
    result['native_replay_max_error'] = float(np.max(np.abs(x-values['native_reference'])))
    result['predictions'] = {}
    for name in ('analytic', 'measured'):
        tangent = values[name+'_tangent']
        row = tangent_accounting(x, crossed, tangent, gradient, result)
        loss, derivative = reconstruct(x, target, directions, tangent)
        close(loss, result['native_loss'])
        row['independent_derivative_error'] = close(row['predicted_delta_loss'], derivative)
        if originals is not None:
            row['stored_derivative_error'] = close(derivative,
                originals['predictions'][name]['derivative']['loss_derivative'])
        result['predictions'][name] = row
    result['analytic_minus_measured'] = float(np.sum(gradient*(
        values['analytic_tangent']-values['measured_tangent'])))
    close(result['analytic_minus_measured'],
          result['predictions']['analytic']['predicted_delta_loss']-
          result['predictions']['measured']['predicted_delta_loss'])
    return result


def controls():
    theta = np.ones((1, 1), dtype=np.float64)
    x = np.array([[-1.], [1.]])
    target = np.array([[-2.], [2.]])
    gradient, permutation = geometry(x, x[::-1].copy(), target, theta)
    for key, expected in [('actual_delta_loss', 0.), ('linear', 4.),
                          ('quadratic', 4.), ('sorting', -8.)]:
        close(permutation[key], expected, 1e-14)
    independent_ad(x, target, theta, gradient)
    ordinary = x+np.array([[.1], [-.2]])
    _, fixed = geometry(x, ordinary, target, theta)
    close(fixed['sorting'], 0., 1e-14)
    rng = np.random.default_rng(20261117)
    points, wanted = rng.normal(size=(17, 3)), rng.normal(size=(17, 3))
    directions = rng.normal(size=(7, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    grad, many = geometry(points, points+.2*rng.normal(size=points.shape), wanted, directions)
    ad = independent_ad(points, wanted, directions, grad)
    tangent = rng.normal(size=points.shape)
    account = tangent_accounting(points, points+.1*tangent, tangent, grad,
        geometry(points, points+.1*tangent, wanted, directions)[1])
    _, stored = reconstruct(points, wanted, directions, tangent)
    close(float(np.sum(grad*tangent)), stored)
    try:
        geometry(np.zeros((2, 1)), x, target, theta)
    except ValueError:
        ties_rejected = True
    else:
        raise AssertionError('Tied-projection negative control was accepted')
    return dict(passed=True, permutation=permutation, fixed_assignment=fixed,
                multidirectional_identity=many['identity_error'],
                independent_ad=ad, tangent_accounting_error=account['accounting_error'],
                exact_ties_rejected=ties_rejected)


def audit(root, case, split, steps):
    keys = ('native', 'native_reference', 'crossed_reference', 'target',
            'analytic_tangent', 'measured_tangent')
    groups = {k: [] for k in keys}
    directions = None
    banks, bindings, metadata = [], [], []
    for bank in range(8):
        folder = root/f'{case}-{split}-bank{bank}-heun{steps}'
        done = json.loads((folder/'COMPLETE.json').read_text())
        independent = json.loads((folder/'independent.json').read_text())
        assert done['passed'] and not done['witness'] and done['points'] == 4096
        assert (done['case'], done['split'], done['bank'], done['steps']) == (case, split, bank, steps)
        assert done['hidden_reference_steps'] == 256 and done['variant'] == 'power-reuse'
        assert independent['passed'] and independent['complete_sha256'] == sha(folder/'COMPLETE.json')
        file = folder/done['file']
        assert sha(file) == done['sha256'] == independent['array_sha256']
        with np.load(file) as arrays:
            values = {k: arrays[k].copy() for k in (*keys, 'directions')}
        if directions is None:
            directions = values['directions']
        else:
            assert np.array_equal(directions, values['directions'])
        assert np.all(np.abs(values['native']-values['native_reference']) <=
                      1e-10+1e-9*np.abs(values['native_reference']))
        result = inspect(values, done)
        close(result['native_reference_loss'], done['native_loss'])
        close(result['crossed_loss'], done['crossed_loss'])
        banks.append(dict(bank=bank, points=4096,
                          original_numerically_qualified=done['numerically_qualified'], result=result))
        bindings.append(dict(folder=str(folder), array_sha256=sha(file),
            complete_sha256=sha(folder/'COMPLETE.json'),
            independent_sha256=sha(folder/'independent.json')))
        assert sha(file) == done['sha256']
        for k in keys:
            groups[k].append(values[k])
        metadata.append(done)
        print('GEOMETRY_BANK', case, split, steps, bank, flush=True)
    for key in ('source_sha256', 'kernel_sha256', 'execution_proof_sha256', 'portable_sha256'):
        assert len({m[key] for m in metadata}) == 1, key
    pooled = {k: np.concatenate(v) for k, v in groups.items()}
    pooled['directions'] = directions
    return dict(case=case, split=split, steps=steps, banks=banks, pooled=inspect(pooled),
                points=32768, bank_bindings=bindings,
                original_prefix_checks_qualified=sum(m['numerically_qualified'] for m in metadata),
                full_source_reference_status='separate required follow-up; not inferred here')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path)
    parser.add_argument('--case', default='product_rk4_trained')
    parser.add_argument('--steps', type=int, choices=(128, 2048), default=128)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists()
    if args.check:
        result = controls()
    else:
        assert args.root is not None
        result = dict(passed=True, scope='retrospective complete-group cached accounting',
            training_runs=0, new_model_evaluations=0, new_samples=0, fitted_scales=0,
            original_gates_unchanged=True,
            groups=[audit(args.root, args.case, split, args.steps)
                    for split in ('selection', 'confirmation')])
    result['source_sha256'] = sha(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('GEOMETRY_PASS', args.output, flush=True)


if __name__ == '__main__':
    main()
