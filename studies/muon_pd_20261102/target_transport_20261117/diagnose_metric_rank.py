"""Cached-only diagnosis of target finite differences across sorting boundaries.

This does not import a model, change the original gates, or create new samples.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

EPSILONS = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), f'Refusing to overwrite {path}'
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def diagnose(x: np.ndarray, u: np.ndarray, target: np.ndarray,
             directions: np.ndarray, epsilons=EPSILONS) -> dict:
    assert x.shape == u.shape == target.shape
    assert directions.ndim == 2 and directions.shape[1] == x.shape[1]
    assert all(np.isfinite(a).all() for a in (x,u,target,directions))
    projected, speed = x@directions.T, u@directions.T
    order = np.argsort(projected, axis=0, kind='stable')
    a = np.take_along_axis(projected, order, axis=0)
    b = np.take_along_axis(speed, order, axis=0)
    c = np.sort(target@directions.T, axis=0)
    gap = np.diff(a, axis=0)
    if np.any(gap == 0):
        raise ValueError('Exact projection ties require a separate one-sided diagnosis')
    residual = a-c
    loss = float(np.mean(residual**2))
    derivative = float(2*np.mean(residual*b))
    curvature = float(np.mean(b*b))
    relative_speed = np.abs(np.diff(b, axis=0))
    moving = relative_speed > 0
    radius = float(np.min(gap[moving]/relative_speed[moving])) if moving.any() else None

    # Independent reverse-mode differentiation of Torch's actual sorted metric.
    xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    theta = torch.tensor(directions, dtype=torch.float64)
    sorted_target = torch.sort(torch.tensor(target)@theta.T, dim=0).values
    torch_loss = (torch.sort(xt@theta.T, dim=0).values-sorted_target).square().mean()
    gradient, = torch.autograd.grad(torch_loss, xt)
    autograd = float((gradient*torch.tensor(u)).sum())
    autograd_error = abs(autograd-derivative)
    assert autograd_error <= 1e-11*(1+abs(derivative))
    assert abs(float(torch_loss.detach())-loss) <= 1e-12*(1+loss)

    rows = []
    for epsilon in epsilons:
        corrections, losses, ideal_losses, fixed_losses = [],[],[],[]
        changed, quadratic_errors = [],[]
        for sign in (1,-1):
            step = sign*epsilon
            fixed = a+step*b
            sorted_ideal = np.sort(fixed, axis=0)
            fixed_residual, sorted_residual = fixed-c, sorted_ideal-c
            # Stable exact sorting correction: difference of squared residuals.
            correction = float(np.mean((sorted_residual-fixed_residual)*
                                       (sorted_residual+fixed_residual)))
            corrections.append(correction)
            assert correction <= 1e-12*(1+loss)
            ideal_losses.append(float(np.mean(sorted_residual**2)))
            polynomial = loss+step*derivative+step*step*curvature
            fixed_loss = float(np.mean(fixed_residual**2))
            fixed_losses.append(fixed_loss)
            quadratic_error = abs(fixed_loss-polynomial)
            assert quadratic_error <= 1e-12*(1+loss)
            quadratic_errors.append(quadratic_error)
            actual_projection = (x+step*u)@directions.T
            actual_order = np.argsort(actual_projection,axis=0,kind='stable')
            changed.append(int(np.count_nonzero(actual_order != order)))
            losses.append(float(np.mean((np.sort(actual_projection,axis=0)-c)**2)))
        observed = (losses[0]-losses[1])/(2*epsilon)
        ideal_fd = (ideal_losses[0]-ideal_losses[1])/(2*epsilon)
        predicted_error = (corrections[0]-corrections[1])/(2*epsilon)
        projection_rounding = observed-ideal_fd
        fixed_fd_rounding = (fixed_losses[0]-fixed_losses[1])/(2*epsilon)-derivative
        correction_subtraction_rounding = (
            ((ideal_losses[0]-fixed_losses[0])-(ideal_losses[1]-fixed_losses[1]))/
            (2*epsilon)-predicted_error)
        reconstruction_error = abs(observed-derivative-predicted_error-projection_rounding-
                                   fixed_fd_rounding-correction_subtraction_rounding)
        assert reconstruction_error <= 1e-9*(1+abs(derivative))
        rows.append(dict(epsilon=epsilon,finite_difference=observed,
            original_tolerance=1e-6+1e-3*abs(derivative),
            original_gate_at_this_epsilon=bool(abs(observed-derivative)<=1e-6+1e-3*abs(derivative)),
            absolute_derivative_error=abs(observed-derivative),
            changed_rank_slots_plus=changed[0],changed_rank_slots_minus=changed[1],
            sorting_correction_plus=corrections[0],sorting_correction_minus=corrections[1],
            sorting_predicted_fd_error=predicted_error,
            projection_rounding_fd_effect=projection_rounding,
            fixed_order_fd_rounding_effect=fixed_fd_rounding,
            correction_subtraction_rounding_effect=correction_subtraction_rounding,
            unadjusted_correction_identity_error=abs(observed-derivative-predicted_error),
            correction_identity_error=reconstruction_error,
            fixed_quadratic_max_error=max(quadratic_errors)))
    return dict(diagnostic_checks_passed=True,points=len(x),projections=len(directions),
        baseline_ties=0,loss=loss,derivative=derivative,autograd_derivative=autograd,
        autograd_error=autograd_error,quadratic_coefficient=curvature,
        nearest_symmetric_rank_crossing_radius=radius,scan=rows)


def controls() -> dict:
    theta = np.ones((1,1))
    x = np.array([[-1.],[0.],[2.]])
    u = np.array([[.2],[-.1],[.3]])
    target = np.array([[-.8],[.1],[1.8]])
    fixed = diagnose(x,u,target,theta)
    assert all(r['changed_rank_slots_plus']==r['changed_rank_slots_minus']==0
               for r in fixed['scan'])
    opposite = diagnose(x,-u,target,theta)
    assert abs(fixed['derivative']+opposite['derivative']) < 1e-14
    permuted = diagnose(x[[2,0,1]],u[[2,0,1]],target[[1,2,0]],theta)
    assert abs(fixed['derivative']-permuted['derivative']) < 1e-14
    crossing = diagnose(np.array([[-.1],[.1]]),np.array([[1.],[-.5]]),
                        np.array([[-1.],[1.]]),theta,(.2,.1,.01))
    assert crossing['scan'][0]['changed_rank_slots_plus'] == 2
    assert crossing['scan'][0]['changed_rank_slots_minus'] == 0
    assert crossing['scan'][0]['absolute_derivative_error'] > .1
    assert crossing['scan'][0]['correction_identity_error'] < 1e-14
    try:
        diagnose(np.zeros((2,1)),np.ones((2,1)),np.zeros((2,1)),theta)
    except ValueError:
        ties_rejected = True
    else:
        raise AssertionError('Tied-projection negative control was not rejected')
    return dict(passed=True,scope='deterministic controls only',fixed_order=fixed,
        crossing=crossing,sign_reversal=True,permutation_invariant=True,
        exact_ties_rejected=ties_rejected)


def original_check(diagnostic: dict, original: dict) -> None:
    assert abs(diagnostic['loss']-original['loss']) < 1e-12
    assert abs(diagnostic['derivative']-original['loss_derivative']) < 1e-12
    for stored in original['finite_differences']:
        row = next(r for r in diagnostic['scan'] if r['epsilon']==stored['epsilon'])
        assert abs(row['finite_difference']-stored['finite_difference']) < 1e-12
        assert row['original_gate_at_this_epsilon'] == stored['qualified']


def inspect_case(root: Path, bank: int | None) -> list[dict]:
    binding = read(root/'binding.json')
    asset = Path(binding['assets']['file'])
    assert sha(asset) == binding['assets']['sha256']
    rows = read(root/'banks.json')
    if bank is None:
        complete, summary = read(root/'COMPLETE.json'), read(root/'summary.json')
        assert complete['passed'] and complete['summary_sha256'] == sha(root/'summary.json')
        assert not binding['witness'] and len(rows) == 8
        assert rows == summary['banks']
    else:
        assert rows[bank]['bank'] == bank
    results, all_x, all_u, all_target = [],[],[],[]
    with np.load(asset) as source:
        directions = source['directions'].astype(float)
        for index in (range(len(rows)) if bank is None else [bank]):
            row = rows[index]
            file = root/row['file']
            before = sha(file)
            assert before == row['sha256']
            with np.load(file) as saved:
                x,u = saved['native'],saved['tangent']
            target = source[f'target_{index}'][:len(x)].astype(float)
            result = diagnose(x,u,target,directions)
            original_check(result,row['derivative'])
            assert sha(file) == before
            results.append(dict(case=binding['case']['id'],split=binding['split'],bank=index,
                original_numerically_qualified=row['numerical']['qualified'],
                original_target_qualified=row['derivative']['qualified'],
                original_target_checks=row['derivative']['finite_differences'],
                input_sha256=before,asset_sha256=binding['assets']['sha256'],**result))
            all_x.append(x);all_u.append(u);all_target.append(target)
    if bank is None:
        pooled = diagnose(np.concatenate(all_x),np.concatenate(all_u),
                          np.concatenate(all_target),directions)
        original_check(pooled,summary['pooled']['derivative'])
        results.append(dict(case=binding['case']['id'],split=binding['split'],bank='pooled',
            original_target_qualified=summary['pooled']['derivative']['qualified'],
            original_target_checks=summary['pooled']['derivative']['finite_differences'],
            asset_sha256=binding['assets']['sha256'],**pooled))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--controls',action='store_true')
    group.add_argument('--input',type=Path)
    group.add_argument('--formal-root',type=Path)
    parser.add_argument('--bank',type=int)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.controls:
        assert args.bank is None
        result = controls()
    else:
        if args.input:
            assert args.bank is not None and 0 <= args.bank < 8
            rows = inspect_case(args.input,args.bank)
            scope = 'post-observation single-bank debugging subset'
        else:
            assert args.bank is None
            cases = sorted(args.formal_root.glob('*/*/binding.json'))
            assert len(cases) == 8
            rows = []
            for file in cases:
                rows.extend(inspect_case(file.parent,None))
                print('METRIC_RANK_CASE',file.parent,flush=True)
            assert len(rows) == 72
            scope = 'complete 64 banks and 8 pooled cases; original gates unchanged'
        result = dict(passed=True,scope=scope,rows=rows,new_model_calls=0,
            original_qualification_changed=False)
    result['source_sha256'] = sha(Path(__file__))
    dump(args.output,result)
    print('METRIC_RANK_DIAGNOSIS_PASS',args.output,flush=True)


if __name__ == '__main__':
    main()
