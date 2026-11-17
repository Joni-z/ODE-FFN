"""Projection-wise NumPy reconstruction, without Torch or model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reconstruct(x, target, directions, tangent=None):
    losses, derivatives = [], []
    for direction in directions:
        projected, y = x@direction, np.sort(target@direction)
        if tangent is None:
            residual = np.sort(projected)-y
        else:
            u = tangent@direction
            order = np.lexsort((u, projected))
            residual = projected[order]-y
            derivatives.append(float(2*np.mean(residual*u[order])))
        losses.append(float(np.mean(residual**2)))
    return float(np.mean(losses)), (float(np.mean(derivatives)) if tangent is not None else None)


def main():
    p = argparse.ArgumentParser()
    for name in ('input', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    done = json.loads((a.input/'COMPLETE.json').read_text())
    assert done['passed'] and done['tensors_methods_rng_unchanged']
    file = a.input/done['file']
    assert sha(file) == done['sha256']
    with np.load(file) as data:
        values = {k: data[k].copy() for k in data.files}
    assert all(v.dtype == np.float64 and np.isfinite(v).all() for v in values.values())
    assert len(values['native']) == done['points']
    assert done['probe_calls'] == [2*(k*(done['steps']//8))+stage for k in range(8) for stage in (0,1)]
    assert len(values['probe_x']) == 16*min(8, done['points'])
    errors = []
    def close(value, reference):
        error = abs(value-reference)
        assert error <= 1e-10*(1+abs(reference)), (value, reference, error)
        errors.append(error)
    x, target, directions = (values[k] for k in ('native', 'target', 'directions'))
    loss0, _ = reconstruct(values['native_reference'], target, directions)
    loss1, _ = reconstruct(values['crossed_reference'], target, directions)
    close(loss0, done['native_loss']); close(loss1, done['crossed_loss'])
    delta = loss1-loss0
    for name in ('analytic', 'measured'):
        tangent = values[name+'_tangent']
        loss, derivative = reconstruct(x, target, directions, tangent)
        row = done['predictions'][name]
        close(loss, row['derivative']['loss'])
        close(derivative, row['derivative']['loss_derivative'])
        ties = sum(int(np.count_nonzero(np.diff(np.sort(x@direction)) == 0)) for direction in directions)
        assert ties == row['derivative']['projection_ties']
        for finite in row['derivative']['finite_differences']:
            epsilon = finite['epsilon']
            plus, _ = reconstruct(x+epsilon*tangent, target, directions)
            minus, _ = reconstruct(x-epsilon*tangent, target, directions)
            observed = (plus-loss)/epsilon if ties else (plus-minus)/(2*epsilon)
            close(observed, finite['finite_difference'])
            error = abs(observed-derivative)
            close(error, finite['absolute_error'])
            assert bool(error <= 1e-6+1e-3*abs(derivative)) == finite['qualified']
        close(delta, row['actual_loss_change'])
        if delta:
            close(1-(derivative-delta)**2/delta**2, row['score'])
        else:
            assert row['score'] is None
        forecast, _ = reconstruct(x+tangent, target, directions)
        close(forecast-loss0, row['endpoint_metric_change'])
        assert bool(np.sign(derivative) == np.sign(delta)) == row['sign_agrees']
    replay = np.abs(x-values['native_reference'])
    assert np.all(replay <= 1e-10+1e-9*np.abs(values['native_reference']))
    close(float(replay.max()), done['native_replay_max_error'])
    for name, left, right, atol, rtol in [
        ('observer_primal_refinement','probe_primal128','probe_primal256',1e-7,1e-7),
        ('observer_tangent_refinement','probe_tangent128','probe_tangent256',1e-8,1e-5),
        ('observer_probe_replay','probe_recorded','probe_replayed',1e-10,1e-8)]:
        reference = values[right]
        error = np.abs(values[left]-reference)
        assert bool(np.all(error <= atol+rtol*np.abs(reference))) == done['numerical'][name]['qualified']
        close(float(error.max()), done['numerical'][name]['max_error'])
    result = dict(passed=True, case=done['case'], split=done['split'], bank=done['bank'],
                  steps=done['steps'], points=done['points'], witness=done['witness'],
                  numerically_qualified=done['numerically_qualified'],
                  max_scalar_error=max(errors), complete_sha256=sha(a.input/'COMPLETE.json'),
                  array_sha256=sha(file), source_sha256=sha(Path(__file__)))
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print('INDEPENDENT_ANALYTIC_TARGET_PASS', done['case'], done['split'], done['bank'],
          done['steps'], max(errors), flush=True)


if __name__ == '__main__':
    main()
