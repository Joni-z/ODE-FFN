"""Independent dense-array reconstruction of cached endpoint geometry."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    saved = json.loads(args.input.read_text())
    assert saved['passed'] and saved['original_gates_unchanged']
    errors, units = [], []

    def check(a, b):
        error = abs(float(a)-float(b))
        assert error < 1e-11*(1+abs(float(b))), (a, b, error)
        errors.append(error)

    def verify(values, wanted):
        x, final, target, theta = (values[k] for k in
            ('native', 'crossed_reference', 'target', 'directions'))
        a, b, c = x@theta.T, final@theta.T, np.sort(target@theta.T, axis=0)
        order = np.argsort(a, axis=0, kind='stable')
        sorted_a = np.take_along_axis(a, order, axis=0)
        assert np.all(np.diff(sorted_a, axis=0) > 0)
        residual = sorted_a-c
        inverse = np.argsort(order, axis=0)
        gradient = np.take_along_axis(2*residual/(len(x)*len(theta)), inverse, axis=0)@theta
        initial_loss = np.mean(residual**2)
        final_loss = np.mean((np.sort(b, axis=0)-c)**2)
        linear = 2*np.mean(residual*np.take_along_axis(b-a, order, axis=0))
        quadratic = np.mean((b-a)**2)
        fixed = np.mean((np.take_along_axis(b, order, axis=0)-c)**2)
        sorting = final_loss-fixed
        for key, value in [('native_loss', initial_loss), ('crossed_loss', final_loss),
                           ('linear', linear), ('quadratic', quadratic), ('sorting', sorting),
                           ('actual_delta_loss', final_loss-initial_loss)]:
            check(wanted[key], value)
        for name in ('analytic', 'measured'):
            tangent = values[name+'_tangent']
            row = wanted['predictions'][name]
            derivative = 2*np.mean(residual*np.take_along_axis(tangent@theta.T, order, axis=0))
            error = tangent-(final-x)
            energy = np.sum(error**2, axis=1)
            contributions = np.sum(gradient*error, axis=1)
            gradient_energy = np.sum(gradient**2, axis=1)
            ranked = np.argsort(-energy, kind='stable')
            for key, value in [('predicted_delta_loss', derivative),
                    ('forecast_error', derivative-final_loss+initial_loss),
                    ('gradient_weighted_residual', contributions.sum()),
                    ('nonlinear_remainder', quadratic+sorting),
                    ('endpoint_mean_squared_error', np.mean(error**2))]:
                check(row[key], value)
            for tail in row['tails']:
                indices = ranked[:tail['count']]
                assert indices.tolist() == tail['source_indices']
                for key, value in [
                        ('residual_energy_share', energy[indices].sum()/energy.sum()),
                        ('gradient_energy_share', gradient_energy[indices].sum()/gradient_energy.sum()),
                        ('absolute_gradient_contribution_share',
                         np.abs(contributions[indices]).sum()/np.abs(contributions).sum()),
                        ('signed_gradient_contribution', contributions[indices].sum())]:
                    check(tail[key], value)
        units.append(len(x))

    for group in saved['groups']:
        assert len(group['bank_bindings']) == len(group['banks']) == 8
        arrays = []
        for binding, bank in zip(group['bank_bindings'], group['banks'], strict=True):
            folder = Path(binding['folder'])
            complete = json.loads((folder/'COMPLETE.json').read_text())
            assert sha(folder/'COMPLETE.json') == binding['complete_sha256']
            assert sha(folder/complete['file']) == binding['array_sha256']
            assert sha(folder/'independent.json') == binding['independent_sha256']
            assert complete['numerically_qualified'] == bank['original_numerically_qualified']
            with np.load(folder/complete['file']) as z:
                values = {k: z[k].copy() for k in ('native', 'crossed_reference', 'target',
                    'directions', 'analytic_tangent', 'measured_tangent')}
            verify(values, bank['result'])
            arrays.append(values)
        pooled = {k: np.concatenate([v[k] for v in arrays]) for k in arrays[0] if k != 'directions'}
        pooled['directions'] = arrays[0]['directions']
        verify(pooled, group['pooled'])
        print('INDEPENDENT_GEOMETRY_GROUP', group['split'], flush=True)
    assert len(units) == 18 and units.count(32768) == 2 and units.count(4096) == 16
    result = dict(passed=True, units=len(units), scalar_comparisons=len(errors),
        max_error=max(errors), input_sha256=sha(args.input), source_sha256=sha(__file__),
        method='dense projections, inverse-rank gradient, direct sorted/fixed loss subtraction',
        original_flags_unchanged=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('INDEPENDENT_GEOMETRY_PASS', len(units), max(errors), flush=True)


if __name__ == '__main__':
    main()
