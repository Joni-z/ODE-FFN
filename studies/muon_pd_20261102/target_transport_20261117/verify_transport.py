"""Separate NumPy reconstruction of every retained endpoint/target statistic."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def loss_and_derivative(x, velocity, target, directions):
    generated = x@directions.T
    speed = velocity@directions.T
    indices = np.argsort(generated,axis=0,kind='stable')
    # Independent implementation: sort normally, then repair exact tie groups.
    values = np.take_along_axis(generated,indices,axis=0)
    for column in range(values.shape[1]):
        edges = np.flatnonzero(np.diff(values[:,column]) == 0)
        if not len(edges):
            continue
        groups = np.split(edges,np.flatnonzero(np.diff(edges)>1)+1)
        for group in groups:
            region = indices[group[0]:group[-1]+2,column]
            indices[group[0]:group[-1]+2,column] = region[np.argsort(speed[region,column],kind='stable')]
    displacement = np.take_along_axis(generated,indices,axis=0)-np.sort(target@directions.T,axis=0)
    derivative = np.take_along_axis(speed,indices,axis=0)
    return float(np.sum(displacement*displacement)/displacement.size),float(2*np.sum(displacement*derivative)/displacement.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    args = parser.parse_args()
    root = Path(args.input)
    binding, summary, complete = (load(root/name) for name in ('binding.json','summary.json','COMPLETE.json'))
    assert complete['passed'] and complete['summary_sha256'] == sha(root/'summary.json')
    asset = Path(binding['assets']['file'])
    assert sha(asset) == binding['assets']['sha256']
    source = np.load(asset)
    directions = source['directions'].astype(float)
    expected_banks = 1 if summary['witness'] else 8
    expected_points = 64 if summary['witness'] else 4096
    assert len(summary['banks']) == expected_banks
    all_x, all_u, all_endpoints, all_targets = [],[],[],[]
    errors = []
    all_fd = True

    def compare(data, result, target):
        x,u,endpoints = data
        loss, derivative = loss_and_derivative(x,u,target,directions)
        errors.extend((abs(loss-result['derivative']['loss']),abs(derivative-result['derivative']['loss_derivative'])))
        projected = np.sort(x@directions.T,axis=0)
        ties = int(np.sum(np.diff(projected,axis=0)==0))
        assert ties == result['derivative']['projection_ties']
        for check in result['derivative']['finite_differences']:
            epsilon = check['epsilon']
            plus, _ = loss_and_derivative(x+epsilon*u,np.zeros_like(u),target,directions)
            if ties:
                observed = (plus-loss)/epsilon
            else:
                minus, _ = loss_and_derivative(x-epsilon*u,np.zeros_like(u),target,directions)
                observed = (plus-minus)/(2*epsilon)
            errors.append(abs(observed-check['finite_difference']))
            assert bool(abs(observed-derivative)<=1e-6+1e-3*abs(derivative)) == check['qualified']
        for coefficient,endpoint,row in zip(binding['lambdas'],endpoints,result['rows'],strict=True):
            actual, _ = loss_and_derivative(endpoint,np.zeros_like(endpoint),target,directions)
            change = actual-loss
            predicted = coefficient*derivative
            signal = float(np.mean((endpoint-x)**2))
            remainder = float(np.mean((endpoint-x-coefficient*u)**2))
            expected = dict(loss=actual,sw2=np.sqrt(actual),actual_loss_change=change,
                predicted_loss_change=predicted,prediction_error=predicted-change,
                endpoint_signal_mse=signal,endpoint_remainder_mse=remainder,
                linear_predicted_loss=loss+predicted)
            if change:
                expected['target_fraction']=1-(predicted-change)**2/change**2
            if signal:
                expected['endpoint_fraction']=1-remainder/signal
            for key,value in expected.items():
                errors.append(abs(value-row[key]))

    for index,row in enumerate(summary['banks']):
        file = root/row['file']
        assert row['bank'] == index and row['points'] == expected_points and sha(file) == row['sha256']
        with np.load(file) as saved:
            arrays = {key:saved[key] for key in saved.files}
        assert all(np.isfinite(value).all() for value in arrays.values())
        x,u,endpoints = (arrays[key] for key in ('native','tangent','endpoints'))
        assert x.shape == u.shape == (expected_points,directions.shape[1])
        assert endpoints.shape == (4,*x.shape)
        assert np.array_equal(arrays['native'],arrays['augmented_native'])
        assert np.array_equal(arrays['native_path'],arrays['augmented_native_path'])
        assert np.array_equal(arrays['native_path'][0],source[f'source_{index}'][:expected_points])
        assert np.array_equal(arrays['native_path'][-1],x)
        assert np.array_equal(arrays['tangent_path'][0],np.zeros_like(u))
        assert np.array_equal(arrays['tangent_path'][-1],u)
        assert np.array_equal(arrays['paths'][:,-1],endpoints)
        target = source[f'target_{index}'][:expected_points].astype(float)
        compare((x,u,endpoints),row,target)
        fd = arrays['finite_difference']
        error = np.abs(fd-u[:64])
        qualified = bool(np.all(error<=2e-5+1e-3*np.abs(fd)))
        assert qualified == row['numerical']['endpoint_finite_difference']['qualified']
        errors.append(abs(float(error.max())-row['numerical']['endpoint_finite_difference']['max_error']))
        all_fd &= qualified
        for key,value in [('native_endpoint_rms',np.sqrt(np.mean((arrays['native32']-x)**2))),
                          ('crossed_endpoint_rms',np.sqrt(np.mean((arrays['crossed32']-endpoints[-1])**2)))]:
            errors.append(abs(float(value)-row['precision'][key]))
        all_x.append(x);all_u.append(u);all_endpoints.append(endpoints);all_targets.append(target)
    compare((np.concatenate(all_x),np.concatenate(all_u),np.concatenate(all_endpoints,axis=1)),
            summary['pooled'],np.concatenate(all_targets))
    maximum = max(errors)
    assert maximum < 1e-9
    output = dict(passed=True,case=summary['case'],split=summary['split'],witness=summary['witness'],
        banks=expected_banks,samples_per_bank=expected_points,independent_max_error=maximum,
        every_augmented_primal_exact=True,every_endpoint_fd_qualified=all_fd,
        numerically_qualified=summary['all_numerically_qualified'],
        summary_sha256=sha(root/'summary.json'),source_sha256=sha(Path(__file__)))
    dest = Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists()
    dest.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))


if __name__ == '__main__':main()
