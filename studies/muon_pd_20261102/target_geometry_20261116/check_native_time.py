"""All-checkpoint native recording/cache and packed-blend equivalence checks."""
import argparse
from pathlib import Path
import time
import numpy as np
from geometry import read, dump, sha
from frozen_probe import restore, tensor_hashes, assets, torch, base
from trajectory import OracleResponse, BlendedField, integrate, LAMBDAS
from native_time import cached_level, packed_blends
from check_trajectory import GaussianOracle, GaussianModel


def main() -> None:
    parser = argparse.ArgumentParser()
    for key in ['study', 'pilot', 'data-root', 'output']:
        parser.add_argument('--'+key, required=True)
    args = parser.parse_args()
    study, pilot, data_root, out = map(Path, [args.study, args.pilot, args.data_root, args.output])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    source = torch.linspace(-1, 1, 16, dtype=torch.double).reshape(8, 2)
    gaussian = GaussianOracle()
    controls = []
    for steps in [1000, 1032]:
        endpoint, path = integrate(gaussian, source, steps)
        direct = base.train.generated_samples(gaussian, source, steps)
        assert torch.equal(endpoint, direct)
        error = float((endpoint-source).abs().max())
        assert error < 1e-6
        controls.append(dict(steps=steps, native_endpoint_exact=True, gaussian_endpoint_error=error,
                             recorded_states=len(path)))
    analytic = GaussianModel(gaussian, .3*np.eye(2), np.zeros(2))
    packed, _ = packed_blends(analytic, gaussian, source, 1000)
    expected = source.numpy()[None]*np.exp(.3*np.asarray(LAMBDAS))[:, None, None]
    exact_packed_error = float(np.max(np.abs(packed-expected)))
    assert exact_packed_error < 1e-6
    anchors = read(study/'anchors.json')
    rows = []
    for index, row in enumerate(anchors['references']+anchors['new_cases']):
        model, law, cell, evaluation, identity = restore(study, anchors, row, data_root)
        model.double()
        snapshot = tensor_hashes(model)
        arrays, _, binding = assets(anchors['assets'], cell['dataset'], 'selection', 'cpu')
        cached, original_binding = cached_level(pilot, row, identity, arrays, 128, 32)
        tensor = torch.from_numpy(cached['source'])
        endpoint, path = integrate(model, tensor, 128)
        native_error = float(np.max(np.abs(endpoint.numpy()-cached['endpoint'])))
        native_path_error = float(np.max(np.abs(path.numpy()-cached['path'])))
        assert max(native_error, native_path_error) < 1e-10
        oracle = OracleResponse(law)
        endpoint, path = packed_blends(model, oracle, tensor, 128)
        with np.load(original_binding['file']) as z:
            packed_error = float(np.max(np.abs(endpoint-z['endpoints'])))
            packed_path_error = float(np.max(np.abs(path-z['paths'])))
        assert max(packed_error, packed_path_error) < 1e-10, row['id']
        times, benchmark_values = {}, {}
        order = ['separate', 'packed'] if index % 2 == 0 else ['packed', 'separate']
        for mode in order:
            before = time.perf_counter()
            if mode == 'separate':
                value = np.stack([integrate(BlendedField(model, oracle, coefficient), tensor, 32)[0].numpy()
                                  for coefficient in LAMBDAS])
            else:
                value = packed_blends(model, oracle, tensor, 32)[0]
            times[mode] = time.perf_counter()-before
            benchmark_values[mode] = value
        assert np.allclose(benchmark_values['packed'], benchmark_values['separate'], atol=1e-10, rtol=1e-10)
        assert snapshot == tensor_hashes(model)
        rows.append(dict(id=row['id'], parameters_and_buffers_unchanged=True, source_count=256,
            native_cached_endpoint_max_error=native_error, native_cached_path_max_error=native_path_error,
            packed_cached_endpoint_max_error=packed_error, packed_cached_path_max_error=packed_path_error,
            benchmark_steps=32, benchmark_order=order, benchmark_seconds=times,
            assets_sha256=binding['sha256'], original_frequency=float(model.time.frequencies[-1]), **identity))
        print(f'{len(rows)}/12 native and packed equivalence', flush=True)
    separate = sum(r['benchmark_seconds']['separate'] for r in rows)
    packed = sum(r['benchmark_seconds']['packed'] for r in rows)
    dump(out/'PASS.json', dict(passed=True, cases=12, controls=controls, analytic_packed_max_error=exact_packed_error,
        rows=rows, aggregate_separate_seconds=separate, aggregate_packed_seconds=packed,
        aggregate_packed_speedup=separate/packed, prefer_packed=packed < separate,
        source_sha256={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('native_time.py')]}))


if __name__ == '__main__':
    main()
