"""Read-only output-envelope check of saved LayerNorm velocity heads."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def squared_distance(rotated: np.ndarray, axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project onto a full-dimensional centered ellipsoid with positive axes."""
    assert np.all(axes > 0)
    radius = np.sum((rotated / axes) ** 2, axis=1)
    outside = radius > 1
    result = np.zeros(len(rotated))
    if np.any(outside):
        y = rotated[outside]
        axis2 = axes ** 2
        low = np.zeros(len(y))
        high = np.maximum(1., np.sum(axis2 * y ** 2, axis=1))
        for _ in range(85):
            middle = (low + high) / 2
            norm = np.sum(axis2 * y ** 2 / (axis2 + middle[:, None]) ** 2, axis=1)
            low = np.where(norm > 1, middle, low)
            high = np.where(norm > 1, high, middle)
        multiplier = (low + high) / 2
        projection = axis2 * y / (axis2 + multiplier[:, None])
        np.testing.assert_allclose(np.sum((projection / axes) ** 2, axis=1), 1, atol=2e-12, rtol=2e-12)
        result[outside] = np.sum((y - projection) ** 2, axis=1)
    return result, radius


def diagnose(root: Path, output: Path, protocol: Path) -> None:
    root, output, protocol = root.resolve(), output.resolve(), protocol.resolve()
    assert root not in output.parents and output != root and output not in root.parents
    assert not output.exists(), 'Use a fresh diagnostic output directory'
    hashes = {str(protocol): sha(protocol), str(Path(__file__).resolve()): sha(Path(__file__))}
    def read(path: Path) -> dict:
        hashes[str(path)] = sha(path)
        return json.loads(path.read_text())
    manifest = read(root / 'source-manifest.json')
    for name, digest in manifest['all_files'].items():
        assert sha(root / 'source' / name) == digest
    rows, totals = [], []
    for dataset, suffix in [('gmm_ring8_2d', 'v1'), ('rotated_product_bimodal_16d', 'product16')]:
        study = root / 'output' / f'shared_features_{suffix}_20261024_m1'
        assert read(study / 'status.json')['state'] == 'COMPLETE'
        for arm in ['vector_parent', 'swiglu_budget', 'swiglu']:
            folder = study / 'diagnostics' / dataset / arm
            native = read(folder / 'summary.json')
            config_path = study / 'configs' / f'{arm}.json'
            cfg = read(config_path)
            assert hashes[str(config_path)] == native['config_sha256']
            checkpoint = study / 'runs' / cfg['experiment_id'] / dataset / native['model'] / 'seed000/checkpoints/step_0005000.pt'
            hashes[str(checkpoint)] = sha(checkpoint)
            assert hashes[str(checkpoint)] == native['checkpoint_sha256']
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model']
            get = lambda name: state[name].double().numpy()
            weight, gamma, beta, bias = (get(k) for k in ['output.weight', 'norm.weight', 'norm.bias', 'output.bias'])
            width = len(gamma)
            center = weight @ beta + bias
            matrix = np.sqrt(width) * (weight * gamma) @ (np.eye(width) - np.ones((width, width)) / width)
            rotation, axes, _ = np.linalg.svd(matrix, full_matrices=False)
            assert axes.min() > axes.max() * 1e-12, 'Rank-deficient envelope requires a separate projection'
            per_arm = []
            for time in [.1, .3, .5, .7, .9]:
                path = folder / f'common-{time:.1f}.npz'
                hashes[str(path)] = sha(path)
                assert hashes[str(path)] == native['arrays_sha256'][path.name]
                with np.load(path, allow_pickle=False) as data:
                    prediction, target = data['prediction'], data['oracle']
                dimension = prediction.shape[1]
                distance, radius = squared_distance((target - center) @ rotation, axes)
                predicted_radius = np.sum((((prediction - center) @ rotation) / axes) ** 2, axis=1)
                assert predicted_radius.max() <= 1 + 2e-11
                mse = np.mean((prediction - target) ** 2)
                floor = distance.mean() / dimension
                assert floor <= mse + 1e-12
                row = dict(dataset=dataset, arm=arm, time=time, field_mse=float(mse), envelope_floor=float(floor),
                           floor_fraction=float(floor / mse), target_outside_fraction=float(np.mean(radius > 1)),
                           target_radius_max=float(radius.max()), prediction_radius_max=float(predicted_radius.max()),
                           minimum_semiaxis=float(axes.min()), maximum_semiaxis=float(axes.max()))
                rows.append(row)
                per_arm.append(row)
            mse = float(np.mean([r['field_mse'] for r in per_arm]))
            floor = float(np.mean([r['envelope_floor'] for r in per_arm]))
            np.testing.assert_allclose(mse, native['field_mse'], atol=1e-12, rtol=1e-12)
            totals.append(dict(dataset=dataset, arm=arm, field_mse=mse, envelope_floor=floor,
                               floor_fraction=floor / mse, target_outside_fraction=float(np.mean([r['target_outside_fraction'] for r in per_arm]))))
    # Exact sphere cases test the secular equation without mirroring its code.
    check, _ = squared_distance(np.array([[0., 0.], [3., 4.]]), np.array([2., 2.]))
    np.testing.assert_allclose(check, [0., 9.], atol=1e-12, rtol=1e-12)
    for name, digest in hashes.items():
        assert sha(Path(name)) == digest
    for name, digest in manifest['all_files'].items():
        assert sha(root / 'source' / name) == digest
    output.mkdir(parents=True)
    for name, records in [('per-panel.csv', rows), ('aggregate.csv', totals)]:
        with (output / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    (output / 'source').mkdir()
    for path in [Path(__file__).resolve(), protocol]:
        shutil.copy2(path, output / 'source' / path.name)
    result = dict(state='PASS', panels=len(rows), aggregates=totals,
                  input_sha256=hashes, source_files_verified=len(manifest['all_files']),
                  scope='Conditional saved-head envelope only; no architecture-wide lower bound or trained-head improvement.')
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(state='PASS', aggregates=totals), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    diagnose(args.root, args.output, args.protocol)
