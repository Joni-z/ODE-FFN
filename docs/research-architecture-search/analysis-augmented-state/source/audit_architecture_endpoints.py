"""Replay saved endpoint fields from a collected archive and its frozen source.

This does not require intermediate training checkpoints and does not claim to
replay optimization, sampling trajectories, or a fresh holdout evaluation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


def audit(study: Path, source: Path, data_root: Path, output: Path) -> dict:
    sys.path.insert(0, str(source))
    from synthetic.models import VelocityNet
    from synthetic.train import load_distribution, restore_model, sha256
    from scripts.check_gate_geometry import jacobian, oracle_geometry

    torch.set_num_threads(1)
    provenance = json.loads((study / 'provenance.json').read_text())
    plan = provenance['plan']
    status = json.loads((study / 'status.json').read_text())
    assert status['state'] == 'COMPLETE'
    assert status['completed'] == status['total'] == len(plan['arms']) * len(plan['datasets'])
    for path, digest in provenance['source_sha256'].items():
        assert sha256(source / path.replace('\\', '/')) == digest, path
    assert sha256(data_root / 'manifest.json') == provenance['dataset_manifest_sha256']
    manifest = json.loads((data_root / 'manifest.json').read_text())
    rows = json.loads((study / 'results.json').read_text())
    checkpoints = arrays = 0
    forward_max = derivative_max = 0.
    hashes = {}
    for dataset in plan['datasets']:
        distribution = load_distribution(manifest['datasets'][dataset])
        common = {}
        for arm in plan['arms']:
            label, kind = arm['label'], arm['model']
            cfg_path = study / 'configs' / f'{label}.json'
            cfg = json.loads(cfg_path.read_text())
            folder = study / 'runs' / cfg['experiment_id'] / dataset / kind / 'seed000'
            assert (folder / 'COMPLETE').read_text().strip() == str(plan['steps'])
            assert sha256(folder / 'config.json') == sha256(cfg_path)
            checkpoint = folder / 'checkpoints' / f'step_{plan["steps"]:07d}.pt'
            hashes[str(checkpoint.relative_to(study))] = sha256(checkpoint)
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            assert state['step'] == plan['steps']
            assert state['identity'] == dict(dataset=dataset, model=kind, seed=0)
            assert state['config_sha256'] == sha256(cfg_path)
            assert state['dataset_manifest_sha256'] == sha256(data_root / 'manifest.json')
            model = VelocityNet(distribution.dim, kind, **cfg['model_options']).double().eval()
            restore_model(model, state)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            diagnostic = study / 'diagnostics' / dataset / label
            summary = json.loads((diagnostic / 'summary.json').read_text())
            assert summary['checkpoint_sha256'] == sha256(checkpoint)
            assert summary['config_sha256'] == sha256(cfg_path)
            assert summary['manifest_sha256'] == provenance['dataset_manifest_sha256']
            assert summary['split'] == 'validation'
            assert summary['targets_sha256'] == sha256(data_root / dataset / 'validation.npz')
            errors = []
            assert len(summary['arrays_sha256']) == 5
            for name, digest in sorted(summary['arrays_sha256'].items()):
                path = diagnostic / name
                assert sha256(path) == digest
                hashes[str(path.relative_to(study))] = digest
                time = float(path.stem.removeprefix('common-'))
                with np.load(path, allow_pickle=False) as a:
                    if name in common:
                        np.testing.assert_array_equal(common[name], a['x'])
                    else:
                        common[name] = a['x'].copy()
                    truth = oracle_geometry(distribution, a['x'], time)
                    np.testing.assert_allclose(a['oracle'], truth['oracle'], atol=1e-11, rtol=1e-11)
                    np.testing.assert_allclose(a['oracle_jacobian'], truth['oracle_jacobian'], atol=1e-10, rtol=1e-11)
                    x = torch.from_numpy(a['x'])
                    pred, derivative = jacobian(model, x, torch.full((len(x), 1), time, dtype=x.dtype))
                    np.testing.assert_allclose(a['prediction'], pred.numpy(), atol=1e-10, rtol=1e-10)
                    np.testing.assert_allclose(a['jacobian'], derivative.numpy(), atol=1e-9, rtol=1e-10)
                    forward_max = max(forward_max, float(np.max(np.abs(pred.numpy() - a['prediction']))))
                    derivative_max = max(derivative_max, float(np.max(np.abs(derivative.numpy() - a['jacobian']))))
                    errors.append([np.mean((pred.numpy() - truth['oracle']) ** 2),
                                   np.mean(np.sum((derivative.numpy() - truth['oracle_jacobian']) ** 2, axis=(1, 2)))])
                arrays += 1
            row = next(r for r in rows if r['dataset'] == dataset and r['label'] == label)
            np.testing.assert_allclose(np.mean(errors, axis=0), [row['field_mse'], row['jacobian_error']],
                                       atol=1e-10, rtol=1e-10)
            checkpoints += 1
    result = dict(state='PASS', final_checkpoints_reloaded=checkpoints, validation_fields_replayed=arrays,
                  maximum_forward_difference=forward_max, maximum_derivative_difference=derivative_max,
                  source_verified=True, oracles_recomputed=True, common_inputs_match=True,
                  scope='Final common-validation velocity/Jacobian fields. No training, sampling-trajectory, or fresh-holdout replay.',
                  script_sha256=sha256(Path(__file__)), files_sha256=hashes)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    return {k: v for k, v in result.items() if k != 'files_sha256'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'source', 'data-root', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.study, args.source, args.data_root, args.output), indent=2))
