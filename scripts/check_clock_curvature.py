"""Measure the centered K2 gate term on common states, without changing weights."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Iterator

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.check_gate_geometry import jacobian, oracle_geometry
from synthetic.models import VelocityNet
from synthetic.train import load_distribution, load_targets, sha256


@contextmanager
def remove_curvature(model: VelocityNet, names: list[str]) -> Iterator[None]:
    changed = []
    try:
        for name, layer in model.ode_layers():
            if name in names:
                changed.append((layer, layer.clock_curvature))
                layer.clock_curvature = 0.0
        yield
    finally:
        for layer, prior in changed:
            layer.clock_curvature = prior


def check(checkpoint: Path, config: Path, data_root: Path, output: Path,
          points: int = 512) -> dict:
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(config.read_text())
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert state['config_sha256'] == sha256(config)
    assert state['dataset_manifest_sha256'] == sha256(data_root/'manifest.json')
    dataset, kind = state['identity']['dataset'], state['identity']['model']
    model = VelocityNet(2, kind, **cfg['model_options']).double().eval()
    model.load_state_dict(state['model'], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    targets = load_targets(data_root/dataset/'validation.npz', 2)[:points].astype(np.float64)
    source = np.random.default_rng(0).standard_normal((len(targets), 2))
    manifest = json.loads((data_root/'manifest.json').read_text())
    dist = load_distribution(manifest['datasets'][dataset])
    names = [name for name, _ in model.ode_layers()]
    variants = [('native', [])]
    if kind == 'ode_original':
        variants += [('all_removed', names)] + [(name, [name]) for name in names]
    rows = []
    for t in (.1, .3, .5, .7, .9):
        x = torch.from_numpy((1-t)*source+t*targets)
        tt = torch.full((len(x), 1), t, dtype=torch.float64)
        truth = oracle_geometry(dist, x.numpy(), t)
        native_prediction, native_jacobian = jacobian(model, x, tt)
        for variant, selected in variants:
            with remove_curvature(model, selected):
                prediction, derivative = jacobian(model, x, tt)
            pred, jac = prediction.numpy(), derivative.numpy()
            field_change = pred-native_prediction.numpy()
            jac_change = jac-native_jacobian.numpy()
            row = dict(time=t, variant=variant,
                velocity_mse=float(np.mean((pred-truth['oracle'])**2)),
                jacobian_mse=float(np.mean(np.sum((jac-truth['oracle_jacobian'])**2, axis=(1, 2)))),
                field_change_squared=float(np.mean(field_change**2)),
                field_reference_squared=float(native_prediction.square().mean()),
                jacobian_change_squared=float(np.mean(np.sum(jac_change**2, axis=(1, 2)))),
                jacobian_reference_squared=float(native_jacobian.square().sum((1, 2)).mean()))
            rows.append(row)
            path = output/f'{variant.replace(".", "_")}-{t}.npz'
            np.savez_compressed(path, x=x.numpy(), prediction=pred, jacobian=jac,
                oracle=truth['oracle'], oracle_jacobian=truth['oracle_jacobian'])
    aggregate = {}
    for variant, _ in variants:
        selected = [r for r in rows if r['variant'] == variant]
        avg = {key:float(np.mean([r[key] for r in selected])) for key in selected[0]
               if key not in ('time', 'variant')}
        avg['field_relative_rms_change'] = float(np.sqrt(avg['field_change_squared']/max(avg['field_reference_squared'], 1e-15)))
        avg['jacobian_relative_rms_change'] = float(np.sqrt(avg['jacobian_change_squared']/max(avg['jacobian_reference_squared'], 1e-15)))
        aggregate[variant] = avg
    result = dict(dataset=dataset, model=kind, checkpoint_step=state['step'],
        checkpoint_sha256=sha256(checkpoint), config_sha256=sha256(config),
        manifest_sha256=sha256(data_root/'manifest.json'),
        validation_sha256=sha256(data_root/dataset/'validation.npz'),
        points=len(targets), times=[.1,.3,.5,.7,.9], gate_center=.6,
        interpretation='Frozen-weight interventions; dependence is not retraining superiority.',
        aggregate=aggregate, rows=rows,
        arrays_sha256={p.name:sha256(p) for p in sorted(output.glob('*.npz'))},
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in
            [Path(__file__), ROOT/'synthetic/models.py', ROOT/'scripts/check_gate_geometry.py']})
    if 'all_removed' in aggregate:
        removed = aggregate['all_removed']
        result['curvature_material'] = max(removed['field_relative_rms_change'],
            removed['jacobian_relative_rms_change']) >= .01
    (output/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'config', 'data-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--points', type=int, default=512)
    args = parser.parse_args()
    result = check(args.checkpoint, args.config, args.data_root, args.output, args.points)
    print(json.dumps({key:value for key,value in result.items() if key not in ('rows',)}))
