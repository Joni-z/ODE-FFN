"""Read-only input and loss-response geometry of archived linear projections."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch


def capture(model, x: torch.Tensor, t: torch.Tensor):
    from synthetic.models import OriginalODELayer, SwiGLUFFN
    nodes, hooks = {}, []
    names = {id(p): name for name, p in model.named_parameters()}

    def linear_hook(key, role):
        def hook(module, args, result):
            nodes[key] = dict(input=args[0], output=result, parameter=module.weight,
                              parameter_name=names[id(module.weight)], role=role)
        return hook

    def ode_hook(key):
        def hook(module, args, result):
            if (module.orders != 1 or module.gate_source != 'input' or module.spectral_cap is not None
                    or getattr(module,'rate_placement','after_projection')!='after_projection'):
                raise ValueError('geometry factorization requires the independent plain K1 layer')
            raw = nodes[key + '.clock_out']['output']
            with torch.no_grad():
                duration = module.duration_from_logit(raw)
            nodes[key + '.P'] = dict(input=args[0], output=result, parameter=module.proj.weight,
                parameter_name=names[id(module.proj.weight)], role='P', duration=duration)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, OriginalODELayer):
            hooks.append(module.t_func[0].register_forward_hook(linear_hook(name + '.clock_in', 'clock_in')))
            hooks.append(module.t_func[-1].register_forward_hook(linear_hook(name + '.clock_out', 'clock_out')))
            hooks.append(module.register_forward_hook(ode_hook(name)))
        elif isinstance(module, SwiGLUFFN):
            hooks.append(module.w12.register_forward_hook(linear_hook(name + '.w12', 'swiglu_encoder')))
            hooks.append(module.w3.register_forward_hook(linear_hook(name + '.w3', 'swiglu_decoder')))
    try:
        prediction = model(x, t)
    finally:
        for hook in hooks:
            hook.remove()
    if not nodes:
        raise ValueError('no supported projection nodes')
    return prediction, dict(sorted(nodes.items()))


def moment_stats(matrix: np.ndarray) -> dict:
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)
    maximum = max(float(eigenvalues[-1]), 0.)
    tolerance = max(1e-12, maximum * 1e-10)
    assert eigenvalues[0] >= -max(1e-10, maximum * 1e-10)
    positive = eigenvalues[eigenvalues > tolerance]
    return dict(rank=int(len(positive)), dimension=len(eigenvalues), rank_tolerance=tolerance,
        eigen_min=float(eigenvalues[0]), eigen_max=maximum, trace=float(np.trace(matrix)),
        positive_condition=float(positive[-1] / positive[0]) if len(positive) else None,
        participation=float(positive.sum()**2 / (positive**2).sum()) if len(positive) else 0.,
        positive_eigen_quantiles=np.quantile(positive, [0, .25, .5, .75, 1]).tolist() if len(positive) else [])


@torch.no_grad()
def moments(model, x: torch.Tensor, t: torch.Tensor) -> dict:
    accumulators = {}
    for start in range(0, len(x), 512):
        _, nodes = capture(model, x[start:start+512], t[start:start+512])
        for name, node in nodes.items():
            h = node['input'].numpy()
            if name not in accumulators:
                accumulators[name] = dict(matrix=np.zeros((h.shape[1], h.shape[1])),
                    role=node['role'], parameter_name=node['parameter_name'])
            accumulators[name]['matrix'] += h.T @ h / len(x)
    return accumulators


def responses(model, x: torch.Tensor, t: torch.Tensor, targets: dict[str, torch.Tensor]):
    prediction, nodes = capture(model, x, t)
    named = dict(model.named_parameters())
    gradients, totals = {}, {}
    residuals = {name: prediction - value for name, value in targets.items()}
    for kind, residual in residuals.items():
        values = torch.autograd.grad(residual.square().mean(), tuple(named.values()), retain_graph=True)
        gradients[kind] = {name: value.detach().numpy() for name, value in zip(named, values)}
        totals[kind] = float(sum(value.detach().square().sum() for value in values))
    derivatives = {name: [] for name in nodes}
    for axis in range(prediction.shape[1]):
        values = torch.autograd.grad(prediction[:, axis].sum(), [n['output'] for n in nodes.values()],
                                     retain_graph=axis+1 < prediction.shape[1])
        for name, value in zip(nodes, values):
            derivatives[name].append(value.detach())
    result = {}
    maximum_error = 0.
    for name, node in nodes.items():
        h = node['input'].detach().numpy()
        response = torch.stack(derivatives[name], dim=1).numpy()
        if node['role'] == 'P':
            response = response * node['duration'].numpy()[:, None, :]
        output_moment = np.einsum('bdi,bdj->ij', response, response) / len(x)
        response_weight = (response**2).sum(axis=(1, 2))
        weighted_input = np.einsum('b,bi,bj->ij', response_weight, h, h) / len(x)
        if response_weight.mean() > 0:
            weighted_input /= response_weight.mean()
        info = dict(input=h, response=response, output_moment=output_moment,
                    weighted_input_moment=weighted_input, role=node['role'], gradients={}, energy_moments={})
        for kind, residual in residuals.items():
            q = np.einsum('bdi,bd->bi', response, residual.detach().numpy())
            gradient = 2 / prediction.numel() * q.T @ h
            expected = gradients[kind][node['parameter_name']]
            np.testing.assert_allclose(gradient, expected, atol=2e-10, rtol=2e-9)
            maximum_error = max(maximum_error, float(np.max(np.abs(gradient-expected))))
            info['gradients'][kind] = gradient
            info['energy_moments'][kind] = 4 / prediction.shape[1]**2 / len(x) * np.einsum(
                'b,bi,bj->ij', (q**2).sum(axis=1), h, h)
        result[name] = info
    return result, totals, maximum_error


def diagnose(study: Path, source: Path, data_root: Path, dataset: str, output: Path) -> dict:
    sys.path.insert(0, str(source))
    from synthetic.models import VelocityNet
    from synthetic.train import load_distribution, restore_model, sha256
    torch.set_num_threads(1)
    provenance = json.loads((study/'provenance.json').read_text())
    for name, digest in provenance['source_sha256'].items():
        assert sha256(source/name.replace('\\', '/')) == digest
    assert sha256(data_root/'manifest.json') == provenance['dataset_manifest_sha256']
    manifest = json.loads((data_root/'manifest.json').read_text())
    assert dataset in provenance['plan']['datasets']
    target_file = data_root/dataset/'train.npz'
    assert sha256(target_file) == manifest['files'][dataset+'/train.npz']['sha256']
    distribution = load_distribution(manifest['datasets'][dataset])
    with np.load(target_file, allow_pickle=False) as bank:
        targets = torch.from_numpy(bank['x1'].copy())
    generator = torch.Generator().manual_seed(250000)
    index = torch.randint(len(targets), (4096,), generator=generator)
    target = targets[index]
    noise = torch.randn(target.shape, generator=generator)
    time = torch.rand((len(target), 1), generator=generator)
    x = (1-time)*noise + time*target
    pair = target-noise
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output/'calibration.npz', x=x.numpy(), t=time.numpy(), indices=index.numpy(),
                        source=noise.numpy(), pair_velocity=pair.numpy())
    x, time = x.double(), time.double()
    oracle = torch.from_numpy(distribution.velocity(x[:128].numpy(), time[:128].numpy()))
    fixed = {}
    reference = study/'diagnostics'/dataset/'vector_k1'
    fixed_summary = json.loads((reference/'summary.json').read_text())
    for value in (.1, .3, .5, .7, .9):
        path = reference/f'common-{value}.npz'
        assert sha256(path) == fixed_summary['arrays_sha256'][path.name]
        with np.load(path, allow_pickle=False) as bank:
            xx = torch.from_numpy(bank['x'].copy())
        fixed[value] = (xx, torch.full((len(xx), 1), value, dtype=torch.float64))
    labels = ['vector_k1', 'vector_reparam', 'swiglu_budget', 'swiglu']
    moment_rows, response_rows, checkpoints = [], [], {}
    largest_error = 0.
    for label in labels:
        config_path = study/'configs'/f'{label}.json'
        config = json.loads(config_path.read_text())
        kind = config['models'][0]
        for step in (0, 1000, 5000):
            path = study/'runs'/config['experiment_id']/dataset/kind/'seed000/checkpoints'/f'step_{step:07d}.pt'
            state = torch.load(path, map_location='cpu', weights_only=False)
            assert state['identity'] == dict(dataset=dataset, model=kind, seed=0)
            assert state['config_sha256'] == sha256(config_path)
            assert state['dataset_manifest_sha256'] == provenance['dataset_manifest_sha256']
            checkpoints[str(path)] = sha256(path)
            model = VelocityNet(distribution.dim, kind, **config['model_options']).double().eval()
            restore_model(model, state)
            uniform = moments(model, x, time)
            arrays = {}
            for panel, (xx, tt) in [('uniform', (x, time)), *[(str(v), batch) for v, batch in fixed.items()]]:
                measured = uniform if panel == 'uniform' else moments(model, xx, tt)
                for name, node in measured.items():
                    row = dict(dataset=dataset, label=label, step=step, panel=panel, node=name,
                               role=node['role'], **moment_stats(node['matrix']))
                    if panel != 'uniform' and name in ('blocks.0.ffn.ode1.P', 'blocks.0.ffn.ode1.clock_in', 'blocks.0.ffn.w12'):
                        row['structural_rank_bound'] = distribution.dim+2
                        assert row['rank'] <= row['structural_rank_bound'], row
                    moment_rows.append(row)
                    arrays[name+'__G__'+panel] = node['matrix']
            measured, totals, error = responses(model, x[:128], time[:128],
                dict(pair=pair[:128].double(), population=oracle))
            largest_error = max(largest_error, error)
            for name, node in measured.items():
                matrix = uniform[name]['matrix']
                eigen, basis = np.linalg.eigh((matrix+matrix.T)/2)
                active = np.flatnonzero(eigen > max(1e-12, eigen[-1]*1e-10))
                lower = basis[:, active[:max(1, len(active)//4)]]
                row = dict(dataset=dataset, label=label, step=step, node=name, role=node['role'],
                    input_dimension=node['input'].shape[1], output_dimension=node['response'].shape[-1],
                    output_response=moment_stats(node['output_moment']),
                    response_weighted_input=moment_stats(node['weighted_input_moment']),
                    lowest_positive_quarter_count=lower.shape[1], losses={})
                for loss, gradient in node['gradients'].items():
                    energy = float((gradient**2).sum())
                    per_example = node['energy_moments'][loss]
                    row['losses'][loss] = dict(weight_gradient_energy=energy,
                        fraction_of_all_parameter_gradient=energy/totals[loss] if totals[loss] else None,
                        coherent_gradient_low_input_share=float(((gradient@lower)**2).sum()/energy) if energy else None,
                        individual_gradient_low_input_share=float(np.trace(lower.T@per_example@lower)/np.trace(per_example))
                            if np.trace(per_example)>0 else None)
                    arrays[name+'__gradient__'+loss] = gradient
                    arrays[name+'__energy__'+loss] = per_example
                arrays[name+'__B'] = node['output_moment']
                arrays[name+'__response_weighted_G'] = node['weighted_input_moment']
                response_rows.append(row)
            np.savez_compressed(output/f'{label}-{step}.npz', **arrays)
            print(json.dumps(dict(label=label, step=step, gradient_identity_error=error)), flush=True)
    result = dict(state='COMPLETE', dataset=dataset, labels=labels, steps=[0, 1000, 5000],
        moments=moment_rows, responses=response_rows, maximum_gradient_identity_error=largest_error,
        calibration_points=4096, response_points=128, calibration_rng_seed=250000,
        dtype='float64 analysis of archived float32 weights and saved runtime buffers',
        scope='Unclipped diagnostic pair gradients and population-oracle counterfactual gradients on bank-derived inputs. Population gradients are not conditional-mean gradients of the empirical-bank objective. No training or architecture selection.',
        source_sha256=sha256(Path(__file__)), checkpoint_sha256=checkpoints,
        manifest_sha256=sha256(data_root/'manifest.json'), train_targets_sha256=sha256(target_file),
        runtime_source=str(source), torch=torch.__version__, numpy=np.__version__, host=platform.node(),
        files_sha256={p.name:sha256(p) for p in sorted(output.glob('*.npz'))})
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'source', 'data-root', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()
    result = diagnose(args.study, args.source, args.data_root, args.dataset, args.output)
    print(json.dumps(dict(state=result['state'], maximum_gradient_identity_error=result['maximum_gradient_identity_error'])))
