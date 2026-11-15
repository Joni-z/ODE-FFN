"""Seeded analytical/finite-difference tests and real native-sampler observers."""
import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
from density_ops import HERE, base, torch, symmetry, MixtureOracle, exact_jacobian, sample_density


def oracle_checks(law, device):
    oracle = MixtureOracle(law).to(device)
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.standard_normal((24, law.dim)), dtype=torch.double, device=device, requires_grad=True)
    t = torch.linspace(.02, .98, len(x), dtype=torch.double, device=device)[:, None].requires_grad_(True)
    true = oracle.evaluate(x, t)
    native_x, native_t = x.detach().cpu().numpy(), t.detach().cpu().numpy()
    velocity_error = float(np.max(np.abs(true['velocity'].detach().cpu().numpy()-law.velocity(native_x, native_t))))
    log_error = float(np.max(np.abs(true['log_density'].detach().cpu().numpy()-law.log_density(native_x, native_t))))
    assert velocity_error < 1e-9 and log_error < 1e-9
    score, time_derivative = torch.autograd.grad(true['log_density'].sum(), (x, t), retain_graph=True)
    actual_div = sum(torch.autograd.grad(true['velocity'][:, j].sum(), x, retain_graph=True)[0][:, j] for j in range(law.dim))
    score_error = float((score-true['score']).abs().max().detach())
    divergence_error = float((actual_div-true['divergence']).abs().max().detach())
    pde = time_derivative[:, 0]+true['divergence']+(true['velocity']*true['score']).sum(1)
    pde_error = float(pde.abs().max().detach())
    assert score_error < 1e-9 and divergence_error < 1e-9 and pde_error < 1e-8
    numerical = []
    for step in [1e-4, 3e-5, 1e-5]:
        score_fd = np.zeros_like(native_x)
        divergence_fd = np.zeros(len(x))
        for j in range(law.dim):
            delta = np.zeros_like(native_x)
            delta[:, j] = step
            score_fd[:, j] = (law.log_density(native_x+delta, native_t)-law.log_density(native_x-delta, native_t))/(2*step)
            divergence_fd += (law.velocity(native_x+delta, native_t)[:, j]-law.velocity(native_x-delta, native_t)[:, j])/(2*step)
        score_ref, div_ref = true['score'].detach().cpu().numpy(), true['divergence'].detach().cpu().numpy()
        numerical.append(dict(step=step, score_relative_max=float(np.max(np.abs(score_fd-score_ref)/(1+np.abs(score_ref)))),
                              divergence_relative_max=float(np.max(np.abs(divergence_fd-div_ref)/(1+np.abs(div_ref))))))
    assert numerical[-1]['score_relative_max'] < 2e-6 and numerical[-1]['divergence_relative_max'] < 2e-6
    return dict(passed=True, velocity_max_error=velocity_error, log_density_max_error=log_error,
                score_autograd_max_error=score_error, divergence_autograd_max_error=divergence_error,
                continuity_max_error=pde_error, native_finite_differences=numerical)


def gaussian_checks(device):
    class Stationary(torch.nn.Module):
        def evaluate(self, x, t):
            zero = x[:, 0]*0
            return dict(velocity=x*0, score=-x, divergence=zero,
                        log_density=-.5*(x.square().sum(1)+x.shape[1]*math.log(2*math.pi)))
    class Linear(torch.nn.Module):
        def __init__(self, matrix):
            super().__init__()
            self.register_buffer('matrix', matrix)
        def forward(self, x, t):
            return x@self.matrix.T
    torch.manual_seed(0)
    source = torch.randn(8, 2, dtype=torch.double, device=device)
    a = .3
    rotation = torch.tensor([[0., -a], [a, 0.]], dtype=torch.double, device=device)
    radial = a*torch.eye(2, dtype=torch.double, device=device)
    errors = []
    for name, matrix in [('rotation', rotation), ('radial', radial)]:
        model = Linear(matrix)
        exact_end = source@torch.matrix_exp(matrix).T
        expected_ratio = .5*(exact_end.square().sum(1)-source.square().sum(1))-torch.trace(matrix)
        refinements = []
        for steps in [16, 32, 64]:
            result = sample_density(model, Stationary(), source, steps)
            identity = torch.eye(2, dtype=torch.double, device=device)
            step_matrix = identity+matrix/steps+.5*(matrix@matrix)/steps**2
            discrete_end = source@torch.linalg.matrix_power(step_matrix, steps).T
            discrete_ratio = .5*(discrete_end.square().sum(1)-source.square().sum(1))-steps*torch.linalg.slogdet(step_matrix)[1]
            assert (discrete_end-result['generated']).abs().max() < 1e-12
            assert (discrete_ratio-result['log_ratio_discrete_local']).abs().max() < 1e-12
            assert int(result['nonpositive_steps']) == 0
            refinements.append(dict(steps=steps, endpoint_error=float((exact_end-result['generated']).abs().max()),
                source_ratio_error=float((expected_ratio-result['log_ratio_source']).abs().max()),
                divergence_ratio_error=float((expected_ratio-result['log_ratio_divergence']).abs().max()),
                chain_residual=float(result['chain_residual'].abs().max()),
                exact_discrete_logdet_error=float((discrete_ratio-result['log_ratio_discrete_local']).abs().max())))
        observed = math.log2(refinements[-2]['endpoint_error']/refinements[-1]['endpoint_error'])
        assert 1.95 < observed < 2.05
        assert refinements[-1]['source_ratio_error'] < 3e-5 and refinements[-1]['divergence_ratio_error'] < 3e-5
        errors.append(dict(name=name, observed_order=observed, refinements=refinements))
    rotation_mse = (source@rotation.T).square().mean()
    radial_mse = (source@radial.T).square().mean()
    assert abs(float(rotation_mse-radial_mse)) < 1e-14
    return dict(passed=True, equal_empirical_field_mse=True, examples=errors,
                radial_population_kl=.5*2*(math.exp(2*a)-1-2*a), rotational_population_kl=0.)


def model_checks(root, case, device, steps):
    model = symmetry.load(root, case, device).double().requires_grad_(False)
    law = symmetry.law_for(root, case)
    oracle = MixtureOracle(law).to(device)
    data = symmetry.arrays(root, case)
    x = torch.tensor(data['field_x'][:4], dtype=torch.double, device=device)
    t = torch.tensor(data['field_t'][:4], dtype=torch.double, device=device)
    value, jacobian = exact_jacobian(model, x, t)
    with torch.no_grad():
        direct = model(x, t)
        small = model(x[:1], t[:1])
    assert (direct-value).abs().max() < 1e-10 and (direct[:1]-small).abs().max() < 1e-10
    numerical = []
    for h in [1e-4, 3e-5, 1e-5]:
        columns = []
        with torch.no_grad():
            for j in range(law.dim):
                shift = torch.zeros_like(x)
                shift[:, j] = h
                columns.append((model(x+shift, t)-model(x-shift, t))/(2*h))
        finite = torch.stack(columns, 2)
        numerical.append(dict(step=h, max_relative_error=float(((finite-jacobian).abs()/(1+jacobian.abs())).max())))
    assert numerical[-1]['max_relative_error'] < 2e-6
    source = torch.tensor(data['source_0'][:4], dtype=torch.double, device=device)
    rows = []
    for mode in ['original', 'projected']:
        field = model if mode == 'original' else symmetry.Projection(model, torch.tensor(data['group'], device=device, dtype=torch.double))
        start = time.monotonic()
        result = sample_density(field, oracle, source, steps)
        native = base.train.generated_samples(field, source, steps)
        assert torch.equal(result['generated'], native), 'Observer changed the native sampler output'
        assert float(result['jacobian_value_error']) < 1e-10
        rows.append(dict(mode=mode, steps=steps, native_endpoints_exact=True,
                         chain_residual=float(result['chain_residual'].abs().max()),
                         nonpositive_step_jacobians=int(result['nonpositive_steps']),
                         seconds=time.monotonic()-start))
    return dict(passed=True, id=case['id'], finite_differences=numerical, entry_rows=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--data')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--oracle-only', action='store_true')
    parser.add_argument('--steps', default=16, type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.device == 'cuda':
        symmetry.exclusive_gpu(args.gpu_uuid)
        torch.backends.cuda.matmul.allow_tf32 = False
    base.verify_sources()
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root/'portable.json').read_text())
    kernels, models, checked = [], [], set()
    for case in manifest['cases']:
        if case['dataset'] not in checked:
            kernels.append(dict(dataset=case['dataset'], **oracle_checks(symmetry.law_for(root, case), args.device)))
            checked.add(case['dataset'])
    examples = gaussian_checks(args.device)
    base.dump(out/'kernel-checks.json', dict(passed=True, kernels=kernels, gaussian_examples=examples))
    meta = None
    if args.data:
        data_root = Path(args.data)
        meta = json.loads((data_root/'density-data.json').read_text())
        assert meta['passed'] and meta['portable_manifest_sha256'] == base.digest(root/'portable.json')
    if not args.oracle_only:
        for case in manifest['cases']:
            if args.device == 'cuda':
                symmetry.exclusive_gpu(args.gpu_uuid)
            row = model_checks(root, case, args.device, args.steps)
            if meta:
                asset = meta['datasets'][case['dataset']]
                ref = next(v for v in meta['references'] if v['id'] == case['id'])
                assert base.digest(data_root/asset['file']) == asset['sha256']
                assert base.digest(data_root/ref['file']) == ref['sha256']
                with np.load(data_root/asset['file']) as z:
                    data = {k: z[k].copy() for k in z.files}
                with np.load(data_root/ref['file']) as z:
                    refs = {k: z[k].copy() for k in z.files}
                model = symmetry.load(root, case, args.device).double().requires_grad_(False)
                tensor = lambda v: torch.tensor(v, dtype=torch.double, device=args.device)
                portability = []
                for mode in ['original', 'projected']:
                    net = model if mode == 'original' else symmetry.Projection(model, tensor(data['group']))
                    value, jac = exact_jacobian(net, tensor(data['field_x'][:4]), tensor(data['field_t'][:4]))
                    native = base.train.generated_samples(net, tensor(data['source_0'][:8]), 128).cpu().numpy()
                    errors = dict(mode=mode, field=float(np.max(np.abs(value.cpu().numpy()-refs[mode+'_field']))),
                                  jacobian=float(np.max(np.abs(jac.cpu().numpy()-refs[mode+'_jacobian']))),
                                  native128=float(np.max(np.abs(native-refs[mode+'_native128']))))
                    assert errors['field'] < 2e-9 and errors['jacobian'] < 2e-8 and errors['native128'] < 1e-7
                    portability.append(errors)
                row['cpu_portability'] = portability
            models.append(row)
            base.dump(out/'progress.json', dict(completed=len(models), total=8, last=models[-1]))
    entries = []
    if meta and not args.oracle_only:
        module_spec = importlib.util.spec_from_file_location('actual_density_probe', HERE/'density_probe.py')
        entry = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(entry)
        for index in [0, 7]:
            target = out/('entry-'+str(index))
            command = ['--root', str(root), '--data', str(data_root), '--index', str(index), '--output', str(target),
                       '--phase', 'witness', '--device', args.device]
            if args.gpu_uuid:
                command += ['--gpu-uuid', args.gpu_uuid]
            entry.main(command)
            done = json.loads((target/'COMPLETE.json').read_text())
            assert done['passed'] and done['witness'] and done['parameters_unchanged']
            entries.append(dict(index=index, completion_sha256=base.digest(target/'COMPLETE.json')))
    receipt = dict(passed=True, device=args.device, oracle_only=args.oracle_only, kernels=kernels,
                   gaussian_examples=examples, models=models, portable_manifest_sha256=base.digest(root/'portable.json'),
                   entry_witnesses=entries, density_data_sha256=base.digest(data_root/'density-data.json') if meta else None,
                   source_sha256={p.name: base.digest(p) for p in [Path(__file__), HERE/'density_ops.py', HERE/'density_probe.py']})
    base.dump(out/'COMPLETE.json', receipt)
    print(json.dumps(dict(passed=True, model_cases=len(models), output=str(out))))


if __name__ == '__main__':
    main()
