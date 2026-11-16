"""All-model derivative and whole-observer equivalence before choosing a backend."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from density_ops import HERE, base, torch, symmetry, MixtureOracle, exact_jacobian, sample_density
from jacobian_backend import batched_jacobian, derivative_backend


def main():
    p = argparse.ArgumentParser()
    for name in ['root', 'data', 'output']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--gpu-uuid')
    a = p.parse_args()
    root, data_root, out = map(Path, [a.root, a.data, a.output])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    if a.device == 'cuda':
        symmetry.exclusive_gpu(a.gpu_uuid)
        torch.backends.cuda.matmul.allow_tf32 = False
    cases = json.loads((root/'portable.json').read_text())['cases']
    meta = json.loads((data_root/'density-data.json').read_text())
    rows = []
    for case in cases:
        if a.device == 'cuda':
            symmetry.exclusive_gpu(a.gpu_uuid)
        asset = meta['datasets'][case['dataset']]
        assert base.digest(data_root/asset['file']) == asset['sha256']
        with np.load(data_root/asset['file']) as z:
            data = {k: z[k].copy() for k in z.files}
        tensor = lambda v: torch.tensor(v, dtype=torch.double, device=a.device)
        model = symmetry.load(root, case, a.device).double().requires_grad_(False)
        snapshot = symmetry.ev.tensor_snapshot(model)
        oracle = MixtureOracle(symmetry.law_for(root, case)).to(a.device)
        # Multiple clocks and marginal states, not only one initialization point.
        select = np.linspace(0, 1279, 32, dtype=int)
        x, t = tensor(data['field_x'][select]), tensor(data['field_t'][select])
        for mode in ['original', 'projected']:
            net = model if mode == 'original' else symmetry.Projection(model, tensor(data['group']))
            ref_value, ref_jac = exact_jacobian(net, x, t)
            value, jac = batched_jacobian(net, x, t)
            assert torch.equal(ref_value, value)
            max_jac = float((ref_jac-jac).abs().max())
            assert torch.allclose(ref_jac, jac, atol=2e-9, rtol=2e-9)
            source = tensor(data['source_0'][:4])
            observations, timing = {}, {}
            for backend in (['coordinate', 'batched'] if len(rows) % 2 == 0 else ['batched', 'coordinate']):
                if a.device == 'cuda':
                    torch.cuda.synchronize()
                before = time.monotonic()
                with derivative_backend(backend):
                    observations[backend] = sample_density(net, oracle, source, 32)
                if a.device == 'cuda':
                    torch.cuda.synchronize()
                timing[backend] = time.monotonic()-before
            assert torch.equal(observations['coordinate']['generated'], observations['batched']['generated'])
            errors = {k: float((observations['coordinate'][k]-observations['batched'][k]).abs().max())
                      for k in ['log_ratio_source', 'log_ratio_divergence', 'log_ratio_discrete_local',
                                'time_parts', 'integral_divergence', 'integral_local_logdet']}
            assert max(errors.values()) < 1e-8
            rows.append(dict(id=case['id'], mode=mode, jacobian_max_error=max_jac,
                native_endpoints_exact=True, observer_array_errors=errors, timing=timing))
            base.dump(out/'progress.json', dict(completed=len(rows), total=16, last=rows[-1]))
        symmetry.ev.same_tensors(model, snapshot)
    if a.device == 'cuda':
        symmetry.exclusive_gpu(a.gpu_uuid)
    base.dump(out/'COMPLETE.json', dict(passed=True, device=a.device, cells=16, rows=rows,
        density_data_sha256=base.digest(data_root/'density-data.json'),
        source_sha256={f.name: base.digest(f) for f in [Path(__file__), HERE/'jacobian_backend.py', HERE/'density_ops.py']}))
    print(json.dumps(dict(passed=True, cells=16, max_jacobian_error=max(r['jacobian_max_error'] for r in rows),
        max_observer_error=max(max(r['observer_array_errors'].values()) for r in rows))))


if __name__ == '__main__':
    main()
