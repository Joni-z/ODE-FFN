"""Refine unresolved sources; retain the full registered population unchanged."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from density_ops import HERE, base, torch, symmetry, MixtureOracle, sample_density
from jacobian_backend import derivative_backend
from verify_formal import verify_case


POINT_KEYS = ('generated', 'log_ratio_source', 'log_ratio_divergence',
              'log_ratio_discrete_local', 'chain_residual', 'time_parts',
              'integral_divergence', 'integral_local_logdet', 'endpoint_log_target')


def point_errors(coarse, fine, dim):
    return np.column_stack((
        np.abs(fine['chain_residual'])/dim,
        np.abs(fine['log_ratio_divergence']-fine['log_ratio_discrete_local'])/dim,
        np.maximum(*(np.abs(fine[k]-coarse[k]) for k in
                     ['log_ratio_source', 'log_ratio_divergence']))/dim,
        np.linalg.norm(fine['generated']-coarse['generated'], axis=1)/
        (1+np.linalg.norm(fine['generated'], axis=1))))


def accepted(errors, nonpositive):
    return (errors <= np.array([.001, .001, .002, .001])).all(1) & (nonpositive == 0)


def read_npz(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def main(argv=None):
    p = argparse.ArgumentParser()
    for name in ['root', 'data', 'case-output', 'parent-commit', 'output']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--derivative-backend', choices=['coordinate', 'batched'], default='coordinate')
    p.add_argument('--gpu-uuid')
    p.add_argument('--witness', action='store_true')
    p.add_argument('--readiness')
    a = p.parse_args(argv)
    root, data_root, parent, out = map(Path, [a.root, a.data, a.case_output, a.output])
    old = json.loads((parent/'COMPLETE.json').read_text())
    phase = old['phase']
    assert phase == 'formal' or a.witness
    verified = verify_case(root, data_root, parent, a.parent_commit, phase)
    source_files = [Path(__file__), HERE/'density_ops.py', HERE/'jacobian_backend.py']
    source_hashes = {f.name: base.digest(f) for f in source_files}
    if not a.witness:
        assert a.readiness
        ready = json.loads(Path(a.readiness).read_text())
        assert ready['passed'] and ready['witness'] and ready['device'] == a.device
        assert ready['derivative_backend'] == a.derivative_backend
        assert ready['source_sha256'] == source_hashes
        assert ready['density_data_sha256'] == base.digest(data_root/'density-data.json')
    manifest, meta = [json.loads(f.read_text()) for f in [root/'portable.json', data_root/'density-data.json']]
    case = next(c for c in manifest['cases'] if c['id'] == old['id'])
    asset = meta['datasets'][case['dataset']]
    data = read_npz(data_root/asset['file'])
    sources = np.concatenate([data[f'source_{b}'] for b in range(4)])
    assert sources.shape == (1024, sources.shape[1])
    if a.witness:
        sources = sources[:4]
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    last_guard = -1e30
    def guard(force=False):
        nonlocal last_guard
        if a.device == 'cuda' and (force or time.monotonic()-last_guard >= 5):
            assert a.gpu_uuid
            symmetry.exclusive_gpu(a.gpu_uuid)
            last_guard = time.monotonic()
    guard(True)
    if a.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
    tensor = lambda x: torch.tensor(x, dtype=torch.double, device=a.device)
    model = symmetry.load(root, case, a.device).double().requires_grad_(False)
    snapshot = symmetry.ev.tensor_snapshot(model)
    oracle = MixtureOracle(symmetry.law_for(root, case)).to(a.device)
    group = tensor(data['group'])
    rows, cells = [], []
    started = time.monotonic()
    for mode in ['original', 'projected']:
        net = model if mode == 'original' else symmetry.Projection(model, group)
        def observe(indices, steps):
            guard(True)
            def progress(completed, total):
                guard()
                base.dump(out/'progress.json', dict(pid=os.getpid(), id=case['id'], mode=mode,
                    witness=a.witness, steps=steps, selected_sources=len(indices),
                    completed_steps=completed, total_steps=total, rows_completed=len(rows),
                    seconds=time.monotonic()-started))
            source = tensor(sources[indices])
            before = time.monotonic()
            with derivative_backend(a.derivative_backend):
                result = sample_density(net, oracle, source, steps, progress)
            native = base.train.generated_samples(net, source, steps)
            assert torch.equal(native, result['generated'])
            value = {k: v.cpu().numpy() for k, v in result.items()}
            assert float(value['jacobian_value_error']) < 1e-10
            file = mode+f'-subset-heun{steps}.npz'
            np.savez_compressed(out/file, flat_indices=indices, **value)
            rows.append(dict(mode=mode, steps=steps, samples=len(indices), file=file,
                sha256=base.digest(out/file), native_endpoints_exact=True,
                nonpositive_steps=int(value['nonpositive_steps']), seconds=time.monotonic()-before))
            base.dump(out/'rows.json', rows)
            return value
        n, dim = sources.shape
        if a.witness:
            coarse = observe(np.arange(n), 16)
            fine = observe(np.arange(n), 32)
            nonpositive = np.full(n, int(fine['nonpositive_steps']))
            coarse = {k: coarse[k] for k in POINT_KEYS}
            fine = {k: fine[k] for k in POINT_KEYS}
            steps, previous_steps = np.full(n, 32), np.full(n, 16)
        else:
            values, counts = {}, []
            for level in [1024, 2048]:
                chunks = [read_npz(parent/f'{mode}-bank{b}-heun{level}.npz') for b in range(4)]
                values[level] = {k: np.concatenate([z[k] for z in chunks]) for k in POINT_KEYS}
                if level == 2048:
                    counts = np.concatenate([np.full(256, int(z['nonpositive_steps'])) for z in chunks])
            coarse, fine = values[1024], values[2048]
            steps, previous_steps = np.full(n, 2048), np.full(n, 1024)
            nonpositive = counts
        errors = point_errors(coarse, fine, dim)
        qualified = accepted(errors, nonpositive)
        initial_unresolved = np.flatnonzero(~qualified).tolist()
        if not a.witness:
            for level in [4096, 8192, 16384]:
                indices = np.flatnonzero(~qualified)
                if not len(indices):
                    break
                refined = observe(indices, level)
                for key in POINT_KEYS:
                    coarse[key][indices] = fine[key][indices]
                    fine[key][indices] = refined[key]
                previous_steps[indices] = steps[indices]
                steps[indices] = level
                nonpositive[indices] = int(refined['nonpositive_steps'])
                errors = point_errors(coarse, fine, dim)
                qualified = accepted(errors, nonpositive)
        file = f'{mode}-all-sources.npz'
        merged = dict(fine, **{'previous_'+k: v for k, v in coarse.items()},
            bank=np.arange(n)//256, source_index=np.arange(n)%256,
            selected_steps=steps, previous_steps=previous_steps, errors=errors,
            nonpositive_steps_bound=nonpositive, qualified=qualified)
        np.savez_compressed(out/file, **merged)
        cells.append(dict(mode=mode, samples=n, file=file, sha256=base.digest(out/file),
            initial_unresolved_indices=initial_unresolved,
            unresolved_indices=np.flatnonzero(~qualified).tolist(),
            numerical_qualified=bool(qualified.all()) and not a.witness,
            resolution_counts={str(k): int((steps == k).sum()) for k in np.unique(steps)},
            max_errors=errors.max(0).tolist()))
    symmetry.ev.same_tensors(model, snapshot)
    guard(True)
    base.dump(out/'COMPLETE.json', dict(passed=True, witness=a.witness, adaptive=True,
        id=case['id'], dataset=case['dataset'], family=case['family'], device=a.device,
        derivative_backend=a.derivative_backend, gpu_uuid=a.gpu_uuid,
        parameters_unchanged=True, parent_completion_sha256=verified['completion_sha256'],
        parent_folder=str(parent), parent_commit=a.parent_commit,
        density_data_sha256=base.digest(data_root/'density-data.json'),
        checkpoint_sha256=case['checkpoint_sha256'], source_sha256=source_hashes,
        readiness_sha256=base.digest(Path(a.readiness)) if a.readiness else None,
        rows=rows, cells=cells, seconds=time.monotonic()-started))
    print(json.dumps(dict(passed=True, witness=a.witness, id=case['id'],
        cells=[dict(mode=c['mode'], qualified=c['numerical_qualified'],
                    unresolved=len(c['unresolved_indices']), resolutions=c['resolution_counts']) for c in cells])))


if __name__ == '__main__':
    main()
