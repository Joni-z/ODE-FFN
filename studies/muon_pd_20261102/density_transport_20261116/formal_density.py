"""Full common-bank density observation with unchanged native dynamics/sampler."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from density_ops import HERE, base, torch, symmetry, MixtureOracle, sample_density
from jacobian_backend import derivative_backend


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ['root', 'data', 'output']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--index', required=True, type=int)
    parser.add_argument('--phase', choices=['witness', 'benchmark', 'formal'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--readiness')
    parser.add_argument('--derivative-backend', choices=['coordinate', 'batched'], default='coordinate')
    a = parser.parse_args(argv)
    root, data_root, out = map(Path, [a.root, a.data, a.output])
    manifest, meta = [json.loads(p.read_text()) for p in [root/'portable.json', data_root/'density-data.json']]
    assert meta['passed'] and meta['portable_manifest_sha256'] == base.digest(root/'portable.json')
    assert meta['independent_banks'] == 4 and meta['points_per_bank'] == 256
    case = manifest['cases'][a.index]
    assert meta['references'][a.index]['id'] == case['id']
    if a.phase == 'formal':
        assert a.readiness
        ready = json.loads(Path(a.readiness).read_text())
        assert ready['passed'] and ready['device'] == a.device and ready['model_cases'] == 8
        assert ready['derivative_backend'] == a.derivative_backend
        assert len(ready['formal_witnesses']) == 2 and ready['density_data_sha256'] == base.digest(data_root/'density-data.json')
        for name in ['density_ops.py', 'formal_density.py', 'jacobian_backend.py']:
            assert ready['source_sha256'][name] == base.digest(HERE/name)
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
        if a.phase == 'formal':
            assert ready['device_name'] == torch.cuda.get_device_name(0)
    asset = meta['datasets'][case['dataset']]
    assert base.digest(data_root/asset['file']) == asset['sha256']
    with np.load(data_root/asset['file']) as z:
        data = {k: z[k].copy() for k in z.files}
    tensor = lambda x: torch.tensor(x, dtype=torch.double, device=a.device)
    model = symmetry.load(root, case, a.device).double().requires_grad_(False)
    snapshot = symmetry.ev.tensor_snapshot(model)
    oracle = MixtureOracle(symmetry.law_for(root, case)).to(a.device)
    group = tensor(data['group'])
    n = 4 if a.phase == 'witness' else 256
    banks = 4 if a.phase == 'formal' else 1
    levels = [1024, 2048] if a.phase == 'formal' else [16 if a.phase == 'witness' else 32]
    rows, resources = [], []
    started = time.monotonic()
    def progress(**kw):
        base.dump(out/'progress.json', dict(pid=os.getpid(), phase=a.phase, rows_completed=len(rows),
            rows_total=2*banks*len(levels), seconds=time.monotonic()-started, **kw))
    def resource(**kw):
        guard(True)
        if a.device == 'cuda':
            value = subprocess.check_output(['nvidia-smi', '-i', a.gpu_uuid,
                '--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits'], text=True).strip()
            resources.append(dict(unix_time=time.time(), value=value, **kw))
            base.dump(out/'resource-history.json', resources)
    for mode in ['original', 'projected']:
        net = model if mode == 'original' else symmetry.Projection(model, group)
        for bank in range(banks):
            source = tensor(data[f'source_{bank}'][:n])
            for steps in levels:
                guard(True)
                progress(action='density_sampling', mode=mode, bank=bank, steps=steps, completed_steps=0)
                def callback(completed, total):
                    guard()
                    progress(action='density_sampling', mode=mode, bank=bank, steps=steps, completed_steps=completed)
                if a.device == 'cuda':
                    torch.cuda.synchronize()
                before = time.monotonic()
                with derivative_backend(a.derivative_backend):
                    result = sample_density(net, oracle, source, steps, callback)
                if a.device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.monotonic()-before
                # Cross-check every complete bank against the actual author sampler.
                native = base.train.generated_samples(net, source, steps)
                assert torch.equal(native, result['generated'])
                filename = mode+f'-bank{bank}-heun{steps}.npz'
                np.savez_compressed(out/filename, **{k: v.cpu().numpy() for k, v in result.items()})
                row = dict(mode=mode, bank=bank, steps=steps, samples=n, native_endpoints_exact=True,
                    seconds=elapsed, file=filename, sha256=base.digest(out/filename),
                    nonpositive_steps=int(result['nonpositive_steps']),
                    max_chain_residual_per_dim=float(result['chain_residual'].abs().max())/source.shape[1],
                    max_logdet_disagreement_per_dim=float((result['log_ratio_divergence']-result['log_ratio_discrete_local']).abs().max())/source.shape[1])
                rows.append(row)
                base.dump(out/'rows.json', rows)
                resource(mode=mode, bank=bank, steps=steps)
                progress(action='row_complete', last=row)
    symmetry.ev.same_tensors(model, snapshot)
    guard(True)
    base.dump(out/'COMPLETE.json', dict(passed=True, phase=a.phase, witness=a.phase=='witness',
        formal=a.phase=='formal', benchmark=a.phase=='benchmark', id=case['id'], dataset=case['dataset'],
        derivative_backend=a.derivative_backend,
        family=case['family'], device=a.device, gpu_uuid=a.gpu_uuid, dtype='float64', parameters_unchanged=True,
        samples_per_bank=n, banks=banks, levels=levels, checkpoint_sha256=case['checkpoint_sha256'],
        portable_manifest_sha256=base.digest(root/'portable.json'), density_data_sha256=base.digest(data_root/'density-data.json'),
        rows=rows, source_sha256={p.name: base.digest(p) for p in [Path(__file__), HERE/'density_ops.py', HERE/'jacobian_backend.py']}))
    print(json.dumps(dict(passed=True, id=case['id'], phase=a.phase, rows=len(rows), seconds=time.monotonic()-started)))


if __name__ == '__main__':
    main()
