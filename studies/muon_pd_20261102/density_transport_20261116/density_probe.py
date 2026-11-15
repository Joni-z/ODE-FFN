"""Frozen density-source measurements and bounded numerical qualification."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
from density_ops import HERE, base, torch, symmetry, MixtureOracle, exact_jacobian, sample_density


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--index', required=True, type=int)
    p.add_argument('--output', required=True)
    p.add_argument('--phase', choices=['witness', 'pilot'], required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--gpu-uuid')
    p.add_argument('--readiness')
    a = p.parse_args(argv)
    root, data_root, out = map(Path, [a.root, a.data, a.output])
    meta = json.loads((data_root/'density-data.json').read_text())
    assert meta['passed'] and meta['portable_manifest_sha256'] == base.digest(root/'portable.json')
    cases = json.loads((root/'portable.json').read_text())['cases']
    case = cases[a.index]
    assert meta['references'][a.index]['id'] == case['id']
    if a.phase == 'pilot':
        assert a.readiness
        ready = json.loads(Path(a.readiness).read_text())
        assert ready['passed'] and len(ready['models']) == 8 and len(ready['entry_witnesses']) == 2
        assert ready['density_data_sha256'] == base.digest(data_root/'density-data.json')
        assert ready['device'] == a.device
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    def guard():
        if a.device == 'cuda':
            symmetry.exclusive_gpu(a.gpu_uuid)
    guard()
    if a.device == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = False
    asset = meta['datasets'][case['dataset']]
    assert base.digest(data_root/asset['file']) == asset['sha256']
    with np.load(data_root/asset['file']) as z:
        data = {k: z[k].copy() for k in z.files}
    tensor = lambda v: torch.tensor(v, dtype=torch.double, device=a.device)
    model = symmetry.load(root, case, a.device).double().requires_grad_(False)
    snapshot = symmetry.ev.tensor_snapshot(model)
    oracle = MixtureOracle(symmetry.law_for(root, case)).to(a.device)
    group = tensor(data['group'])
    n = 4 if a.phase == 'witness' else 8
    levels = [16, 32] if a.phase == 'witness' else [128, 256, 512]
    points = 16 if a.phase == 'witness' else 1280
    rows, fields, resources = [], [], []
    started = time.monotonic()
    def progress(**extra):
        base.dump(out/'progress.json', dict(pid=os.getpid(), phase=a.phase, completed=len(rows), total=2*len(levels),
                                          seconds=time.monotonic()-started, **extra))
    for mode in ['original', 'projected']:
        net = model if mode == 'original' else symmetry.Projection(model, group)
        guard()
        progress(action='field', mode=mode)
        x, t = tensor(data['field_x'][:points]), tensor(data['field_t'][:points])
        value, jac = exact_jacobian(net, x, t)
        with torch.no_grad():
            true = oracle.evaluate(x, t)
            error = value-true['velocity']
            div_error = jac.diagonal(dim1=1, dim2=2).sum(1)-true['divergence']
            alignment = (error*true['score']).sum(1)
            arrays = dict(velocity=value.cpu().numpy(), divergence_error=div_error.cpu().numpy(),
                alignment=alignment.cpu().numpy(), density_source=(div_error+alignment).cpu().numpy(),
                field_mse=error.square().mean(1).cpu().numpy())
        filename = mode+'-field.npz'
        np.savez_compressed(out/filename, **arrays)
        fields.append(dict(mode=mode, file=filename, sha256=base.digest(out/filename), points=points))
        for steps in levels:
            guard()
            before = time.monotonic()
            callback = lambda completed, total: (guard(), progress(action='density_sampling', mode=mode,
                                                              heun_steps=steps, heun_completed=completed))
            result = sample_density(net, oracle, tensor(data['source_0'][:n]), steps, callback)
            native = base.train.generated_samples(net, tensor(data['source_0'][:n]), steps)
            assert torch.equal(native, result['generated'])
            filename = mode+f'-heun{steps}.npz'
            np.savez_compressed(out/filename, **{k: v.cpu().numpy() for k, v in result.items()})
            rows.append(dict(mode=mode, steps=steps, base_points=n, file=filename, sha256=base.digest(out/filename),
                             seconds=time.monotonic()-before, native_endpoints_exact=True,
                             nonpositive_steps=int(result['nonpositive_steps']),
                             max_chain_residual_per_dim=float(result['chain_residual'].abs().max())/case_dim(data),
                             max_logdet_disagreement_per_dim=float((result['log_ratio_divergence']-result['log_ratio_discrete_local']).abs().max())/case_dim(data)))
            progress(action='resolution_complete', mode=mode, last=rows[-1])
            guard()
    symmetry.ev.same_tensors(model, snapshot)
    base.dump(out/'COMPLETE.json', dict(passed=True, witness=a.phase=='witness', pilot=a.phase=='pilot', id=case['id'],
        dataset=case['dataset'], family=case['family'], device=a.device, gpu_uuid=a.gpu_uuid, parameters_unchanged=True,
        checkpoint_sha256=case['checkpoint_sha256'], portable_manifest_sha256=base.digest(root/'portable.json'),
        density_data_sha256=base.digest(data_root/'density-data.json'), fields=fields, rows=rows,
        source_sha256={f.name: base.digest(f) for f in [Path(__file__), HERE/'density_ops.py']}))
    print(json.dumps(dict(passed=True, id=case['id'], phase=a.phase, output=str(out))))


def case_dim(data):
    return data['source_0'].shape[1]


if __name__ == '__main__':
    main()
