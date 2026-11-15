"""Refine only numerically unresolved frozen cells, with unchanged tolerances."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from density_ops import HERE, base, torch, symmetry, MixtureOracle, sample_density


def measures(coarse, fine, dim):
    values = dict(nonpositive_steps=int(fine['nonpositive_steps']),
        chain=float(fine['chain_residual'].abs().max())/dim,
        logdet=float((fine['log_ratio_divergence']-fine['log_ratio_discrete_local']).abs().max())/dim,
        ratio_refinement=max(float((fine[k]-coarse[k]).abs().max())/dim for k in ['log_ratio_source', 'log_ratio_divergence']),
        endpoint_refinement=float(((fine['generated']-coarse['generated']).norm(dim=1)/(1+fine['generated'].norm(dim=1))).max()))
    flags = [values['nonpositive_steps'] == 0, values['chain'] <= .001, values['logdet'] <= .001,
             values['ratio_refinement'] <= .002, values['endpoint_refinement'] <= .001]
    return values, all(flags)


def main(argv=None):
    p = argparse.ArgumentParser()
    for name in ['root', 'data', 'pilot', 'verification', 'output']:
        p.add_argument('--'+name, required=True)
    p.add_argument('--index', type=int, required=True)
    p.add_argument('--witness', action='store_true')
    a = p.parse_args(argv)
    root, data_root, pilot, verification, out = map(Path, [a.root, a.data, a.pilot, a.verification, a.output])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    base.verify_sources()
    manifest, meta = [json.loads(p.read_text()) for p in [root/'portable.json', data_root/'density-data.json']]
    case = manifest['cases'][a.index]
    checks = json.loads((verification/'COMPLETE.json').read_text())
    assert checks['passed'] and checks['cases'] == 8
    prior = next(v for v in json.loads((verification/'verified-cases.json').read_text()) if v['id'] == case['id'])
    complete = pilot/'cases'/case['id']/'COMPLETE.json'
    assert base.digest(complete) == prior['completion_sha256']
    old = json.loads(complete.read_text())
    assert old['density_data_sha256'] == base.digest(data_root/'density-data.json')
    assert old['checkpoint_sha256'] == case['checkpoint_sha256']
    selected = [v['mode'] for v in prior['qualifications'] if not v['qualified']]
    assert selected or all(v['qualified'] for v in prior['qualifications'])
    asset = meta['datasets'][case['dataset']]
    assert base.digest(data_root/asset['file']) == asset['sha256']
    with np.load(data_root/asset['file']) as z:
        source = torch.from_numpy(z['source_0'][:8].copy())
        group = torch.from_numpy(z['group'].copy())
    model = symmetry.load(root, case, 'cpu').double().requires_grad_(False)
    snapshot = symmetry.ev.tensor_snapshot(model)
    oracle = MixtureOracle(symmetry.law_for(root, case))
    rows, decisions = [], []
    started = time.monotonic()
    for mode in selected:
        net = model if mode == 'original' else symmetry.Projection(model, group)
        old_row = next(v for v in old['rows'] if v['mode'] == mode and v['steps'] == 512)
        file = complete.parent/old_row['file']
        assert base.digest(file) == old_row['sha256']
        with np.load(file) as z:
            coarse = {k: torch.from_numpy(z[k].copy()) for k in z.files}
        previous_steps = 512
        # A witness exercises the real entry with a small bounded level, no acceptance claim.
        levels = [16] if a.witness else [1024, 2048]
        accepted = False
        for steps in levels:
            def progress(completed, total):
                base.dump(out/'progress.json', dict(pid=os.getpid(), id=case['id'], mode=mode, steps=steps,
                    completed_steps=completed, total_steps=total, completed_rows=len(rows), seconds=time.monotonic()-started))
            result = sample_density(net, oracle, source, steps, progress)
            native = base.train.generated_samples(net, source, steps)
            assert torch.equal(native, result['generated'])
            filename = mode+f'-heun{steps}.npz'
            np.savez_compressed(out/filename, **{k: v.cpu().numpy() for k, v in result.items()})
            metrics, accepted = measures(coarse, result, source.shape[1])
            rows.append(dict(mode=mode, steps=steps, previous_steps=previous_steps, file=filename,
                             sha256=base.digest(out/filename), native_endpoints_exact=True,
                             metrics=metrics, qualified=accepted if not a.witness else False))
            if accepted and not a.witness:
                break
            coarse, previous_steps = result, steps
        decisions.append(dict(mode=mode, final_steps=steps, qualified=accepted if not a.witness else False))
    symmetry.ev.same_tensors(model, snapshot)
    base.dump(out/'COMPLETE.json', dict(passed=True, witness=a.witness, refinement=True, id=case['id'],
        selected_modes=selected, rows=rows, decisions=decisions, parameters_unchanged=True,
        density_data_sha256=base.digest(data_root/'density-data.json'), checkpoint_sha256=case['checkpoint_sha256'],
        pilot_completion_sha256=prior['completion_sha256'], verification_sha256=base.digest(verification/'COMPLETE.json'),
        source_sha256={f.name: base.digest(f) for f in [Path(__file__), HERE/'density_ops.py']}))
    print(json.dumps(dict(passed=True, refinement=True, witness=a.witness, id=case['id'], decisions=decisions)))


if __name__ == '__main__':
    main()
