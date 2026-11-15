"""Bind new evaluation-only streams and CPU derivative references; no training."""
import argparse
import json
from pathlib import Path
import numpy as np
from density_ops import HERE, base, torch, symmetry, exact_jacobian


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root, out = Path(a.root), Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    (out/'reference').mkdir()
    torch.set_num_threads(1)
    base.verify_sources()
    manifest = json.loads((root/'portable.json').read_text())
    assets, references = {}, []
    for index, dataset in enumerate(['gmm_ring8_2d', 'rotated_product_bimodal_16d']):
        case = next(c for c in manifest['cases'] if c['dataset'] == dataset)
        law = symmetry.law_for(root, case)
        rng = np.random.default_rng(2026112300+index)
        arrays = {}
        for bank in range(4):
            arrays[f'source_{bank}'] = rng.standard_normal((256, law.dim)).astype(np.float64)
            arrays[f'target_{bank}'] = law.sample(rng, 256).astype(np.float64)
        x, t, banks, clocks = [], [], [], []
        for bank in range(4):
            for clock, time in enumerate([.05, .25, .5, .75, .95]):
                source, target = rng.standard_normal((64, law.dim)), law.sample(rng, 64).astype(float)
                x.append((1-time)*source+time*target)
                t.append(np.full((64, 1), time))
                banks.extend([bank]*64)
                clocks.extend([clock]*64)
        arrays.update(field_x=np.concatenate(x), field_t=np.concatenate(t),
                      field_banks=np.array(banks), field_clocks=np.array(clocks))
        arrays['field_oracle'] = law.velocity(arrays['field_x'], arrays['field_t'])
        old = symmetry.arrays(root, case)
        arrays.update(group=old['group'], directions=old['directions'])
        filename = dataset+'.npz'
        np.savez_compressed(out/filename, **arrays)
        assets[dataset] = dict(file=filename, sha256=base.digest(out/filename), stream=2026112300+index)
    for case in manifest['cases']:
        with np.load(out/assets[case['dataset']]['file']) as z:
            data = {k: z[k].copy() for k in z.files}
        model = symmetry.load(root, case, 'cpu').double().requires_grad_(False)
        x, t = [torch.from_numpy(data[k][:4]) for k in ['field_x', 'field_t']]
        source = torch.from_numpy(data['source_0'][:8])
        reference = {}
        for mode in ['original', 'projected']:
            net = model if mode == 'original' else symmetry.Projection(model, torch.from_numpy(data['group']))
            value, jac = exact_jacobian(net, x, t)
            reference[mode+'_field'] = value.cpu().numpy()
            reference[mode+'_jacobian'] = jac.cpu().numpy()
            reference[mode+'_native128'] = base.train.generated_samples(net, source, 128).cpu().numpy()
        filename = 'reference/'+case['id']+'.npz'
        np.savez_compressed(out/filename, **reference)
        references.append(dict(id=case['id'], checkpoint_sha256=case['checkpoint_sha256'],
                               file=filename, sha256=base.digest(out/filename)))
    base.dump(out/'density-data.json', dict(passed=True, train_seed=0, evaluation_stream=2026112300,
        independent_banks=4, points_per_bank=256, pilot_points=8, field_points=1280,
        datasets=assets, references=references, portable_manifest_sha256=base.digest(root/'portable.json'),
        source_sha256=base.digest(__file__)))
    print(json.dumps(dict(passed=True, cases=len(references), output=str(out))))


if __name__ == '__main__':
    main()
