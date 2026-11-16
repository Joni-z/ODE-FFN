"""Finite differences of the full unchanged native sampler, not its field."""
import argparse
import json
from pathlib import Path
import numpy as np
from density_ops import base, torch, symmetry


def main():
    p = argparse.ArgumentParser()
    for name in ['root', 'data', 'entries', 'output']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    root, data_root, entries, out = map(Path, [a.root, a.data, a.entries, a.output])
    assert not out.exists()
    torch.set_num_threads(1)
    base.verify_sources()
    cases = json.loads((root/'portable.json').read_text())['cases']
    meta = json.loads((data_root/'density-data.json').read_text())
    rows = []
    assert len(cases) >= 2
    for index in [0, len(cases)-1]:
        case = cases[index]
        with np.load(data_root/meta['datasets'][case['dataset']]['file']) as z:
            source = torch.from_numpy(z['source_0'][:1].copy())
            group = torch.from_numpy(z['group'].copy())
        model = symmetry.load(root, case, 'cpu').double().requires_grad_(False)
        dim = source.shape[1]
        done = json.loads((entries/('entry-'+str(index))/'COMPLETE.json').read_text())
        assert done['passed'] and done['witness'] and done['checkpoint_sha256'] == case['checkpoint_sha256']
        for row in done['rows']:
            field = model if row['mode'] == 'original' else symmetry.Projection(model, group)
            file = entries/('entry-'+str(index))/row['file']
            assert base.digest(file) == row['sha256']
            with np.load(file) as z:
                reference = float(z['integral_local_logdet'][0])
            values = []
            for h in [1e-4, 1e-5]:
                shift = h*torch.eye(dim, dtype=torch.double)
                points = torch.cat((source+shift, source-shift))
                value = base.train.generated_samples(field, points, row['steps'])
                jac = ((value[:dim]-value[dim:])/(2*h)).T
                sign, logdet = torch.linalg.slogdet(jac)
                assert float(sign) > 0
                values.append(dict(step=h, logdet=float(logdet), absolute_error=abs(float(logdet)-reference),
                                   condition=float(torch.linalg.cond(jac))))
            assert values[-1]['absolute_error'] < 5e-5
            rows.append(dict(id=case['id'], mode=row['mode'], steps=row['steps'],
                             observer_logdet=reference, full_native_map_finite_differences=values))
    result = dict(passed=True, actual_native_map_checks=len(rows), rows=rows, source_sha256=base.digest(__file__))
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, actual_native_map_checks=len(rows), max_error=max(r['full_native_map_finite_differences'][-1]['absolute_error'] for r in rows))))


if __name__ == '__main__':
    main()
