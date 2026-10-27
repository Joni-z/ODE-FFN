"""Native panel replay and independent dense-AD/reference-energy audit."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args();root = args.root.resolve();out = args.output.resolve()
out.mkdir(exist_ok=False);sys.path.insert(0, str(root/'source'))
import derivative_study as formal
from verify_derivatives import dense_field

started = time.monotonic();formal.verify_inputs(root)
settings, _, base, previous, native, core = formal.setup(root)
from conditional_oracle import best_field

assert formal.load(root/'output/status.json')['state']=='COMPLETE'
original = {str(p): formal.sha(p) for p in (root/'output').rglob('*') if p.is_file()}
with (root/'output/panels.csv').open() as stream:
    panels = list(csv.DictReader(stream))
assert len(panels)==90
basis = np.load(base/'basis.npy');checks = [];ad_max = 0.;field_max = 0.


def independent_metrics(model, reference, truth):
    result = {}
    for scope in ('full', 'tangent'):
        matrices = [np.asarray(v, dtype=np.float64) for v in (model, reference, truth)]
        if scope=='tangent':
            matrices = [basis.T@v@basis for v in matrices]
        fitted, optimum, full = matrices
        info, error = optimum-full, fitted-optimum
        norm = lambda v: np.einsum('bij,bij->b', v, v)
        symmetric = (error+np.swapaxes(error, 1, 2))/2
        skew = error-symmetric
        values = dict(total=norm(fitted-full), information_displacement=norm(info), residual=norm(error),
            cross=2*np.einsum('bij,bij->b', info, error), residual_symmetric=norm(symmetric),
            residual_skew=norm(skew))
        for name, matrix in [('model', fitted), ('reference', optimum), ('full_oracle', full)]:
            values[name+'_skew'] = norm((matrix-np.swapaxes(matrix, 1, 2))/2)
        if scope=='tangent':
            maximum = np.linalg.eigvalsh((full+np.swapaxes(full, 1, 2))/2)[:, -1]
            values.update(full_oracle_max_symmetric_eigenvalue=maximum,
                          full_oracle_expanding=(maximum>1e-8).astype(float))
        result.update({scope+'_'+key: value for key, value in values.items()})
    return result


for entry in formal.load(base/'settings.json')['laws']:
    law = entry['law'];banks = Path(entry['banks'])
    for arm in previous.ARMS:
        family, head = arm.rsplit('_', 1);folder = base/'output/runs'/law/arm
        state = torch.load(folder/'step_0005000.pt', weights_only=False)
        model = previous.build(base, law, arm);native.restore_model(model, state);model.eval()
        matrix = state['model']['raw.input.weight'].double().numpy()
        q, r = np.linalg.qr(matrix.T, mode='reduced');assert np.min(np.abs(np.diag(r)))>1e-10
        rows = q.T
        for clock in previous.TIMES:
            with np.load(banks/f'panel_{clock:.1f}.npz') as data:
                x, t, truth = data['x'], data['t'], data['oracle_jacobian']
            actual_clock = float(t[0, 0]);assert np.all(t==actual_clock)
            with np.load(folder/'diagnostics'/f'panel_{clock:.1f}.npz') as data:
                old_prediction, old_jacobian = data['prediction'], data['jacobian']
            prediction, jacobian = native.legacy.jacobian(model, torch.from_numpy(x), torch.from_numpy(t))
            np.testing.assert_array_equal(prediction.numpy(), old_prediction)
            np.testing.assert_array_equal(jacobian.numpy(), old_jacobian)
            np.testing.assert_array_equal(core.oracle(x, t, basis, entry['normal_variance'], jacobian=True)['oracle_jacobian'], truth)
            with np.load(root/'output/raw'/f'{law}-{arm}-{clock:.1f}.npz') as data:
                reference_field, reference_jacobian = data['reference_field'], data['reference_jacobian']
                saved_metrics = {k: data[k] for k in data.files if k not in ('input_rows', 'reference_field', 'reference_jacobian')}
            independent_field = best_field(x, t, rows, basis, entry['normal_variance'], head)
            field_difference = float(np.max(np.abs(independent_field-reference_field)))
            assert field_difference<=1e-8;field_max = max(field_max, field_difference)
            function = lambda value: dense_field(value, actual_clock, rows, basis, entry['normal_variance'], head)
            ad = torch.func.vmap(torch.func.jacrev(function))(torch.tensor(x[:2], dtype=torch.float64)).numpy()
            difference = float(np.max(np.abs(ad-reference_jacobian[:2])))
            assert difference<=1e-8;ad_max = max(ad_max, difference)
            metrics = independent_metrics(old_jacobian, reference_jacobian, truth)
            assert metrics.keys()==saved_metrics.keys()
            record = next(row for row in panels if row['law']==law and row['family']==family
                and row['head']==head and abs(float(row['time'])-clock)<1e-8)
            assert float(record['actual_time'])==actual_clock and int(record['count'])==64
            for key, value in metrics.items():
                np.testing.assert_allclose(value, saved_metrics[key], rtol=1e-6, atol=1e-8)
                np.testing.assert_allclose(np.mean(value), float(record[key]), rtol=1e-6, atol=1e-8)
            residual = old_prediction.astype(np.float64)-independent_field
            np.testing.assert_allclose([np.mean(residual**2), np.mean((residual@basis)**2)],
                [float(record['field_residual_per_ambient']), float(record['field_residual_per_tangent'])], atol=1e-8, rtol=1e-6)
            checks.append(dict(law=law, family=family, head=head, time=clock,
                conditional_field_max_difference=field_difference, conditional_J_AD_max_difference=difference,
                native_jacobians_bitwise=True))
            assert time.monotonic()-started<600
            print(dict(checked=len(checks), law=law, arm=arm, time=clock,
                       elapsed=time.monotonic()-started), flush=True)

assert len(checks)==90
with (root/'output/late-summary.csv').open() as stream:
    late = list(csv.DictReader(stream))
assert len(late)==6
for row in late:
    selected = [p for p in panels if p['law']==row['law'] and p['family']==row['family']
                and p['head']=='clean' and float(p['time']) in (.7, .9)]
    symmetric, skew = [np.mean([float(p['tangent_residual_'+key]) for p in selected]) for key in ('symmetric', 'skew')]
    share = symmetric/(symmetric+skew)
    np.testing.assert_allclose(share, float(row['symmetric_share']), atol=1e-12)
    label = 'symmetric_dominant' if share>=.8 else 'skew_dominant' if share<=.2 else 'mixed'
    assert row['classification']==label
formal.verify_inputs(root)
for name, digest in original.items():assert formal.sha(name)==digest, name
formal.table(out/'independent-checks.csv', checks)
formal.write(out/'audit.json', dict(status='PASS', native_panels_bitwise=90,
    native_jacobian_anchors=5760, conditional_dense_AD_anchors=180, summary_panels=90,
    conditional_field_max_difference=field_max, conditional_J_AD_max_difference=ad_max,
    formal_files_unchanged=len(original), elapsed_seconds=time.monotonic()-started,
    new_training=False, new_architecture_nominated=False))
for name in ('panels.csv', 'late-summary.csv', 'receipt.json', 'telemetry.csv'):
    shutil.copyfile(root/'output'/name, out/name)
shutil.copyfile(root/'preflight.json', out/'preflight.json')
receipt = formal.load(root/'output/receipt.json')
with (root/'results.tsv').open('w') as stream:
    stream.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    for row in panels:
        stream.write(f"{settings['registration_commit'][:7]}\t{float(row['tangent_residual']):.10g}\t{receipt['peak_sampled_rss_gib']:.3f}\tkeep\tconditional tangent derivative {row['law']} {row['family']} {row['head']} t={row['time']}; no nomination\n")
shutil.copyfile(root/'results.tsv', out/'results.tsv')
print(dict(status='AUDIT_PASS', panels=90, conditional_AD_anchors=180,
           elapsed=time.monotonic()-started), flush=True)
