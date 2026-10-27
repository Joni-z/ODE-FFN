"""Independent QR observations and Woodbury conditional fields, no fitting."""
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
args = parser.parse_args()
root, out = args.root.resolve(), args.output.resolve()
out.mkdir(exist_ok=False)
sys.path.insert(0, str(root/'source'))
import conditional_study as formal
from conditional_oracle import best_field

started = time.monotonic()
formal.verify_inputs(root)
settings, _, base, previous, native, core = formal.setup(root)
assert formal.load(root/'output/status.json')['state']=='COMPLETE'
original = {str(p): formal.sha(p) for p in (root/'output').rglob('*') if p.is_file()}
with (root/'output/decomposition.csv').open() as stream:
    summaries = list(csv.DictReader(stream))
assert len(summaries)==1296
basis = np.load(base/'basis.npy')
angles = 2*np.pi*np.arange(8)/8
means = 3*np.column_stack((np.cos(angles), np.sin(angles)))
records = [];native_replays = 0;maximum = 0.


def woodbury(x, t, rows, variance, head):
    x = x.astype(np.float64);t = t.astype(np.float64)
    a = 1-t;delta = np.maximum(a, .05)
    c = rows@basis;gram = c.T@c;u = x@rows.T
    d0 = a*a+t*t*variance;coefficient = t*t*(.04-variance)
    matrix = d0[:, :, None]*np.eye(2)+coefficient[:, :, None]*gram[None]
    alpha = np.linalg.solve(matrix, (u@c)[:, :, None])[:, :, 0]
    gain = np.linalg.solve(matrix, np.broadcast_to(gram, matrix.shape))
    logits = t*(alpha@means.T)-.5*t*t*np.einsum('ki,bij,kj->bk', means, gain, means)
    weights = np.exp(logits-logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    mean = weights@means
    residual = u-t*(mean@c.T)
    reduced = alpha-t*np.einsum('bij,bj->bi', gain, mean)
    corrected = residual-coefficient*(reduced@c.T)
    clean = mean@basis.T+t*((variance/d0)*(corrected@rows)+(.04-variance)*(reduced@basis.T))
    if head=='direct':
        return a/delta*clean-(a*a/(delta*d0))*(corrected@rows)
    return (clean-x)/delta


def independent_components(prediction, optimum, truth):
    first, second, total = optimum-truth, prediction.astype(np.float64)-optimum, prediction.astype(np.float64)-truth
    results = []
    for scope in ('full', 'tangent', 'normal'):
        vectors = []
        for value in (first, second, total):
            plane = value@basis
            vectors.append(value if scope=='full' else plane if scope=='tangent' else value-plane@basis.T)
        i, f, v = vectors
        dimension = 126 if scope=='normal' else v.shape[1]
        results.append(np.column_stack((np.sum(v*v, axis=1), np.sum(i*i, axis=1),
                                        np.sum(f*f, axis=1), 2*np.sum(i*f, axis=1)))/dimension)
    return np.stack(results, axis=1)


for entry in formal.load(base/'settings.json')['laws']:
    law = entry['law']
    with np.load(Path(entry['banks'])/'field.npz') as data:
        x, t, truth = [data[k] for k in ('x', 't', 'oracle')]
    np.testing.assert_array_equal(core.oracle(x, t, basis, entry['normal_variance'])['oracle'], truth)
    for arm in previous.ARMS:
        family, head = arm.rsplit('_', 1)
        folder = base/'output/runs'/law/arm
        for step in previous.STEPS:
            state = torch.load(folder/f'step_{step:07d}.pt', weights_only=False)
            matrix = state['model']['raw.input.weight'].double().numpy()
            q, triangular = np.linalg.qr(matrix.T, mode='reduced')
            assert np.min(np.abs(np.diag(triangular)))>1e-10
            rows = q.T
            with np.load(root/'output/raw'/f'{law}-{arm}-{step:07d}.npz') as data:
                old, old_rows = data['components'], data['input_rows']
            np.testing.assert_allclose(rows.T@rows, old_rows.T@old_rows, atol=1e-9, rtol=0)
            np.testing.assert_allclose(matrix@(np.eye(128)-rows.T@rows), 0, atol=1e-9, rtol=0)
            conditioned = woodbury(x, t, rows, entry['normal_variance'], head)
            primary = best_field(x, t, old_rows, basis, entry['normal_variance'], head)
            difference = float(np.max(np.abs(conditioned-primary)))
            assert difference<=1e-6, difference
            maximum = max(maximum, difference)
            with np.load(folder/f'field_{step:07d}.npz') as data:
                prediction = data['prediction']
            values = independent_components(prediction, conditioned, truth)
            np.testing.assert_allclose(values, old, atol=1e-8, rtol=1e-6)
            selected = [r for r in summaries if r['law']==law and r['family']==family
                        and r['head']==head and int(r['step'])==step]
            assert len(selected)==18
            for row in selected:
                scope = formal.SCOPES.index(row['scope'])
                if row['time_bin']=='all':
                    mask = np.ones(len(t), dtype=bool)
                else:
                    lower, upper = map(float, row['time_bin'].split('-'))
                    mask = (t[:, 0]>=lower) & (t[:, 0]<upper)
                assert int(mask.sum())==int(row['count'])
                np.testing.assert_allclose(values[mask, scope].mean(axis=0),
                    [float(row[k]) for k in formal.METRICS], atol=1e-8, rtol=1e-6)
            if step==5000:
                model = previous.build(base, law, arm)
                native.restore_model(model, state);model.eval()
                with torch.no_grad():
                    replay = np.concatenate([model(torch.from_numpy(x[i:i+1024]),
                        torch.from_numpy(t[i:i+1024])).numpy() for i in range(0, len(x), 1024)])
                np.testing.assert_array_equal(replay, prediction);native_replays += 1
            records.append(dict(law=law, family=family, head=head, step=step,
                conditional_field_max_difference=difference,
                component_max_difference=float(np.max(np.abs(values-old)))))
            print(dict(checked=len(records), law=law, arm=arm, step=step,
                       difference=difference, elapsed=time.monotonic()-started), flush=True)

assert len(records)==72 and native_replays==18
formal.verify_inputs(root)
for path, digest in original.items():
    assert formal.sha(Path(path))==digest, path
formal.table(out/'independent-checks.csv', records)
formal.write(out/'audit.json', dict(status='PASS', checkpoints=72, native_endpoint_replays=native_replays,
    summary_rows=1296, conditional_field_max_difference=maximum,
    formal_files_unchanged=len(original), elapsed_seconds=time.monotonic()-started,
    scope='QR row spaces, 2D Woodbury conditional fields, independently projected error components and native endpoint replays'))
for name in ('decomposition.csv', 'decision.json', 'receipt.json', 'telemetry.csv'):
    shutil.copyfile(root/'output'/name, out/name)
shutil.copyfile(root/'preflight.json', out/'preflight.json')
receipt = formal.load(root/'output/receipt.json')
with (root/'results.tsv').open('w') as stream:
    stream.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    for row in summaries:
        if row['time_bin']=='all' and row['scope']=='full':
            stream.write(f"{settings['registration_commit'][:7]}\t{float(row['total']):.10g}\t{receipt['peak_sampled_rss_gib']:.3f}\tkeep\tinformation attribution {row['law']} {row['family']} {row['head']} step {row['step']}; no nomination\n")
shutil.copyfile(root/'results.tsv', out/'results.tsv')
print(dict(status='AUDIT_PASS', checkpoints=72, native_replays=18,
           elapsed=time.monotonic()-started), flush=True)
