"""Independent native replay, stream reconstruction and endpoint algebra checks."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time as timer

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
ROOT = args.root.resolve()
sys.path.insert(0, str(ROOT / 'source'))
import cubic_study as study
from cubic_core import SplitCore, explicit_split, input_jacobian

torch.set_num_threads(1)
started = timer.monotonic()
ready = study.load(ROOT / 'ready.json')
for category in ('bindings', 'prior_documents'):
    for path, digest in ready[category].items(): assert study.sha(path) == digest, path
assert study.load(ROOT / 'output/status.json')['state'] == 'COMPLETE'
cohort, stats_records, derivative_records = [], [], []
counts = dict(checkpoints=0, endpoint_panels=0, sampling_replays=0,
              independent_matrix_spectra=0, replayed_stream_batches=0)
maxima = dict(explicit_forward=0., jacobian=0., adjoint=0.)


def metric_close(actual: float, expected: float) -> None:
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-9)


for dataset in study.legacy.DATASETS:
    bank = study.targets(dataset)
    dim = bank.shape[1]
    banks = study.OLD / 'output/banks' / dataset
    # Reconstruct the native training stream directly, without the study's draw helper.
    generator = torch.Generator().manual_seed(100000)
    digest = study.EMPTY
    streams = {0: digest}
    for step in range(1, 5001):
        target = bank[torch.randint(len(bank), (512,), generator=generator)]
        source = torch.randn(target.shape, generator=generator)
        clock = torch.rand((512, 1), generator=generator)
        digest = study.legacy.fingerprint_batch(digest, target, source, clock)
        if step in study.SNAPSHOTS: streams[step] = digest
    counts['replayed_stream_batches'] += 5000
    for family in (*study.NEW, *study.REFERENCES):
        reused = family in study.REFERENCES
        folder = study.reference(dataset, family) if reused else ROOT / 'output/runs' / dataset / family
        assert study.load(folder / 'status.json')['state'] == 'COMPLETE'
        if reused:
            reference = study.load(ROOT / 'output/reused' / f'{dataset}--{family}.json')
            for name, expected in reference['files_sha256'].items():
                assert study.sha(folder / name) == expected, name
        result = study.load(folder / 'result.json')
        metrics = [json.loads(line) for line in (folder / 'metrics.jsonl').read_text().splitlines()]
        for step in study.SNAPSHOTS:
            model = study.build(dim, family)
            state = torch.load(folder / f'step_{step:07d}.pt', map_location='cpu', weights_only=False)
            study.restore_model(model, state)
            assert state['step'] == step and state['input_stream_digest'] == streams[step]
            actual = study.legacy.field_evaluation(model, banks)
            saved = next(row for row in metrics if row['step'] == step)
            for key in ('oracle_mse', 'pair_mse', 'prediction_rms'): metric_close(actual[key], saved[key])
            if not reused:
                with np.load(banks / 'field.npz') as data:
                    x, t = data['x'], data['t']
                with torch.no_grad():
                    prediction = np.concatenate([model(torch.from_numpy(x[k:k+1024]),
                        torch.from_numpy(t[k:k+1024])).numpy() for k in range(0, len(x), 1024)])
                with np.load(folder / f'field_{step:07d}.npz') as data:
                    np.testing.assert_array_equal(prediction, data['prediction'])
                statistics = study.load(folder / f'statistics_{step:07d}.json')
                for row in statistics:
                    if 'P_singular_values' not in row: continue
                    p = model.raw.blocks[row['position']].ffn.proj.weight.detach().double().numpy()
                    sv = np.linalg.svd(p, compute_uv=False)
                    eigenvalues = np.linalg.eigvalsh(.5 * (p + p.T))
                    np.testing.assert_allclose(sv, row['P_singular_values'], rtol=1e-10, atol=1e-11)
                    np.testing.assert_allclose(eigenvalues, row['P_symmetric_eigenvalues'], rtol=1e-10, atol=1e-11)
                    counts['independent_matrix_spectra'] += 1
                stats_records.append(dict(dataset=dataset, family=family, step=step, statistics=statistics))
            counts['checkpoints'] += 1
        # model is now the native endpoint, with actual nonpersistent time buffers restored.
        actual = study.legacy.field_evaluation(model, banks)
        for key in ('oracle_mse', 'pair_mse', 'prediction_rms'): metric_close(actual[key], result[key])
        for clock in study.TIMES:
            with np.load(banks / f'panel_{clock:.1f}.npz') as data:
                x, t, oracle, truth = (data[k] for k in ('x', 't', 'oracle', 'oracle_jacobian'))
            value, jac = study.legacy.jacobian(model, torch.from_numpy(x), torch.from_numpy(t))
            with np.load(folder / 'diagnostics' / f'panel_{clock:.1f}.npz') as data:
                np.testing.assert_array_equal(value.numpy(), data['prediction'])
                np.testing.assert_array_equal(jac.numpy(), data['jacobian'])
            row = next(r for r in result['panels'] if abs(r['time'] - clock) < 1e-6)
            metric_close(float(np.mean((value.numpy() - oracle)**2)), row['field_mse'])
            metric_close(float(np.mean(np.sum((jac.numpy() - truth)**2, axis=(1, 2)))), row['jacobian_error'])
            counts['endpoint_panels'] += 1
        with np.load(banks / 'sampling.npz') as data:
            source, target, directions = [torch.from_numpy(data[k]) for k in ('source', 'target', 'directions')]
        with torch.no_grad(): samples = study.legacy.generated_samples(model, source, 32)
        with np.load(folder / 'diagnostics/samples.npz') as data:
            np.testing.assert_array_equal(samples.numpy(), data['generated'])
        assert torch.isfinite(samples).all()
        metric_close(study.legacy.sliced_w2(samples, target, directions), result['sw2'])
        counts['sampling_replays'] += 1
        if family in ('cubic_split', 'silu_split'):
            visits, hooks = {}, []
            for j, block in enumerate(model.raw.blocks):
                def capture(module, inputs, outputs, j=j): visits[j] = inputs[0].detach().clone()
                hooks.append(block.ffn.register_forward_hook(capture))
            try:
                with np.load(banks / 'probe.npz') as data, torch.no_grad():
                    model(torch.from_numpy(data['x'][:5]), torch.from_numpy(data['t'][:5]))
            finally:
                for hook in hooks: hook.remove()
            for j, block in enumerate(model.raw.blocks):
                local = copy.deepcopy(block.ffn).double()
                x = visits[j].double().requires_grad_(True)
                native, explicit = local(x), explicit_split(local, x)
                torch.testing.assert_close(native, explicit, rtol=1e-10, atol=1e-11)
                jac = torch.stack([torch.autograd.grad(native[:, k].sum(), x, retain_graph=True)[0]
                                   for k in range(64)], dim=1)
                manual = input_jacobian(local, x)
                torch.testing.assert_close(jac, manual, rtol=1e-9, atol=1e-10)
                gn = torch.autograd.grad(native.square().mean(), tuple(local.parameters()))
                ge = torch.autograd.grad(explicit.square().mean(), tuple(local.parameters()))
                for a, b in zip(gn, ge): torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-10)
                values = dict(explicit_forward=float((native-explicit).detach().abs().max()),
                              jacobian=float((jac-manual).detach().abs().max()),
                              adjoint=max(float((a-b).abs().max()) for a, b in zip(gn, ge)))
                for key, value in values.items(): maxima[key] = max(maxima[key], value)
                derivative_records.append(dict(dataset=dataset, family=family, position=j, **values))
        assert result['input_stream_digest'] == streams[5000]
        cohort.append(dict(dataset=dataset, family=family, reused=reused, folder=str(folder),
                           costs=study.compute_budget(model)))
    print(dict(dataset=dataset, audited_cells=7, elapsed_seconds=timer.monotonic()-started), flush=True)

for category in ('bindings', 'prior_documents'):
    for path, digest in ready[category].items(): assert study.sha(path) == digest, path
study.write(ROOT / 'audit-cohort.json', cohort)
study.write(ROOT / 'audit-statistics.json', stats_records)
study.write(ROOT / 'audit-derivatives.json', derivative_records)
receipt = dict(status='PASS', new_cells=15, reused_cells=6, **counts, maxima=maxima,
               derivative_comparisons=len(derivative_records), input_bindings=len(ready['bindings']),
               preserved_prior_files=len(ready['prior_documents']), ready_sha256=study.sha(ROOT/'ready.json'),
               cohort_sha256=study.sha(ROOT/'audit-cohort.json'), statistics_sha256=study.sha(ROOT/'audit-statistics.json'),
               elapsed_seconds=timer.monotonic()-started)
study.write(ROOT / 'audit.json', receipt)
print(json.dumps(receipt), flush=True)
