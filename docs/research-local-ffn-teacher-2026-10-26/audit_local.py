"""Post-training native replay and independent local-derivative audit."""
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT/'source'))
import local_study as study
from local_math import value_jacobian
from teacher_support import (OLD, BANKS, DATASET, TEACHERS, STUDENTS, TIMES,
                             SNAPSHOTS, load, write, sha, teacher, signature, assert_equal)


def risk(a: np.ndarray, b: np.ndarray) -> dict:
    delta = np.asarray(a, dtype=np.float64)-np.asarray(b, dtype=np.float64)
    mse = np.vdot(delta.ravel(), delta.ravel()).real/delta.size
    energy = np.vdot(np.asarray(b, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()).real/b.size
    return dict(mse=float(mse), relative_mse=float(mse/energy), target_energy=float(energy))


def equal_metrics(first: dict, second: dict) -> None:
    for key in first:
        np.testing.assert_allclose(first[key], second[key], rtol=2e-12, atol=2e-14, err_msg=key)


def audit() -> None:
    torch.set_num_threads(1)
    started = time.monotonic()
    out, cache = ROOT/'output', ROOT/'output/cache'
    assert load(out/'status.json')['state'] == 'COMPLETE'
    ready = load(ROOT/'ready.json')
    for path, digest in {**ready['bindings'], **ready['prior_documents']}.items():
        assert sha(path) == digest, path
    metadata = load(cache/'metadata.json')
    for name, digest in metadata['files_sha256'].items():
        assert sha(cache/name) == digest, name
    models = {name: teacher(name)[0] for name in TEACHERS}
    assert metadata['teachers'] == {name: signature(m) for name, m in models.items()}
    # Reconstruct common prefix states explicitly rather than using the pre-hooks.
    old_cache = study.PREVIOUS/'output/cache'
    input_sets = {'train': (np.load(old_cache/'x.npy', mmap_mode='r').reshape(-1, 16)[:100000],
                            np.load(old_cache/'t.npy', mmap_mode='r').reshape(-1, 1)[:100000])}
    for split, name in [('validation', 'field.npz')]+[(f'panel-{t:.1f}', f'panel_{t:.1f}.npz') for t in TIMES]:
        with np.load(BANKS/name) as data:
            input_sets[split] = (data['x'], data['t'])
    raw = models['swiglu128'].raw
    captured_rows = label_rows = 0
    for split, (x, t) in input_sets.items():
        zz = [np.load(cache/study.file_name(split, j), mmap_mode='r') for j in range(3)]
        yy = {(j, name): np.load(cache/study.file_name(split, j, name), mmap_mode='r')
              for j in range(3) for name in TEACHERS}
        for start in range(0, len(x), 1024):
            xx = torch.from_numpy(np.array(x[start:start+1024], copy=True))
            tt = torch.from_numpy(np.array(t[start:start+1024], copy=True))
            with torch.no_grad():
                context, h = raw.time(tt), raw.input(xx)
                for j, block in enumerate(raw.blocks):
                    scale, shift = block.condition(context).chunk(2, -1)
                    z = block.norm(h)*(1+scale)+shift
                    assert np.array_equal(z.numpy(), zz[j][start:start+len(xx)])
                    captured_rows += len(xx)
                    for name, model in models.items():
                        prediction = model.raw.blocks[j].ffn(z)
                        assert np.array_equal(prediction.numpy(), yy[j, name][start:start+len(xx)])
                        label_rows += len(xx)
                    h = h+block.residual_scale*block.ffn(z)
        print(dict(stage='cache_replay', split=split, rows=len(x)), flush=True)
    for name, model in models.items():
        for j in range(3):
            z, _ = study.arrays(cache, 'validation', j, name)
            _, native = study.native_jacobian(model.raw.blocks[j].ffn, torch.from_numpy(np.array(z[:128], copy=True)))
            assert np.array_equal(native.numpy(), np.load(cache/f'jac-{name}-{j}.npy'))
    # Rebuild all consumed streams once per teacher/position, not per student.
    expected = {}
    for j in range(3):
        for name in TEACHERS:
            z, y = study.arrays(cache, 'train', j, name)
            rng = torch.Generator().manual_seed(0)
            digests = dict.fromkeys(('indices', 'input', 'label'), study.EMPTY)
            states = {0: digests.copy()}
            for step in range(1, 5001):
                study.consumed(z, y, rng, digests)
                if step in SNAPSHOTS:
                    states[step] = digests.copy()
            expected[j, name] = states
    print(dict(stage='all_streams_replayed', streams=6, batches=30000), flush=True)
    forward_count = jacobian_count = p_count = analytic_count = 0
    analytic_max = dict(forward=0., jacobian=0.)
    for folder in sorted((out/'runs').iterdir()):
        result = load(folder/'result.json')
        name, family, j = result['teacher'], result['student'], result['position']
        assert load(folder/'status.json')['state'] == 'COMPLETE'
        provenance = load(folder/'provenance.json')
        assert provenance['threads'] == 1 and provenance['seed'] == 0 and provenance['device'] == 'cpu'
        module = study.cold(family, j)
        assert signature(module) == provenance['initial_signature']
        assert study.budget(module) == provenance['costs']
        native_initial = copy.deepcopy(module.state_dict())
        metrics = [json.loads(line) for line in (folder/'metrics.jsonl').read_text().splitlines()]
        assert len(metrics) == 11
        for step in SNAPSHOTS:
            state = torch.load(folder/f'step_{step:07d}.pt', map_location='cpu', weights_only=False)
            assert state['step'] == step
            assert state['consumed_digests'] == expected[j, name][step]
            if step == 0:
                assert_equal(state['model'], native_initial)
            module.load_state_dict(state['model'])
            assert_equal(dict(module.named_buffers()), state['runtime_buffers'])
            for value in state['optimizer']['state'].values():
                assert float(value['step']) == step
            z, target = study.arrays(cache, 'validation', j, name)
            native = study.predict(module, z)
            assert np.array_equal(native, np.load(folder/f'validation_{step:07d}.npy'))
            measure = risk(native, target)
            equal_metrics(measure, next(row for row in metrics if row['step'] == step))
            if step == 5000:
                equal_metrics(measure, result)
            forward_count += 1
            recorded = load(folder/f'spectrum_{step:07d}.json')
            for layer_name, values in recorded.items():
                p = dict(module.named_modules())[layer_name].proj.weight.detach().double().numpy()
                np.testing.assert_allclose(np.linalg.svd(p, compute_uv=False), values['singular_values'], rtol=2e-11, atol=2e-12)
                np.testing.assert_allclose(np.max(np.abs(np.linalg.eigvals(p))), values['spectral_radius'], rtol=2e-11, atol=2e-12)
                np.testing.assert_allclose(np.linalg.eigvalsh((p+p.T)/2).min(), values['symmetric_minimum'], rtol=2e-11, atol=2e-12)
                p_count += 1
        assert signature(module) == result['final_signature']
        assert result['consumed_digests'] == expected[j, name][5000]
        for t in TIMES:
            z, target = study.arrays(cache, f'panel-{t:.1f}', j, name)
            native = study.predict(module, z)
            assert np.array_equal(native, np.load(folder/f'panel_{t:.1f}.npy'))
            equal_metrics(risk(native, target), next(row for row in result['panels'] if row['time'] == t))
        z, _ = study.arrays(cache, 'validation', j, name)
        xx = torch.from_numpy(np.array(z[:128], copy=True))
        _, native_j = study.native_jacobian(module, xx)
        assert np.array_equal(native_j.numpy(), np.load(folder/'jacobian.npy'))
        truth = np.load(cache/f'jac-{name}-{j}.npy').astype(np.float64)
        je = float(np.mean(np.sum((native_j.numpy().astype(np.float64)-truth)**2, axis=(1, 2))))
        np.testing.assert_allclose(je, result['jacobian_error'], rtol=2e-12, atol=2e-14)
        np.testing.assert_allclose(je/float(np.mean(np.sum(truth**2, axis=(1, 2)))), result['relative_jacobian_error'], rtol=2e-12, atol=2e-14)
        jacobian_count += 1
        # Independent closed Jacobians, not autodifferentiation through a copied forward.
        for subject, subject_family in [(module, family), (models[name].raw.blocks[j].ffn, name)]:
            double = copy.deepcopy(subject).double()
            value, jac = study.native_jacobian(double, xx.double())
            explicit, analytic = value_jacobian(double, xx.double(), subject_family)
            torch.testing.assert_close(value, explicit, rtol=1e-10, atol=1e-11)
            torch.testing.assert_close(jac, analytic, rtol=1e-10, atol=1e-11)
            analytic_max['forward'] = max(analytic_max['forward'], float((value-explicit).detach().abs().max()))
            analytic_max['jacobian'] = max(analytic_max['jacobian'], float((jac-analytic).detach().abs().max()))
            analytic_count += 1
        print(dict(stage='cell_replayed', cell=folder.name), flush=True)
    assert len(list((out/'runs').iterdir())) == 18
    assert metadata['teachers'] == {name: signature(m) for name, m in models.items()}
    for path, digest in {**ready['bindings'], **ready['prior_documents']}.items():
        assert sha(path) == digest, path
    receipt = dict(status='PASS', cells=18, checkpoint_forwards=forward_count,
                   common_prefix_rows=captured_rows, label_rows=label_rows,
                   full_consumed_streams=6, replayed_stream_batches=30000,
                   endpoint_time_panels=90, endpoint_jacobians=jacobian_count,
                   independently_checked_P_records=p_count, analytic_panels=analytic_count,
                   analytic_max_abs=analytic_max, cache_files=len(metadata['files_sha256']),
                   input_bindings=len(ready['bindings']), preserved_prior_files=len(ready['prior_documents']),
                   elapsed_seconds=time.monotonic()-started, pid=__import__('os').getpid(),
                   final_rss_bytes=psutil.Process().memory_info().rss,
                   ready_sha256=sha(ROOT/'ready.json'), audit_source_sha256=sha(__file__))
    write(ROOT/'audit.json', receipt)
    print(receipt, flush=True)


if __name__ == '__main__':
    audit()
