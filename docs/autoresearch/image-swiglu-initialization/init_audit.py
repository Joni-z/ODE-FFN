"""Read-only full-bank and boundary audits for the initialization control."""
from __future__ import annotations
import json
import time
import numpy as np
import torch
import init_study as study


def helpers():
    study.parents()
    return study.module('swiglu_init_audit_helpers', study.COLD/'source/audit_training.py')


def begin(root, phase, kind):
    study.environment(root, kind)
    study.parents()
    study.verify(root)
    out = root/f'audit-{phase}'
    out.mkdir()
    hashes = {str(p): study.sha(p) for name in ('output', 'sampling', 'metric-output')
              for p in (root/name).rglob('*') if p.is_file()}
    return out, hashes, time.monotonic()


def finish(root, out, hashes, tick, result):
    for path, expected in hashes.items():
        assert study.sha(path) == expected, path
    study.verify(root)
    study.write(out/'result.json', dict(status='PASS', **result,
                seconds=time.monotonic()-tick, formal_files_unchanged=len(hashes)))
    print('SWIGLU_INIT_AUDIT_PASS', out.name, flush=True)


def audit_training(root):
    out, hashes, tick = begin(root, 'training', 'training')
    c, n, _ = study.parents()
    audit = helpers()
    folder = root/'output/runs'/study.ARM
    assert study.load(folder/'status.json')['state'] == 'COMPLETE'
    records = [json.loads(v) for v in (folder/'metrics.jsonl').read_text().splitlines()]
    streams = [json.loads(v) for v in (folder/'streams.jsonl').read_text().splitlines()]
    assert [v['step'] for v in records] == list(study.STEPS)
    assert [v['step'] for v in streams] == sorted(set(range(1000, 20001, 1000)) | {2500})
    images, ids = n.data_for(study.NATIVE)
    _, _, rng, digest = study.restored()
    recorded = {v['step']: v['digest'] for v in streams}
    for step in range(1, 20001):
        batch = n.draw(images, ids, rng)
        digest = n.fingerprint(digest, batch)
        if step in recorded:
            assert digest == recorded[step] == c.native_state(step)['input_stream_digest']
    rows = []
    initial = study.construct().state_dict()
    for step, record in zip(study.STEPS, records):
        post = torch.load(folder/f'step_{step:07d}.pt', weights_only=False, map_location='cpu')
        assert post['step'] == step
        assert post['input_stream_digest'] == c.native_state(step)['input_stream_digest']
        model, _, _, _ = study.restored(post)
        model.eval()
        bank = c.bank()
        prediction = n.predict(model, bank)
        with np.load(folder/f'validation_{step:07d}.npz') as saved:
            audit.same(prediction, saved['prediction'])
        error = prediction.astype(np.float64)-bank['target'].astype(np.float64)
        delta = bank['delta'].astype(np.float64)[:, None, None, None]
        np.testing.assert_allclose([record['weighted_mse'], record['clean_mse']],
            [np.mean((error/delta)**2), np.mean(error**2)], rtol=1e-12, atol=1e-12)
        for t in (.1, .3, .5, .7, .9):
            actual = n.predict(model, c.bank(t))
            with np.load(folder/f'panel_{step:07d}_{t:.1f}.npz') as saved:
                audit.same(actual, saved['prediction'])
        arrays = study.values(model)
        with np.load(folder/f'matrices-{step:07d}.npz') as saved:
            assert set(saved.files) == set(arrays)
            for name, value in arrays.items():
                audit.same(value, saved[name])
        history = study.load(folder/f'matrix-history-{step:07d}.json')
        assert len(history) == 9
        for row in history:
            value = arrays[row['layer']].astype(np.float64)
            np.testing.assert_allclose(np.linalg.svd(value, compute_uv=False),
                                       row['singular_values'], rtol=1e-12, atol=1e-12)
            assert float(np.linalg.norm(value)) == row['frobenius']
        branches = study.load(folder/f'branch-history-{step:07d}.json')
        for row, block in zip(branches, model.blocks):
            assert row['main_value_weight_norm'] == float(block.mlp.w12.weight.detach()[86:150].norm())
            assert row['extra_output_norm'] == float(block.mlp.w3.weight.detach()[:, 64:].norm())
        if step:
            pre = torch.load(folder/f'before_{step:07d}.pt', weights_only=False, map_location='cpu')
            assert pre['step'] == step-1
            active, opt, rng, digest = study.restored(pre)
            active.train()
            batch = n.draw(images, ids, rng)
            digest = n.fingerprint(digest, batch)
            audit.update(active, opt, batch)
            active.eval()
            n.predict(active, bank)
            n.equal(n.checkpoint(active, opt, rng, step, digest), post)
        else:
            n.equal(initial, post['model'])
        rows.append(dict(step=step, validation_rows_bitwise=1024, panels_bitwise=5,
                         matrices_bitwise=9, boundary_update_bitwise=bool(step)))
        study.write(out/'progress.json', dict(completed=len(rows), total=7))
        print(json.dumps(rows[-1]), flush=True)
    assert all(v['main_value_weight_norm'] > 0 and v['extra_output_norm'] > 0 for v in branches)
    finish(root, out, hashes, tick, dict(rows=rows, input_draws_reconstructed=20000,
        stream_boundaries_bitwise=21, optimizer_boundaries_bitwise=6,
        full_optimizer_trajectory_replayed=False, active_extra_and_value_routes=True))


def audit_sampling(root):
    out, hashes, tick = begin(root, 'sampling', 'training')
    assert study.load(root/'sampling/result.json')['status'] == 'PASS'
    audit = helpers()
    source = np.load(study.NATIVE/'banks/sampling-source.npy')
    rows = []
    for step in study.SAMPLES:
        model = study.model_at(root, step)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
        def progress(count):
            study.write(out/'progress.json', dict(step=step, images=count, completed=len(rows), total=3))
        actual = audit.manual(model, source, progress)
        audit.same(actual, np.load(root/'sampling'/f'{step:07d}.npy'))
        for name, value in model.state_dict().items():
            assert torch.equal(value, before[name])
        for name, value in model.named_buffers():
            assert torch.equal(value, buffers[name])
        rows.append(dict(step=step, independent_Heun_images_bitwise=len(actual),
                         parameters_buffers_unchanged=True))
        print(json.dumps(rows[-1]), flush=True)
    finish(root, out, hashes, tick, dict(rows=rows))


def audit_metrics(root):
    out, hashes, tick = begin(root, 'metrics', 'metrics')
    result = study.load(root/'metric-output/result.json')
    assert result['status'] == 'PASS'
    assert result['historical_rows'] == study.load(study.COLD/'metric-output/result.json')['rows']
    assert len(result['historical_rows']) == 32
    m, core = study.metrics_modules(root)
    audit = helpers()
    ex = m.extractor(study.NATIVE)
    reference = np.load(study.NATIVE/'metric-inputs/reference-features.npy')
    audit.same(m.features(ex, np.load(study.NATIVE/'banks/fid-reference.uint8.npy'),
                          progress='audit-reference'), reference)
    assert [v['step'] for v in result['rows']] == list(study.SAMPLES)
    rows = []
    for row in result['rows']:
        step = row['step']
        raw = np.load(root/'sampling'/f'{step:07d}.npy')
        images = np.rint((np.maximum(-1, np.minimum(1, raw.astype(np.float32)))
                         +np.float32(1))*np.float32(127.5)).astype(np.uint8)
        audit.same(images, core.uint8_images(raw))
        expected = np.load(root/'metric-output'/f'{step:07d}-features.npy')
        audit.same(m.features(ex, images, progress=f'audit-{step}'), expected)
        value = audit.fid(expected, reference)
        assert abs(value-row['fid']) < 1e-6
        rows.append(dict(step=step, generated_features_bitwise=len(expected),
                         independent_FID=value, absolute_error=abs(value-row['fid'])))
        study.write(out/'progress.json', dict(completed=len(rows), total=3))
    finish(root, out, hashes, tick, dict(rows=rows, reference_features_bitwise=2048,
        historical_rows_retained=32, independent_uint8_conversion=True))
