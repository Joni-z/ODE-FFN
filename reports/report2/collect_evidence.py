"""CPU-only evidence snapshot for Report2; never modifies or samples the live run."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import datetime
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import scipy.linalg
import torch

ARMS = ('isru', 'swiglu_matched', 'swiglu_wide')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def table(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def rows(path):
    with Path(path).open() as handle:
        return list(csv.DictReader(handle))


def complete_log(path, cap):
    text = Path(path).read_text()
    lines = text.splitlines()
    if not text.endswith('\n'):
        lines = lines[:-1]
    return [r for r in map(json.loads, lines) if r['step'] <= cap]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runtime', type=Path, required=True)
    ap.add_argument('--snapshot', type=Path, required=True)
    args = ap.parse_args()
    root, snap = args.runtime.resolve(), args.snapshot.resolve()
    repo = Path(__file__).resolve().parents[2]
    report = Path(__file__).resolve().parent
    assert not snap.exists(), 'use a new snapshot; retain earlier evidence'
    snap.mkdir(parents=True)
    data = report/'data'; data.mkdir(exist_ok=True)
    source_hash = sha(Path(__file__))
    bindings = load(root/'bindings.json')
    for name, digest in bindings.items():
        path = Path(name) if Path(name).is_absolute() else root/name
        assert sha(path) == digest, str(path)
    config = load(root/'config.json')
    assert config['study'] == 'oct29-isru-cifar10-v1'
    assert config['ema_decay'] == .9999 and config['sampling_steps'] == 50
    assert config['arms'] == list(ARMS)
    write(snap/'config.json', config)
    observed = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    status = load(root/'output/status.json')
    write(snap/'status.json', status)
    write(snap/'gpu.json', load(root/'output/apple-gpu-monitor-latest.json'))
    validation = complete_log(root/'output/validation.jsonl', 40000)
    training = complete_log(root/'output/train.jsonl', 40000)
    assert [r['step'] for r in validation] == list(range(500, 40001, 500))
    assert training[0]['step'] == 1 and training[-1]['step'] == 40000
    val_rows, train_rows = [], []
    for row in validation:
        for arm in ARMS:
            val_rows.append(dict(step=row['step'], arm=arm, **row['arms'][arm]))
    for row in training:
        for arm in ARMS:
            train_rows.append(dict(step=row['step'], arm=arm, **row['arms'][arm],
                                  cohort_seconds=row['cohort_seconds'], stream_digest=row['stream_digest']))
    table(data/'cifar-validation.csv', val_rows)
    table(data/'cifar-training.csv', train_rows)
    write(snap/'validation.json', validation)
    write(snap/'training.json', training)
    # Retain all five target laws and both ordinary budgets, never only winners.
    mapping = [('Gaussian2D', 'docs/research-cubic-split-2026-10-26/comparison.csv','dataset','gaussian_aniso_2d'),
               ('Ring8', 'docs/research-cubic-split-2026-10-26/comparison.csv','dataset','gmm_ring8_2d'),
               ('Product16', 'docs/research-cubic-split-2026-10-26/comparison.csv','dataset','rotated_product_bimodal_16d'),
               ('Dependent signs', 'docs/research-joint-dependence-2026-10-26/comparison.csv','law','sign_code'),
               ('Half mixture', 'docs/autoresearch/cycle02-full-support/results/comparison.csv','law','half_mixture')]
    synthetic, input_hashes = [], {}
    for label, relative, key, law in mapping:
        path = repo/relative; input_hashes[relative] = sha(path)
        for row in rows(path):
            if row[key] == law and row['family'] in ('free_isru','swiglu64','swiglu160'):
                synthetic.append(dict(law=label, arm=row['family'], oracle_mse=float(row['oracle_mse']),
                                      sw2=float(row['sw2']), source=relative))
    assert len(synthetic) == 15
    table(data/'synthetic.csv', synthetic)
    frame_path = repo/'docs/autoresearch/cycle11-conditional-frame/results/comparison.csv'
    shutil.copy2(frame_path, data/'conditional-frame.csv')
    input_hashes[str(frame_path.relative_to(repo))] = sha(frame_path)
    diagnostics = []
    for step in (0, 2000, 5000, 10000, 25000):
        path = root/'output'/('cold-matrices.json' if step == 0 else f'eval-{step:06d}/matrix-diagnostics.json')
        if step:
            assert load(path.parent/'complete.json')['state'] == 'COMPLETE'
        for row in load(path):
            row = dict(row); row.pop('shape')
            diagnostics.append(dict(step=step, **row))
        input_hashes[str(path)] = sha(path)
    table(data/'cifar-matrices.csv', diagnostics)
    # Verify matrix statistics against actual CPU checkpoint tensors, including
    # all ordinary controls. This does not run an optimizer or use MPS.
    torch.set_num_threads(1)
    matrix_errors = []
    for step, name in ((0, 'cold.pt'), (25000, 'checkpoint-025000.pt')):
        checkpoint = root/'output'/name
        value = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert value['step'] == step
        assert value['config_sha256'] == sha(root/'config.json')
        assert value['bindings_sha256'] == sha(root/'bindings.json')
        maximum = 0.
        for record in (r for r in diagnostics if r['step'] == step):
            tensor = value['models'][record['arm']][f"blocks.{record['block']}.mlp.{record['matrix']}"]
            singular = torch.linalg.svdvals(tensor.double())
            probability = singular/singular.sum()
            actual = dict(singular_min=float(singular.min()), singular_max=float(singular.max()),
                          singular_mean=float(singular.mean()), singular_std=float(singular.std()),
                          frobenius=float(torch.linalg.norm(tensor.double())),
                          effective_rank=float(torch.exp(-(probability*probability.log()).sum())))
            for key, number in actual.items():
                difference = abs(number-record[key]); maximum = max(maximum, difference)
                assert np.isclose(number, record[key], atol=1e-8, rtol=1e-9), (step, key)
        matrix_errors.append(dict(step=step, max_abs_difference=maximum, checkpoint_sha256=sha(checkpoint)))
        del value
        gc.collect()
    # Independent low-rank FID identity on the saved features. All nine rows
    # are hash-checked; recompute every completed checkpoint, not just25k.
    reference_path = root/'reference-features.npy'
    assert sha(reference_path) == load(root/'reference-features-binding.json')['sha256']
    reference = np.asarray(np.load(reference_path), np.float64)
    assert reference.shape == (5000, 2048)
    real_mean = reference.mean(0)
    real_factor = (reference-real_mean)/np.sqrt(len(reference)-1)
    fid_rows, numerical = [], []
    for step in (5000, 10000, 25000):
        folder = root/'output'/f'eval-{step:06d}'
        saved = load(folder/'fid.json')
        assert saved['state'] == 'COMPLETE' and saved['reference_features_sha256'] == sha(reference_path)
        for record in saved['rows']:
            arm = record['arm']
            assert record['generated'] == 1024 and record['reference'] == 5000
            raw_path, features_path = folder/f'{arm}-ema.npy', folder/f'{arm}-features.npy'
            assert sha(raw_path) == record['generated_sha256']
            assert sha(features_path) == record['features_sha256']
            generated = np.asarray(np.load(features_path), np.float64)
            assert generated.shape == (1024, 2048) and np.isfinite(generated).all()
            mean = generated.mean(0)
            factor = (generated-mean)/np.sqrt(len(generated)-1)
            tick = time.monotonic()
            singular = scipy.linalg.svdvals(factor@real_factor.T, check_finite=False)
            independent = float(np.sum((mean-real_mean)**2)+np.sum(factor*factor)+np.sum(real_factor*real_factor)-2*singular.sum())
            assert np.isclose(independent, record['FID'], atol=2e-4, rtol=1e-6), (step, arm, independent, record['FID'])
            numerical.append(dict(step=step, arm=arm, registered_FID=record['FID'],
                  independent_low_rank_FID=independent, abs_difference=abs(independent-record['FID']), seconds=time.monotonic()-tick))
            fid_rows.append(dict(step=step, arm=arm, FID=record['FID'], generated=1024, reference=5000,
                  EMA_decay=.9999, initial_EMA_mass=.9999**step))
            print(json.dumps(numerical[-1]), flush=True)
            # Preserve exact inspected grid pixels for future figure rebuilds.
            for tag in ('raw','ema'):
                grid_path = folder/f'{arm}-{tag}-grid.png'
                input_hashes[str(grid_path)] = sha(grid_path)
                if step == 25000:
                    target = data/f'{arm}-{tag}-25k.png'
                    shutil.copy2(grid_path, target)
            input_hashes[str(folder/'fid.json')] = sha(folder/'fid.json')
        write(snap/f'fid-{step}.json', saved)
    table(data/'cifar-fid.csv', fid_rows)
    write(data/'FID-verification.json', numerical)
    write(data/'matrix-verification.json', matrix_errors)
    summary = dict(observed_at=observed, runtime=str(root), progress_at_start=status,
        source_commit=load(root/'registration.json')['source_commit'], bindings_sha256=sha(root/'bindings.json'),
        collector_source_sha256=source_hash, snapshot=str(snap), config=config,
        validation_last_step=40000, completed_FID_last_step=25000, final_50k_FID_available=False,
        source_bindings_verified=True, completed_feature_and_generated_hashes_verified=True,
        all_nine_FIDs_recomputed=True, all_cold_and_25k_FFN_matrix_summaries_verified=True,
        training_modified=False, new_training=False, new_sampling=False,
        checkpoints_selected='all completed registered FID checkpoints, fixed validation cutoff40k',
        input_hashes=input_hashes)
    write(data/'snapshot.json', summary)
    write(snap/'result.json', summary)
    write(data/'file-hashes.json', {str(p.relative_to(report)): sha(p) for p in sorted(data.iterdir())
                                   if p.is_file() and p.name != 'file-hashes.json'})
    print('REPORT2_EVIDENCE_PASS', flush=True)


if __name__ == '__main__':
    main()
