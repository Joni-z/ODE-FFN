"""Standard Inception FID in the existing metrics environment (no model imports)."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import importlib.metadata
import time
from pathlib import Path
import numpy as np
import scipy.linalg
import torch
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
from common import load, sha, uint8_images, verify, write


def fid(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    assert min(len(x), len(y)) > 1 and x.shape[1] == y.shape[1]
    assert np.isfinite(x).all() and np.isfinite(y).all()
    cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    eig, vectors = scipy.linalg.eigh(cx, check_finite=False)
    assert eig.min() > -1e-7*max(eig.max(), 1)
    square = (vectors*np.sqrt(np.maximum(eig, 0)))@vectors.T
    product = square@cy@square
    eig = scipy.linalg.eigvalsh((product+product.T)*.5, check_finite=False)
    assert eig.min() > -1e-7*max(eig.max(), 1)
    result = np.sum((x.mean(0)-y.mean(0))**2)+np.trace(cx)+np.trace(cy)-2*np.sqrt(np.maximum(eig, 0)).sum()
    assert result > -1e-5, result
    return max(0., float(result))


def extractor(paths, device='mps'):
    return FeatureExtractorInceptionV3('inception-v3-compat', ['2048'],
        feature_extractor_weights_path=paths['inception']).eval().requires_grad_(False).to(device)


@torch.no_grad()
def features(model, images, device='mps'):
    values = []
    for i in range(0, len(images), 32):
        values.append(model(torch.from_numpy(np.array(images[i:i+32])).to(device))[0].cpu().numpy())
        if i % 256 == 0:
            print(f'FEATURES {min(i+32,len(images))}/{len(images)}', flush=True)
    result = np.concatenate(values)
    assert result.shape == (len(images), 2048) and np.isfinite(result).all()
    return result


def fixture(root, output):
    _, paths = verify(root)
    for phase in load(root/'metrics-env-spec.json')['pip_phases']:
        for package in phase:
            name, expected = package.split('==')
            assert importlib.metadata.version(name) == expected
    rng = np.random.default_rng(0)
    rows = []
    for n, m, d in ((32, 40, 8), (12, 15, 20), (9, 11, 3)):
        x, y = rng.normal(size=(n, d)), rng.normal(size=(m, d))+.2
        a, b = (x-x.mean(0))/np.sqrt(n-1), (y-y.mean(0))/np.sqrt(m-1)
        expected = np.sum((x.mean(0)-y.mean(0))**2)+np.sum(a*a)+np.sum(b*b)-2*np.linalg.svd(a@b.T, compute_uv=False).sum()
        actual = fid(x, y)
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-7)
        np.testing.assert_allclose(fid(x, x), 0., atol=2e-6)
        rows.append(dict(n=n, m=m, d=d, error=abs(actual-expected)))
    pixels = rng.integers(0, 256, size=(8, 3, 32, 32), dtype=np.uint8)
    np.testing.assert_array_equal(uint8_images(pixels.astype(np.float32)/127.5-1), pixels)
    model = extractor(paths, 'cpu')
    cpu = features(model, pixels, 'cpu')
    mps = features(model.to('mps'), pixels)
    error = mps.astype(np.float64)-cpu.astype(np.float64)
    maximum, rms = float(np.abs(error).max()), float(np.sqrt(np.mean(error**2)))
    assert maximum < 1e-3 and rms < 1e-4, (maximum, rms)
    write(output, dict(status='PASS', FID_low_rank_comparison=rows, pixel_roundtrip=True,
        CPU_MPS_feature_max=maximum, CPU_MPS_feature_RMS=rms, inception_sha256=sha(paths['inception']),
        versions={n: importlib.metadata.version(n) for n in ('torch','torchvision','torch-fidelity','scipy','numpy')}))
    print('METRICS_WITNESS_PASS', flush=True)


def evaluate(root, folder):
    config, paths = verify(root)
    tick = time.monotonic()
    model = extractor(paths)
    cache = root/'reference-features.npy'
    if not cache.exists():
        images = np.load(Path(paths['data'])/'images.uint8.npy', mmap_mode='r')
        split = np.load(Path(paths['data'])/'split.npz')
        real = features(model, images[split['validation']])
        np.save(cache, real)
        write(root/'reference-features-binding.json', dict(sha256=sha(cache), count=5000,
              bindings_sha256=sha(root/'bindings.json'), split='validation'))
    assert sha(cache) == load(root/'reference-features-binding.json')['sha256']
    real = np.load(cache)
    assert real.shape == (config['fid_reference'], 2048)
    rows = []
    for arm in config['arms']:
        raw_path = folder/f'{arm}-ema.npy'
        raw = np.load(raw_path)
        generated = features(model, uint8_images(raw))
        feature_path = folder/f'{arm}-features.npy'
        np.save(feature_path, generated)
        row = dict(arm=arm, FID=fid(generated, real), generated=len(raw), reference=len(real),
                   EMA_decay=config['ema_decay'], generated_sha256=sha(raw_path), features_sha256=sha(feature_path))
        rows.append(row)
        print(row, flush=True)
    write(folder/'fid.json', dict(state='COMPLETE', rows=rows, seconds=time.monotonic()-tick,
          reference_features_sha256=sha(cache), inception_sha256=sha(paths['inception']),
          definition='standard 2048-dimensional empirical FID; float64 covariance PSD trace'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('fixture', 'evaluate'))
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--folder', type=Path)
    ap.add_argument('--output', type=Path)
    a = ap.parse_args()
    torch.set_num_threads(1)
    if a.action == 'fixture':
        assert not a.output.exists()
        fixture(a.root, a.output)
    else:
        evaluate(a.root, a.folder)
