"""Generate immutable, versioned target splits and fixed FM probes on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from synthetic.data import GaussianMixture,RotatedGaussianMixture,RotatedProductMixture,fm_batch
from synthetic.geometry import available_suite


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(config_path, output):
    config_path, output = Path(config_path), Path(output)
    cfg = json.loads(config_path.read_text())
    if output.exists():
        raise FileExistsError(f'{output} already exists; choose a new version directory')
    if cfg['source'] != 'standard_normal' or cfg['coupling'] != 'independent':
        raise ValueError('Only independent standard-normal source is implemented')
    distributions = available_suite()
    for name in cfg['datasets']:
        if name not in distributions:
            raise ValueError(f'unknown dataset {name}')
    for key in ('train_size', 'validation_size', 'test_size', 'probe_size'):
        if not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    output.mkdir(parents=True)
    manifest = {'config': cfg, 'numpy_version': np.__version__, 'files': {}, 'datasets': {},
                'generator_sha256': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    for name in cfg['datasets']:
        distribution = distributions[name]
        # A dataset's seed is independent of its position in the suite.
        identifier = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
        streams = np.random.SeedSequence([cfg['seed'], identifier]).spawn(4)
        folder = output / name
        folder.mkdir()
        for i, split in enumerate(('train', 'validation', 'test')):
            rng = np.random.default_rng(streams[i])
            x1 = distribution.sample(rng, cfg[f'{split}_size'])
            np.savez_compressed(folder / f'{split}.npz', x1=x1)
        rng = np.random.default_rng(streams[3])
        x1 = distribution.sample(rng, cfg['probe_size'])
        xt, t, target = fm_batch(x1, rng)
        # Source reconstructed only for reproducible evaluation, never used as training pairs.
        x0 = x1 - target
        probe = dict(x0=x0, x1=x1, xt=xt, t=t, pair_velocity=target)
        has_oracle = isinstance(distribution, (GaussianMixture,RotatedGaussianMixture,RotatedProductMixture))
        if has_oracle:
            probe['oracle_velocity'] = distribution.velocity(xt, t).astype(np.float32)
        np.savez_compressed(folder / 'probe.npz', **probe)
        mean, covariance = distribution.moments()
        specification = {'distribution': distribution.to_dict(), 'dim': distribution.dim,
                         'family': ('rotated_product_mixture' if isinstance(distribution,RotatedProductMixture)
                             else 'rotated_gaussian_mixture' if isinstance(distribution,RotatedGaussianMixture)
                             else 'gaussian_mixture' if has_oracle else 'geometric'),
                         'has_velocity_oracle': has_oracle,
                         'mean': mean.tolist(), 'covariance': covariance.tolist(),
                         'source': cfg['source'], 'coupling': cfg['coupling']}
        (folder / 'distribution.json').write_text(json.dumps(specification, indent=2) + '\n')
        manifest['datasets'][name] = specification
        print(f'Prepared {name}: train={cfg["train_size"]}, validation={cfg["validation_size"]}, test={cfg["test_size"]}', flush=True)
    for path in sorted(output.rglob('*')):
        if path.is_file():
            manifest['files'][path.relative_to(output).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/synthetic/data_v1.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(args.config, args.output)
