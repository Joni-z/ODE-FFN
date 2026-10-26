"""Validate prepared arrays and hashes, then export a small data preview."""
import argparse
import json
from pathlib import Path
import numpy as np
from synthetic.data import GaussianMixture,RotatedGaussianMixture,RotatedProductMixture
from synthetic.geometry import GeometricTarget
from synthetic.prepare import sha256


def validate(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    for filename, expected in manifest['files'].items():
        path = root / filename
        if sha256(path) != expected['sha256']:
            raise ValueError(f'Checksum mismatch: {filename}')
    report = {}
    for name, spec in manifest['datasets'].items():
        cls={'gaussian_mixture':GaussianMixture,'rotated_gaussian_mixture':RotatedGaussianMixture,
            'rotated_product_mixture':RotatedProductMixture,
            'geometric':GeometricTarget}[spec['family']]
        dist=cls(**spec['distribution'])
        splits = {}
        for split in ('train','validation','test'):
            with np.load(root/name/f'{split}.npz', allow_pickle=False) as data:
                x = data['x1']
                assert data.files == ['x1'], 'Targets must not expose class labels'
                assert x.shape == (manifest['config'][f'{split}_size'], dist.dim)
                assert x.dtype == np.float32 and np.isfinite(x).all()
                mean, cov = dist.moments()
                z = float(np.max(np.abs(x.mean(0)-mean)/np.sqrt(np.diag(cov)/len(x))))
                assert z < 7, f'{name}/{split}: sample mean is unexpectedly far from population mean'
                splits[split] = {'shape': list(x.shape), 'max_mean_standard_errors': z}
        with np.load(root/name/'probe.npz', allow_pickle=False) as probe:
            assert all(np.isfinite(probe[k]).all() for k in probe.files)
            np.testing.assert_allclose(probe['xt'], (1-probe['t'])*probe['x0']+probe['t']*probe['x1'], atol=2e-6)
            if spec['has_velocity_oracle']:
                np.testing.assert_allclose(probe['oracle_velocity'], dist.velocity(probe['xt'], probe['t']), atol=2e-6, rtol=2e-6)
            else:
                assert 'oracle_velocity' not in probe.files
        report[name] = splits
    (root/'validation_report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return manifest


def preview(root, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    fig, axes = plt.subplots(2, 4, figsize=(16,8), constrained_layout=True)
    axes = axes.flatten()
    for unused in axes[len(manifest['datasets']):]:
        unused.axis('off')
    for ax, (name,spec) in zip(axes, manifest['datasets'].items()):
        with np.load(root/name/'test.npz') as data:
            points = data['x1'][:3000]
        if spec['dim'] == 2:
            ax.scatter(points[:,0], points[:,1], s=2, alpha=.25)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('x[0]'); ax.set_ylabel('x[1]')
        else:
            ax.plot(np.var(points,axis=0), marker='o', label='sample')
            ax.plot(np.diag(spec['covariance']), linestyle='--', label='population')
            ax.set_yscale('log'); ax.set_xlabel('dimension'); ax.set_ylabel('variance'); ax.legend()
        ax.set_title(name, fontsize=10)
    fig.savefig(root/'preview.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root')
    p.add_argument('--preview', action='store_true')
    args = p.parse_args()
    manifest = validate(args.root)
    if args.preview:
        preview(args.root, manifest)
