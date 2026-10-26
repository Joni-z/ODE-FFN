"""Plot all frozen covariance-field arms against both practical references."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(studies: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    arms = ['vector_k1', 'cov_tied', 'cov_untied', 'cov_linear', 'swiglu_budget', 'swiglu']
    labels = ['Original', 'Tied covariance', 'Untied', 'Static tangent', 'SwiGLU small', 'SwiGLU full']
    ticks = ['Orig.', 'Tied', 'Untied', 'Linear', 'Swi S', 'Swi F']
    colors = ['#2678b2', '#db8321', '#8155a3', '#999999', '#638c43', '#222222']
    datasets = ['gaussian_aniso_2d', 'gmm_ring8_2d', 'rotated_product_bimodal_16d']
    titles = ['Gaussian 2D', 'Ring8', 'Rotated product 16D']
    rows, locations, hashes = [], {}, {}
    for study in sorted(studies.iterdir()):
        path = study / 'audit/audit.json'
        audit = json.loads(path.read_text())
        assert audit['state'] == 'PASS'
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in audit['rows']:
            rows.append(row)
            locations[row['dataset']] = study
    assert len(rows) == 18
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), layout='constrained')
    for ax, dataset, title in zip(axes, datasets, titles):
        selected = {r['label']: r for r in rows if r['dataset'] == dataset}
        denominator = min(selected[a]['validation_oracle_mse'] for a in arms[-2:])
        values = [selected[a]['validation_oracle_mse'] / denominator for a in arms]
        ax.bar(np.arange(6), values, color=colors, width=.7)
        ax.axhline(1, color='#555555', lw=.8, ls='--')
        ax.set_xticks(np.arange(6), ticks, rotation=30)
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.18)
        for i, value in enumerate(values):
            ax.text(i, value + .035 * max(values), f'{value:.2f}', ha='center', fontsize=8)
    axes[0].set_ylabel('Oracle MSE / better SwiGLU\n(lower is better)')
    for suffix in ('pdf', 'png'):
        fig.savefig(output / f'covariance-endpoints.{suffix}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), layout='constrained')
    for ax, dataset, title in zip(axes, datasets[1:], titles[1:]):
        study = locations[dataset]
        for arm, label, color in zip(arms, labels, colors):
            row = next(r for r in rows if r['dataset'] == dataset and r['label'] == arm)
            cfg_path = study / 'configs' / f'{arm}.json'
            cfg = json.loads(cfg_path.read_text())
            path = study / 'runs' / cfg['experiment_id'] / dataset / row['model'] / 'seed000/metrics.jsonl'
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            curve = [json.loads(line) for line in path.read_text().splitlines()]
            assert curve[-1]['step'] == 5000
            ax.plot([r['step'] for r in curve], [r['validation_oracle_mse'] for r in curve],
                    label=label, color=color, lw=1.4)
        ax.set_yscale('log')
        ax.set_title(title)
        ax.set_xlabel('Training update')
    axes[0].set_ylabel('Validation oracle MSE')
    axes[1].legend(frameon=False, fontsize=8)
    for suffix in ('pdf', 'png'):
        fig.savefig(output / f'covariance-learning.{suffix}', dpi=180)
    plt.close(fig)
    (output.parent / 'plot-provenance.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), inputs_sha256=hashes,
        scope='All registered endpoints and learning curves; descriptive, no inferential error bars.'), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('studies', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    plot(args.studies, args.output)
