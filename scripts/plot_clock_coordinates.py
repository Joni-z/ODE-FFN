"""Plot the frozen function-preserving clock-coordinate comparison."""
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
    arms = ['vector_k1', 'vector_reparam', 'vector_tau', 'swiglu_budget', 'swiglu']
    labels = ['Original clock', 'Paired coordinates', 'Unpaired temperature', 'SwiGLU small', 'SwiGLU full']
    colors = ['#2678b2', '#db8321', '#8155a3', '#638c43', '#222222']
    datasets = ['gaussian_aniso_2d', 'gmm_ring8_2d', 'rotated_product_bimodal_16d']
    titles = ['Gaussian 2D', 'Ring8', 'Rotated product 16D']
    rows, inputs = [], []
    for study in sorted(studies.iterdir()):
        if not (study / 'audit/audit.json').exists():
            continue
        audit = json.loads((study / 'audit/audit.json').read_text())
        assert audit['state'] == 'PASS'
        assert audit['source_verified'] and audit['oracles_recomputed']
        for row in audit['rows']:
            rows.append(row)
            cfg_path = study / 'configs' / f'{row["label"]}.json'
            cfg = json.loads(cfg_path.read_text())
            trajectory = study / 'runs' / cfg['experiment_id'] / row['dataset'] / row['model'] / 'seed000/metrics.jsonl'
            inputs.extend([study / 'audit/audit.json', trajectory])
    assert len(rows) == 15
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3), layout='constrained')
    for ax, dataset, title in zip(axes, datasets, titles):
        selected = {r['label']: r for r in rows if r['dataset'] == dataset}
        denominator = min(selected[a]['validation_oracle_mse'] for a in arms[-2:])
        values = [selected[a]['validation_oracle_mse'] / denominator for a in arms]
        ax.bar(np.arange(5), values, color=colors, width=.7)
        ax.axhline(1, color='#555555', lw=.8, ls='--')
        ax.set_xticks(np.arange(5), ['Orig.', 'Paired', 'Temp.', 'Swi S', 'Swi F'], rotation=25)
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.18)
        for i, value in enumerate(values):
            ax.text(i, value + .035 * max(values), f'{value:.2f}', ha='center', fontsize=8)
    axes[0].set_ylabel('Oracle MSE / better SwiGLU\n(lower is better)')
    for suffix in ('pdf', 'png'):
        fig.savefig(output / f'clock-coordinates-endpoints.{suffix}', dpi=180)
    plt.close(fig)

    curves = {}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), layout='constrained')
    for ax, dataset, title in zip(axes, datasets[1:], titles[1:]):
        for arm, label, color in zip(arms, labels, colors):
            row = next(r for r in rows if r['dataset'] == dataset and r['label'] == arm)
            study = next(s for s in studies.iterdir() if (s / 'configs' / f'{arm}.json').exists()
                         and dataset in json.loads((s / 'provenance.json').read_text())['plan']['datasets'])
            cfg = json.loads((study / 'configs' / f'{arm}.json').read_text())
            path = study / 'runs' / cfg['experiment_id'] / dataset / row['model'] / 'seed000/metrics.jsonl'
            curve = [json.loads(line) for line in path.read_text().splitlines()]
            curves[(dataset, arm)] = curve
            ax.plot([r['step'] for r in curve], [r['validation_oracle_mse'] for r in curve],
                    label=label, color=color, lw=1.4)
        ax.set_yscale('log')
        ax.set_title(title)
        ax.set_xlabel('Training update')
    axes[0].set_ylabel('Validation oracle MSE')
    axes[1].legend(frameon=False, fontsize=8)
    for suffix in ('pdf', 'png'):
        fig.savefig(output / f'clock-coordinates-learning.{suffix}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.0), layout='constrained')
    for i, dataset in enumerate(datasets[1:]):
        for arm, label, color in zip(arms[:3], labels[:3], colors[:3]):
            curve = curves[(dataset, arm)]
            step = [r['step'] for r in curve]
            p = [np.mean([v['raw']['sigma_max'] for v in r['ode_layers']]) for r in curve]
            gate = [np.mean([v['activation']['gate_output_weight_over_tau_frobenius'] for v in r['ode_layers']]) for r in curve]
            axes[i, 0].plot(step, p, label=label, color=color)
            axes[i, 1].plot(step, gate, label=label, color=color)
        axes[i, 0].set_title(titles[i+1] + ' / projection norm')
        axes[i, 1].set_title(titles[i+1] + ' / invariant clock coordinate')
        axes[i, 0].set_ylabel('Mean P spectral norm')
        axes[i, 1].set_ylabel('Mean ||W_last / tau||_F')
        for ax in axes[i]:
            ax.set_xlabel('Training update')
    axes[0, 1].legend(frameon=False, fontsize=8)
    for suffix in ('pdf', 'png'):
        fig.savefig(output / f'clock-coordinates-operators.{suffix}', dpi=180)
    plt.close(fig)

    (output.parent / 'plot-provenance.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(inputs))},
        'scope': 'Descriptive fixed-protocol comparisons. No error bars or inferential tests.',
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('studies', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    plot(args.studies, args.output)
