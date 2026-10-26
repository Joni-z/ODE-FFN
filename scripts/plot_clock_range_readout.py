"""Plot the frozen clock-range screen and the separate response-subspace audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(studies: Path, response: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    arms = ['vector_k1', 'vector_wide', 'vector_affine', 'swiglu_budget', 'swiglu']
    labels = ['Original clock', 'Wider positive', 'Affine clock', 'SwiGLU small', 'SwiGLU full']
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
        ax.set_xticks(np.arange(5), ['Orig.', 'Wide', 'Affine', 'Swi S', 'Swi F'], rotation=25)
        ax.set_title(title)
        ax.set_ylim(0, max(values) * 1.18)
        for i, value in enumerate(values):
            ax.text(i, value + .035 * max(values), f'{value:.2f}', ha='center', fontsize=8)
    axes[0].set_ylabel('Oracle MSE / better SwiGLU\n(lower is better)')
    for suffix in ('pdf', 'png'):
        fig.savefig(output / f'clock-range-endpoints.{suffix}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), layout='constrained')
    for ax, dataset, title in zip(axes, datasets[1:], titles[1:]):
        for arm, label, color in zip(arms, labels, colors):
            row = next(r for r in rows if r['dataset'] == dataset and r['label'] == arm)
            study = next(s for s in studies.iterdir() if (s / 'configs' / f'{arm}.json').exists()
                         and dataset in json.loads((s / 'provenance.json').read_text())['plan']['datasets'])
            cfg = json.loads((study / 'configs' / f'{arm}.json').read_text())
            path = study / 'runs' / cfg['experiment_id'] / dataset / row['model'] / 'seed000/metrics.jsonl'
            curve = [json.loads(line) for line in path.read_text().splitlines()]
            ax.plot([r['step'] for r in curve], [r['validation_oracle_mse'] for r in curve],
                    label=label, color=color, lw=1.4)
        ax.set_yscale('log')
        ax.set_title(title)
        ax.set_xlabel('Training update')
    axes[0].set_ylabel('Validation oracle MSE')
    axes[1].legend(frameon=False, fontsize=8)
    for suffix in ('pdf', 'png'):
        fig.savefig(output / f'clock-range-learning.{suffix}', dpi=180)
    plt.close(fig)

    result = json.loads(response.read_text())
    assert result['state'] == 'COMPLETE'
    r = result['rows']
    x = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(7.0, 3.5), layout='constrained')
    for offset, key, label, color in [(-.2, 'rank_only_floor', 'Any rank-six correction', '#9d9d9d'),
                                     (.2, 'fixed_response_floor', 'Measured response space', '#2678b2')]:
        ax.bar(x + offset, [v[key] / v['actual_error'] for v in r], width=.36, label=label, color=color)
    ax.axhline(1, color='#555555', lw=.8, ls='--')
    ax.set_xticks(x, [f'{v["time"]:.1f}' for v in r])
    ax.set_xlabel('FM time')
    ax.set_ylabel('Conditional floor / actual Jacobian error')
    ax.set_ylim(0, 1.04)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title('Frozen scalar model on product16 (20,000 updates)')
    for suffix in ('pdf', 'png'):
        fig.savefig(output / f'clock-response-floor.{suffix}', dpi=180)
    plt.close(fig)
    inputs.append(response)
    (output.parent / 'plot-provenance.json').write_text(json.dumps({
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(inputs))},
        'scope': 'Descriptive fixed-protocol comparisons. No error bars or inferential tests.',
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('studies', 'response', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    plot(args.studies, args.response, args.output)
