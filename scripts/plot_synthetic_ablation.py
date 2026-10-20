"""Plot the complete, fixed-seed ablation summary; no inferential error bars."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.summary.read_text())
    if len(records) != 15 or not all(r['complete'] and r['last_step'] == 5000
                                    and r['seed'] == 0 for r in records):
        raise SystemExit('All 15 seed-0 runs must finish 5000 steps before plotting.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    datasets = [('gaussian_aniso_2d', 'Anisotropic Gaussian'),
                ('gmm_ring8_2d', '8-Gaussian ring'), ('two_moons_2d', 'Two Moons')]
    variants = [('ffn', None, 'FFN', '#333333', '-'),
                ('ode_original', 1, 'ODE K=1', '#0072B2', '--'),
                ('ode_spectral_cap', 1, 'Cap K=1', '#D55E00', '--'),
                ('ode_original', 2, 'ODE K=2', '#0072B2', '-'),
                ('ode_spectral_cap', 2, 'Cap K=2', '#D55E00', '-')]
    rows = {(r['dataset'], r['model'], r['orders']): r for r in records}
    assert len(rows) == 15
    args.output.mkdir(parents=True, exist_ok=True)

    def save(fig, filename):
        for extension in ('png', 'pdf'):
            fig.savefig(args.output / f'{filename}.{extension}', dpi=180,
                        bbox_inches='tight', facecolor='white')
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, (dataset, title) in zip(axes, datasets):
        values = []
        for index, (model, order, label, color, style) in enumerate(variants):
            row = rows[(dataset, model, order)]
            value = row['last_sample']['sliced_w2']
            values.append(value)
            ax.bar(index, value, color=color, alpha=0.8,
                   hatch='//' if order == 1 else None)
            ax.text(index, value, f'{value:.4f}', ha='center', va='bottom', fontsize=8,
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.3})
        floor = rows[(dataset, 'ffn', None)]['last_sample']['reference_floor']
        ax.axhline(floor, color='#777777', linestyle=':',
                   label='Independent reference samples')
        ax.set_xticks(range(5), [v[2] for v in variants], rotation=30, ha='right')
        ax.set_ylim(0, max(values + [floor]) * 1.2)
        ax.set_title(title)
        ax.set_ylabel('Final sliced W2 (lower is better)')
        ax.legend(fontsize=7)
    fig.suptitle('5000 steps, seed 0 only; no run-wise uncertainty estimates')
    save(fig, 'figure-01-final-generation')

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for column, (dataset, title) in enumerate(datasets):
        metric = 'validation_pair_mse' if dataset == 'two_moons_2d' else 'validation_oracle_mse'
        for model, order, label, color, style in variants:
            row = rows[(dataset, model, order)]
            curve = row['curve']
            axes[0, column].plot([r['step'] for r in curve], [r[metric] for r in curve],
                                  color=color, linestyle=style, label=label)
            sampled = [r for r in curve if r.get('sample') is not None]
            axes[1, column].plot([r['step'] for r in sampled],
                                 [r['sample']['sliced_w2'] for r in sampled],
                                 color=color, linestyle=style, marker='.', label=label)
        axes[0, column].set_title(title)
        axes[0, column].set_ylabel('Paired MSE' if dataset == 'two_moons_2d' else 'Oracle MSE')
        axes[1, column].set_ylabel('Sliced W2')
        for ax in axes[:, column]:
            ax.set_yscale('log')
            ax.set_xlabel('Training step')
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Fixed evaluation data; single-run trajectories, seed 0')
    save(fig, 'figure-02-training-dynamics')

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for column, (dataset, title) in enumerate(datasets):
        for model, order, label, color, style in variants[1:]:
            curve = rows[(dataset, model, order)]['curve']
            axes[0, column].plot([r['step'] for r in curve],
                                 [r['layers']['effective.sigma_max']['max'] for r in curve],
                                 color=color, linestyle=style, label=label)
            if order == 2:
                axes[1, column].plot([r['step'] for r in curve],
                    [r['layers']['activation.term2_rms_over_input']['mean'] for r in curve],
                    color=color, linestyle=style, label=label)
        axes[0, column].set_title(title)
        axes[0, column].set_ylabel('Largest P_eff singular value across layers')
        axes[1, column].set_ylabel('Term 2 / input RMS, mean across layers')
        for ax in axes[:, column]:
            ax.set_xlabel('Training step')
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle('Internal operator dynamics; second-order terms exist only for K=2')
    save(fig, 'figure-03-operator-dynamics')


if __name__ == '__main__':
    main()
