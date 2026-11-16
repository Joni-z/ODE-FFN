"""Plot the qualified density comparison without exporting held-out profiles."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from verify_density import read, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--statistics', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    stats = read(a.statistics)
    coordinate = stats.get('coordinate_control', False)
    count, contrasts = (20, 8) if coordinate else (16, 4)
    assert stats['passed'] and stats['qualified_cells'] == count and stats['qualified_contrasts'] == contrasts
    assert len(stats['summaries']) == count and len(stats['contrasts']) == contrasts
    if coordinate:
        assert stats['discovery'] is None and not stats['confirmation_time_profiles_exported']
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    rows = stats['summaries']
    datasets = [('gmm_ring8_2d', 'Ring8'), ('rotated_product_bimodal_16d', 'Product16')]
    families = [('state_midpoint4', 'Shared P', '#2166ac'),
                ('stage_operator_midpoint4', 'Stage P', '#d6604d'),
                ('free_isru', 'Free ISRU', '#757575'), ('swiglu', 'SwiGLU', '#222222')]
    if coordinate:
        families.insert(2, ('contrast_operator_midpoint4', 'Mean-preserving', '#4d9221'))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    saved = []
    def save(fig, name):
        fig.savefig(out/(name+'.png'), dpi=180, bbox_inches='tight')
        fig.savefig(out/(name+'.pdf'), bbox_inches='tight')
        plt.close(fig)
        saved.append(name)
    def get(dataset, mode, prefix):
        return next(r for r in rows if r['dataset'] == dataset and r['mode'] == mode and r['family'].startswith(prefix))
    fig, axes = plt.subplots(2, 2, figsize=(12 if coordinate else 10, 6.5), constrained_layout=True)
    for j, (dataset, label) in enumerate(datasets):
        for i, mode in enumerate(['original', 'projected']):
            ax = axes[i, j]
            for x, (family, name, color) in enumerate(families):
                r = get(dataset, mode, family)
                lo, hi = r['bootstrap_95']
                ax.vlines(x, lo, hi, color=color, lw=2)
                ax.scatter(x, r['mean'], color=color, s=42, zorder=3)
            ax.axhline(0, color='.7', lw=.7)
            ax.set_xticks(range(len(families)), [f[1] for f in families], rotation=15 if coordinate else 0)
            ax.set_title(f'{label}: {mode}')
            ax.set_ylabel('Mean terminal log-density ratio (nats)')
    fig.suptitle('Four common source banks; 1024 independent paths per fixed model/field\nIntervals: within-bank bootstrap 95%')
    save(fig, 'terminal-density')

    fig, axes = plt.subplots(1, 2, figsize=(12 if coordinate else 11, 4.2), constrained_layout=True)
    if coordinate:
        for ax, (dataset, title) in zip(axes, datasets):
            comparisons = [r for r in stats['contrasts'] if r['dataset'] == dataset]
            assert len(comparisons) == 4
            labels = []
            for i, r in enumerate(comparisons):
                lo, hi = r['bonferroni_eight_contrasts_99_375']
                assert r['candidate'] == 'contrast_operator_midpoint4'
                ax.hlines(i, lo, hi, color='#4d9221', lw=2)
                ax.scatter(r['mean'], i, color='#4d9221', s=36)
                reference = 'shared' if r['reference'] == 'state_midpoint4' else 'anchored'
                labels.append('vs '+reference+' / '+r['mode'])
            ax.axvline(0, color='.5', lw=.8)
            ax.set_yticks(range(4), labels)
            ax.set_xlabel('Mean-preserving minus reference (nats)\nPositive favors reference; intervals: 99.375%')
            ax.set_title(title)
        fig.suptitle('Eight registered paired contrasts at fixed trained weights')
        save(fig, 'paired-coordinate-density')
    else:
        ax = axes[0]
        labels = []
        for i, r in enumerate(stats['contrasts']):
            lo, hi = r['bonferroni_four_contrasts_98_75']
            ax.hlines(i, lo, hi, color='#2166ac', lw=2)
            ax.scatter(r['mean'], i, color='#2166ac', s=36)
            label = 'Ring8' if r['dataset'] == 'gmm_ring8_2d' else 'Product16'
            labels.append(label+' / '+r['mode'])
        ax.axvline(0, color='.5', lw=.8)
        ax.set_yticks(range(4), labels)
        ax.set_xlabel('Stage minus shared density error (nats)\nPositive favors sharing; intervals: 98.75%')
        ax.set_title('Four predeclared paired contrasts')
        d = stats['discovery']
        x = (np.asarray(d['bin_edges'])[:-1]+np.asarray(d['bin_edges'])[1:])/2
        ci = np.asarray(d['paired_95_by_bin'])
        axes[1].vlines(x, ci[:, 0], ci[:, 1], color='#2166ac')
        axes[1].scatter(x, d['mean_stage_minus_shared_by_bin'], color='#2166ac', s=30)
        axes[1].axhline(0, color='.5', lw=.8)
        axes[1].set(xlabel='Flow-matching time', ylabel='Signed contribution to stage-minus-shared ratio',
                    title='Product16 projected: discovery bank 0 only')
        save(fig, 'paired-density-and-discovery')
    with (out/'scores.csv').open('w') as f:
        fields = ['dataset', 'family', 'mode', 'mean', 'sd', 'mc_se', 'same_sources_sw2_pool', 'resolution_mean_change']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in rows)
    (out/'figure-bindings.json').write_text(json.dumps(dict(statistics_sha256=sha(a.statistics),
        plot_source_sha256=sha(__file__), figures=saved, confirmation_time_profiles_exported=False), indent=2)+'\n')
    print(json.dumps(dict(passed=True, figures=len(saved), qualified_cells=count)))


if __name__ == '__main__':
    main()
