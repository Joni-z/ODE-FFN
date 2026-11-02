"""Extract the existing ODEFFN evidence and draw the current Report2 figure.

No checkpoints, model execution, remote access or new evaluation is involved.
The row selectors and source hashes make every plotted value traceable.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def extract():
    records = []
    specifications = [
        ('readout', 'docs/research-conservative-readout-2026-10-26/results/comparison.csv',
         ('dataset', 'arm'), ('oracle_mse', 'sw2', 'parameters', 'dense_forward_macs')),
        ('frame', 'docs/autoresearch/cycle11-conditional-frame/results/comparison.csv',
         ('law', 'family'), ('oracle_mse', 'tangent_mse_per_dimension', 'tangent_sw2',
                            'parameters', 'dense_forward_macs')),
        ('cubic', 'docs/research-cubic-split-2026-10-26/comparison.csv',
         ('dataset', 'family'), ('oracle_mse', 'sw2')),
    ]
    hashes = {}
    for study, source, keys, metrics in specifications:
        path = REPO / source
        hashes[source] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            selector = {k: row[k] for k in keys}
            assert sum(all(r[k] == v for k, v in selector.items()) for r in rows) == 1
            for metric in metrics:
                records.append(dict(study=study, dataset=row[keys[0]], arm=row[keys[1]],
                                    metric=metric, value=row[metric], source=source,
                                    selector=json.dumps(selector, sort_keys=True)))
    output = ROOT / 'data/ode-mainline.csv'
    with output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (ROOT / 'data/ode-mainline-sources.json').write_text(json.dumps(hashes, indent=2) + '\n')
    return records


def main():
    records = extract()

    def value(study, dataset, arm, metric):
        rows = [r for r in records if (r['study'], r['dataset'], r['arm'], r['metric'])
                == (study, dataset, arm, metric)]
        assert len(rows) == 1
        return float(rows[0]['value'])

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.axisbelow': True, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(6.85, 2.18), layout='constrained',
                             gridspec_kw={'width_ratios': [1.08, 1, 1]})
    names = ['ODE K2', 'Swi128', 'Swi192']
    families = ['ode_k2', 'swiglu128', 'swiglu192']
    y = np.arange(3)
    for offset, mode, color, label in [(-.18, 'direct', '#93999f', 'Direct'),
                                      (.18, 'gradient', '#0072B2', 'Full gradient')]:
        values = [value('readout', 'rotated_product_bimodal_16d', f'{f}_{mode}',
                        'oracle_mse') for f in families]
        axes[0].barh(y + offset, values, height=.33, color=color, label=label)
    axes[0].set(yticks=y, yticklabels=names, xlabel='Field MSE', xlim=(0, .35),
                title='A  Product16: readout')
    axes[0].invert_yaxis()
    axes[0].legend(loc='lower right', fontsize=7, frameon=False)
    families = ['ode_k2', 'frame_k1', 'frame_k2', 'swiglu192', 'swiglu257']
    names = ['ODE K2', 'Frame K1', 'Frame K2', 'Swi192', 'Swi257']
    colors = ['#93999f', '#56B4E9', '#0072B2', '#D55E00', '#E69F00']
    for ax, law, metric, scale, title, xlabel in [
        (axes[1], 'thin', 'oracle_mse', 1000, 'B  Thin manifold', 'Field MSE × 1,000'),
        (axes[2], 'thick', 'tangent_sw2', 1, 'C  Thick distribution', 'Tangent SW2')]:
        ax.barh(np.arange(5), [scale * value('frame', law, f, metric) for f in families],
                color=colors, height=.65)
        ax.set(yticks=np.arange(5), yticklabels=names, xlabel=xlabel, title=title)
        ax.invert_yaxis()
    for ax in axes:
        ax.grid(axis='x', alpha=.2)
        ax.tick_params(axis='y', length=0)
        ax.title.set_fontsize(8.5)
    for suffix in ('pdf', 'png'):
        fig.savefig(ROOT / f'figures/ode-mainline.{suffix}', dpi=240,
                    bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    print(f'Extracted {len(records)} values from {len(set(r["source"] for r in records))} tables.')


if __name__ == '__main__':
    main()
