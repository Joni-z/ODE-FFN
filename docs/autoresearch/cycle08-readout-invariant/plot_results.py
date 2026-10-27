"""Plot the audited fixed-readout identities; never recompute training or samples."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, required=True)
args = parser.parse_args()
results = args.results
out = results / 'figures'
out.mkdir(exist_ok=True)

def read(name):
    with (results / name).open() as stream:
        return list(csv.DictReader(stream))

rows = read('invariants.csv')
interventions = read('P-interventions.csv')
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))

for head, color in [('clean', '#55a868'), ('shifted', '#4c72b0'), ('direct', '#c44e52')]:
    selected = [r for r in rows if r['head'] == head]
    axes[0].scatter(range(len(selected)), [float(r['identity_max']) for r in selected],
                    label=head.capitalize(), color=color, s=28)
axes[0].axhline(1e-4, color='black', linestyle='--', label='Registered tolerance')
axes[0].set_yscale('log')
axes[0].set_ylim(3e-8, 3e-4)
axes[0].set_xticks(range(6), ['Thin\nODE', 'Thin\nSwi128', 'Thin\nSwi192',
                             'Thick\nODE', 'Thick\nSwi128', 'Thick\nSwi192'], fontsize=8)
axes[0].set_ylabel('Max coordinate error: sample vs affine prediction')
axes[0].set_title('All 18 frozen models satisfy the identity')
axes[0].legend(fontsize=8, frameon=False)

x = np.arange(6)
axes[1].bar(x - .18, [float(r['whole_sample_displacement_rms']) for r in interventions],
            .34, color='#4c72b0', label='Whole sample: RMS change')
axes[1].bar(x + .18, [float(r['change_in_null_coordinates_max']) for r in interventions],
            .34, color='#c44e52', label='Null coordinates: max change')
axes[1].set_yscale('log')
axes[1].set_ylim(5e-8, 4)
axes[1].set_xticks(x, [r['law'].capitalize() + '\n' + r['head'] for r in interventions], fontsize=8)
axes[1].set_ylabel('Absolute change after setting all six P to zero')
axes[1].set_title('P changes samples inside the readout span')
axes[1].legend(fontsize=8, frameon=False)

for i, (law, color) in enumerate([('thin', '#55a868'), ('thick', '#4c72b0')]):
    # Show each fitted family, rather than picking the most favorable bound.
    for j, head in enumerate(['clean', 'shifted', 'direct']):
        selected = [r for r in rows if r['law'] == law and r['head'] == head]
        axes[2].scatter(j + (i - .5)*.23 + np.linspace(-.025, .025, 3),
                        [float(r['population_full_W2_squared_lower_bound']) for r in selected],
                        color=color, s=25, label=law.capitalize() if j == 0 else None)
axes[2].set_yscale('log')
axes[2].set_xticks(range(3), ['Clean', 'Shifted', 'Direct'])
axes[2].set_ylabel('Population full W2 squared: lower bound')
axes[2].set_title('Distribution obstruction; this is not SW2')
axes[2].legend(frameon=False)
for ax in axes:
    ax.grid(axis='y', alpha=.2)
fig.tight_layout()
for suffix in ['png', 'pdf']:
    fig.savefig(out / f'readout-invariant.{suffix}', dpi=170, bbox_inches='tight')
plt.close(fig)
print('Saved one PNG/PDF figure from audited CSV files')
