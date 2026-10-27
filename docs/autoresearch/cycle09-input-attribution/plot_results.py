"""Plots of audited input-information attribution, without selecting checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, required=True)
args = parser.parse_args()
out = args.results/'figures';out.mkdir(exist_ok=True)
with (args.results/'decomposition.csv').open() as stream:
    rows = list(csv.DictReader(stream))
decision = json.loads((args.results/'decision.json').read_text())
families = ['ode_k2', 'swiglu128', 'swiglu192']
names = ['Original ODE K2', 'Swi128', 'Swi192']
colors = ['#c44e52', '#55a868', '#4c72b0']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})

def select(law, family, scope='full', step=5000, time_bin='all'):
    return next(r for r in rows if r['law']==law and r['family']==family and r['head']=='clean'
                and r['scope']==scope and int(r['step'])==step and r['time_bin']==time_bin)

fig, axes = plt.subplots(2, 2, figsize=(11.8, 8))
ax = axes[0, 0]
for i, (key, label, color) in enumerate([('total', 'Total oracle error', '#777777'),
    ('information', 'Input-information component', '#ddaa33'), ('residual', 'Remaining residual', '#4c72b0')]):
    ax.bar(np.arange(3)+(i-1)*.25, [float(select('thin', f)[key]) for f in families],
           .23, label=label, color=color)
ax.set_xticks(range(3), names);ax.set_yscale('log');ax.set_ylim(1e-5, .02)
ax.set_ylabel('Field MSE per ambient coordinate')
ax.set_title('Thin / clean: little error comes from input loss')
ax.legend(fontsize=8, frameon=False)

ax = axes[0, 1]
for j, (key, label, color) in enumerate([('information_fraction', 'Information', '#ddaa33'),
    ('residual_fraction', 'Residual', '#4c72b0'), ('cross_fraction', 'Cross', '#888888')]):
    values = [100*r[key] for r in decision['differences']]
    positions = np.arange(2)+(j-1)*.23
    ax.bar(positions, values, .21, label=label, color=color)
    for position, value in zip(positions, values):
        ax.text(position, max(value, 0)+2, f'{value:.1f}%', ha='center', fontsize=8)
ax.set_xticks(range(2), ['Swi128 minus ODE', 'Swi192 minus ODE'])
ax.set_ylim(-9, 120);ax.axhline(0, color='black', linewidth=.7)
ax.set_ylabel('Signed share of the already observed field gap (%)')
ax.set_title('95–98% of the gap is in the remaining residual')
ax.legend(ncol=3, fontsize=8, frameon=False, loc='upper center')

ax = axes[1, 0]
bins = [f'{i/5:.1f}-{(i+1)/5:.1f}' for i in range(5)]
for family, name, color in zip(families, names, colors):
    ax.plot(range(5), [float(select('thin', family, 'tangent', time_bin=b)['residual']) for b in bins],
            marker='o', label=name+' residual', color=color)
    ax.plot(range(5), [float(select('thin', family, 'tangent', time_bin=b)['information']) for b in bins],
            linestyle='--', color=color, alpha=.8)
ax.set_xticks(range(5), bins);ax.set_yscale('log')
ax.set_xlabel('External time bin');ax.set_ylabel('Tangent field MSE per latent coordinate')
ax.set_title('Thin / clean: larger residuals after time 0.6')
ax.legend(fontsize=8, frameon=False)
ax.text(.02, .03, 'Dashed: input-information component', transform=ax.transAxes, fontsize=8)

ax = axes[1, 1]
for family, name, color in zip(families, names, colors):
    steps = [0, 1000, 2500, 5000]
    ax.plot(steps, [float(select('thick', family, 'tangent', step=s)['residual']) for s in steps],
            marker='o', color=color, label=name)
ax.set_xlabel('Training updates');ax.set_ylabel('Tangent residual MSE per latent coordinate')
ax.set_yscale('log');ax.set_title('Thick / clean: ODE still trails the compute control')
ax.legend(fontsize=8, frameon=False)
for ax in axes.flat:
    ax.grid(axis='y', alpha=.2)
fig.tight_layout()
for suffix in ('png', 'pdf'):
    fig.savefig(out/f'input-attribution.{suffix}', dpi=170, bbox_inches='tight')
plt.close(fig)
print('Saved one PNG/PDF figure from independently audited tables')
