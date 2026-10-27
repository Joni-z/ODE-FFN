"""Plot audited complete-field derivative diagnostics, no quality selection."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser();parser.add_argument('--results', type=Path, required=True)
args = parser.parse_args();out = args.results/'figures';out.mkdir(exist_ok=True)
def read(name):
    with (args.results/name).open() as stream:
        return list(csv.DictReader(stream))
panels, late = read('panels.csv'), read('late-summary.csv')
families = ['ode_k2', 'swiglu128', 'swiglu192'];names = ['Original ODE', 'Swi128', 'Swi192']
colors = ['#c44e52', '#55a868', '#4c72b0']
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.7))
labels = [r['law'].capitalize()+'\n'+names[families.index(r['family'])] for r in late]
x = np.arange(6)
ax = axes[0, 0]
symmetric = [float(r['tangent_symmetric']) for r in late]
skew = [float(r['tangent_skew']) for r in late]
ax.bar(x, symmetric, label='Symmetric residual', color='#4c72b0')
ax.bar(x, skew, bottom=symmetric, label='Skew residual', color='#ddaa33')
ax.set_xticks(x, labels, fontsize=8);ax.set_ylabel('Mean squared Frobenius error (2 by 2 tangent block)')
ax.set_title('Late clean-head derivative error is mostly symmetric')
ax.legend(fontsize=8, frameon=False, loc='lower right')

ax = axes[0, 1]
shares = [100*(1-float(r['symmetric_share'])) for r in late]
ax.scatter(x, shares, c=[colors[families.index(r['family'])] for r in late], s=35)
for position, value in zip(x, shares):
    ax.annotate(f'{value:.3f}%', (position, value), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
ax.set_xticks(x, labels, fontsize=8);ax.set_yscale('log');ax.set_ylim(.01, 3)
ax.set_ylabel('Skew share of the same residual energy (%)')
ax.set_title('A zero-skew constraint targets only a small component')

ax = axes[1, 0]
for family, name, color in zip(families, names, colors):
    rows = sorted([r for r in panels if r['law']=='thin' and r['family']==family and r['head']=='clean'], key=lambda r: float(r['time']))
    ax.plot([float(r['time']) for r in rows], [float(r['tangent_residual']) for r in rows],
            color=color, marker='o', label=name)
ax.set_yscale('log');ax.set_xlabel('External time');ax.set_ylabel('Tangent derivative residual energy')
ax.set_title('Thin / clean: the late panel at 0.7 is hardest')
ax.legend(frameon=False, fontsize=8)

ax = axes[1, 1]
for law, style, color in [('thin', 'o-', '#c44e52'), ('thick', 's--', '#4c72b0')]:
    rows = sorted([r for r in panels if r['law']==law and r['family']=='ode_k2' and r['head']=='clean'], key=lambda r: float(r['time']))
    ax.plot([float(r['time']) for r in rows], [float(r['tangent_full_oracle_expanding']) for r in rows],
            style, color=color, label=law.capitalize(), markersize=5)
ax.set_xlabel('External time');ax.set_ylabel('Fraction of 64 anchors with a positive eigenvalue')
ax.set_ylim(-.04, 1.08);ax.set_title('The true tangent field sometimes needs local expansion')
ax.legend(frameon=False, fontsize=8)
for ax in axes.flat:
    ax.grid(axis='y', alpha=.2)
fig.tight_layout()
for suffix in ('png', 'pdf'):
    fig.savefig(out/f'field-derivatives.{suffix}', dpi=170, bbox_inches='tight')
plt.close(fig)
print('Saved one PNG/PDF figure from audited derivative tables')
