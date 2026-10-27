"""Curated frame-screen figures from audited tables and immutable sample arrays."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser();parser.add_argument('--root', type=Path, required=True)
parser.add_argument('--results', type=Path, required=True);args = parser.parse_args()
out = args.results/'figures';out.mkdir(exist_ok=True)
def read(name):
    with (args.results/name).open() as stream:return list(csv.DictReader(stream))
rows = read('comparison.csv');curves = read('curves.csv');history = read('P-history.csv')
import json
settings = json.loads((args.root/'settings.json').read_text());reference = Path(settings['geometry_reference'])
families = ['ode_k2', 'frame_k2', 'global_k2', 'frame_k1', 'swiglu192', 'swiglu257']
names = ['Original K2', 'Input frame K2', 'Global frame K2', 'Input frame K1', 'Swi192', 'Swi257']
colors = ['#888888', '#c44e52', '#ddaa33', '#8172b3', '#55a868', '#4c72b0']
plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False, 'pdf.fonttype':42})
def save(fig, name):
    for suffix in ('png', 'pdf'):fig.savefig(out/f'{name}.{suffix}', dpi=170, bbox_inches='tight')
    plt.close(fig)

fig, axes = plt.subplots(2, 3, figsize=(13, 6.8))
for i, law in enumerate(('thin', 'thick')):
    for j, (key, label) in enumerate([('oracle_mse', 'Field MSE / ambient coordinate'),
            ('tangent_mse_per_dimension', 'Tangent field MSE / latent coordinate'),
            ('tangent_sw2', 'Tangent sliced W2')]):
        values = [float(next(r[key] for r in rows if r['law']==law and r['family']==f)) for f in families]
        ax = axes[i, j];ax.scatter(range(6), values, c=colors, s=34);ax.set_yscale('log')
        ax.set_xticks(range(6), ['Orig', 'Frame\nK2', 'Global\nK2', 'Frame\nK1', 'Swi\n192', 'Swi\n257'], fontsize=8)
        ax.set_title(f'{law.capitalize()}: {label}');ax.grid(axis='y', alpha=.2)
fig.tight_layout();save(fig, 'fixed-comparison')

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7))
for law, ax in zip(('thin', 'thick'), axes):
    for family, name, color in zip(families, names, colors):
        selected = sorted([r for r in curves if r['law']==law and r['family']==family], key=lambda r:int(r['step']))
        ax.plot([int(r['step']) for r in selected], [float(r['oracle_mse']) for r in selected],
                color=color, marker='.', label=name)
    ax.set_yscale('log');ax.set_xlabel('Updates');ax.set_ylabel('Full-bank decoded field MSE')
    ax.set_title(law.capitalize());ax.grid(alpha=.2)
handles, labels = axes[0].get_legend_handles_labels();fig.legend(handles, labels, ncol=3, loc='upper center', fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .85));save(fig, 'learning-curves')

fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.2))
for i, law in enumerate(('thin', 'thick')):
    for j, family in enumerate(('frame_k2', 'global_k2', 'frame_k1')):
        ax = axes[i, j]
        layers = sorted({r['layer'] for r in history if r['law']==law and r['family']==family})
        for layer in layers:
            selected = sorted([r for r in history if r['law']==law and r['family']==family and r['layer']==layer], key=lambda r:int(r['step']))
            ax.plot([int(r['step']) for r in selected], [float(r['sigma_max']) for r in selected], marker='.', label=layer.replace('blocks.', 'B').replace('.ffn.ode', '/'))
        ax.set_title(f'{law.capitalize()} / {names[families.index(family)]}');ax.set_xlabel('Updates')
        ax.set_ylabel('Largest singular value of learned P');ax.grid(alpha=.2)
handles, labels = axes[0, 0].get_legend_handles_labels();fig.legend(handles, labels, loc='upper center', ncol=6, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .94));save(fig, 'P-histories')

basis = np.load(args.root/'basis.npy')
columns = [('Target', None), ('Oracle Heun32', 'oracle')]+list(zip(names, families))
fig, axes = plt.subplots(2, len(columns), figsize=(16, 5.4))
for i, law in enumerate(('thin', 'thick')):
    entry = next(r for r in settings['laws'] if r['law']==law);banks = Path(entry['banks'])
    for j, (title, family) in enumerate(columns):
        if family is None:
            with np.load(banks/'sampling.npz') as data:points = data['target']
        elif family=='oracle':
            with np.load(banks/'oracle-samples.npz') as data:points = data['generated']
        else:
            base = reference if family in ('ode_k2', 'swiglu192') else args.root
            with np.load(base/'output/runs'/law/f'{family}_clean/diagnostics/samples.npz') as data:points = data['generated']
        plane = points@basis;normal = points-plane@basis.T
        rms = float(np.sqrt(np.sum(normal**2)/(len(points)*126)))
        ax = axes[i, j];ax.scatter(plane[:, 0], plane[:, 1], s=1.3, alpha=.3, rasterized=True)
        ax.set_xlim(-4.5,4.5);ax.set_ylim(-4.5,4.5);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
        if i==0:ax.set_title(title, fontsize=8)
        if j==0:ax.set_ylabel(law.capitalize())
        ax.set_xlabel(f'Normal RMS {rms:.3f}', fontsize=8)
fig.suptitle('Evaluation-only latent-plane projection; all 2048 fixed sources', fontsize=10)
fig.subplots_adjust(left=.025, right=.995, bottom=.06, top=.88, wspace=.12, hspace=.37)
save(fig, 'generation-cases')
print('Saved four PNG/PDF figures')
