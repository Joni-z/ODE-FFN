"""Figures from audited fixed-budget tables; no model or metric modification."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root):
    out = root/'curation'
    assert (out/'decision.json').exists()
    inputs = json.loads((out/'plot-inputs.json').read_text())
    labels, focus, cases = inputs['labels'], inputs['focus'], inputs['cases']
    read = lambda name: list(csv.DictReader((out/name).open()))
    figures = out/'figures'
    figures.mkdir()
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = dict(zip(focus, plt.get_cmap('tab10').colors))
    def export(fig, name):
        for extension in ('png', 'pdf'):
            fig.savefig(figures/f'{name}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    primary = read('primary-20k.csv')
    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    y = np.arange(len(primary))
    ax.barh(y, [float(v['fid']) for v in primary],
            color=[colors.get(v['arm'], '#b8b8b8') for v in primary])
    ax.set_yticks(y, [labels[v['arm']] for v in primary])
    for j, row in enumerate(primary):
        ax.text(float(row['fid'])+1, j, f"{float(row['fid']):.2f}", va='center', fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel='FID (lower is better)', xlim=(0, max(float(v['fid']) for v in primary)*1.13),
           title='All 15 endpoints at 20k; 1,024 generated / 2,048 reference images')
    export(fig, 'primary-FID')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3), layout='constrained')
    comparison = read('comparison.csv')
    for arm in focus:
        values = sorted((v for v in comparison if v['arm'] == arm), key=lambda v: int(v['step']))
        for ax, key in zip(axes, ('fid', 'weighted_mse', 'clean_mse')):
            ax.plot([int(v['step']) for v in values], [float(v[key]) for v in values],
                    '-o', label=labels[arm], color=colors[arm], markersize=3)
    for ax, title in zip(axes, ('FID', 'Weighted validation MSE', 'Clean validation MSE')):
        ax.set(xlabel='Training updates', title=title)
    axes[0].legend(fontsize=7)
    export(fig, 'budget-comparison')
    curves = read('training-curves.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3), layout='constrained')
    for arm in focus:
        values = [v for v in curves if v['arm'] == arm and int(v['step']) > 0]
        for ax, key in zip(axes, ('weighted_mse', 'clean_mse', 'clip_fraction')):
            selected = [v for v in values if v[key] != '']
            ax.plot([int(v['step']) for v in selected], [float(v[key]) for v in selected],
                    '-o', label=labels[arm], color=colors[arm], markersize=2)
    for ax, title in zip(axes, ('Weighted validation MSE', 'Clean validation MSE', 'Gradient clipping fraction')):
        ax.set(xlabel='Training updates', title=title)
    axes[0].legend(fontsize=7)
    export(fig, 'training-curves')
    matrices, branches = read('matrix-history.csv'), read('branch-history.csv')
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), layout='constrained')
    for kind in ('gate', 'value', 'output'):
        selected = [v for v in matrices if v['layer'].endswith('.'+kind)]
        steps = sorted({int(v['step']) for v in selected})
        for ax, key in zip(axes[:2], ('sigma_max', 'singular_CV')):
            means = []
            for t in steps:
                values = [float(v[key]) for v in selected if int(v['step']) == t and v[key] != '']
                means.append(float(np.mean(values)) if values else np.nan)
            ax.plot(steps, means, '-o', label=kind, markersize=3)
    for block in range(3):
        selected = [v for v in branches if int(v['block']) == block]
        for ax, key in zip(axes[2:], ('main_value_weight_norm', 'extra_output_norm')):
            ax.plot([int(v['step']) for v in selected], [float(v[key]) for v in selected],
                    '-o', label=f'Block {block}', markersize=3)
    for ax, title in zip(axes, ('Mean largest singular value', 'Mean singular-value CV',
                               'Main value-weight norm', 'Extra output-weight norm')):
        ax.set(xlabel='Training updates', title=title)
        ax.legend(fontsize=7)
    export(fig, 'branch-and-matrix-histories')
    fig, axes = plt.subplots(len(cases), 8, figsize=(10, 7), squeeze=False, layout='constrained')
    with np.load(out/'case-images.npz') as values:
        for i, arm in enumerate(cases):
            images = np.rint((np.clip(values[arm], -1, 1)+np.float32(1))*np.float32(127.5)).astype(np.uint8)
            for j, raw in enumerate(images):
                ax = axes[i, j]
                ax.imshow(raw.transpose(1, 2, 0), interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(labels[arm], rotation=0, ha='right', va='center', fontsize=8)
                if i == 0:
                    ax.set_title(f'Start {j}', fontsize=8)
    export(fig, 'fixed-start-cases')
    print('SWIGLU_INIT_FIGURES_COMPLETE', len(list(figures.glob('*.png'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root.resolve())
