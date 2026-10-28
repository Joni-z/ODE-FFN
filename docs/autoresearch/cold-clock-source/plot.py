"""Seven figures from complete audited tables, without model or metric changes."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(root):
    out = root/'curation'
    assert json.loads((out/'summary.json').read_text())['status'] == 'PASS'
    layout = json.loads((out/'plot-inputs.json').read_text())
    labels, focus, p_arms, cases = (layout[k] for k in ('labels','focus','P_arms','cases'))
    read = lambda name: list(csv.DictReader((out/name).open()))
    figures = out/'figures'
    figures.mkdir()
    colors = dict(zip(focus, ('#1f77b4','#ff7f0e','#2ca02c','#9467bd',
                             '#8c564b','#7f7f7f','#00897b','#c2185b')))
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    def export(fig, name):
        for extension in ('png','pdf'):
            fig.savefig(figures/f'{name}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)

    primary = read('primary-20k.csv')
    fig, ax = plt.subplots(figsize=(10, 6.5), layout='constrained')
    indices = np.arange(len(primary))
    values = [float(v['fid']) for v in primary]
    ax.barh(indices, values, color=[colors.get(v['arm'], '#b8b8b8') for v in primary])
    ax.set_yticks(indices, [labels[v['arm']] for v in primary])
    for i, value in enumerate(values):
        ax.text(value+1, i, f'{value:.2f}', va='center', fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel='FID (lower is better)', xlim=(0, 1.13*max(values)),
           title='All 17 fixed 20k endpoints; 1,024 generated / 2,048 reference images')
    export(fig, 'primary-FID')

    comparison = read('comparison.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), layout='constrained')
    for arm in focus:
        rows = sorted((v for v in comparison if v['arm'] == arm), key=lambda v: int(v['step']))
        for ax, key in zip(axes, ('fid','weighted_mse','clean_mse')):
            ax.plot([int(v['step']) for v in rows], [float(v[key]) for v in rows],
                    '-o', label=labels[arm], color=colors[arm], markersize=3)
    for ax, title in zip(axes, ('FID','Weighted validation MSE','Clean validation MSE')):
        ax.set(xlabel='Training updates', title=title)
    axes[0].legend(fontsize=6.5)
    export(fig, 'budget-comparison')

    curves = read('training-curves.csv')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), layout='constrained')
    for arm in focus:
        rows = [v for v in curves if v['arm'] == arm and int(v['step']) > 0]
        for ax, key in zip(axes, ('weighted_mse','clean_mse','clip_fraction')):
            selected = [v for v in rows if v[key] != '']
            ax.plot([int(v['step']) for v in selected], [float(v[key]) for v in selected],
                    '-o', label=labels[arm], color=colors[arm], markersize=2)
    for ax, title in zip(axes, ('Weighted validation MSE','Clean validation MSE','Gradient clipping fraction')):
        ax.set(xlabel='Training updates', title=title)
    axes[0].legend(fontsize=6.5)
    export(fig, 'training-curves')

    spectra = read('P-history.csv')
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3), layout='constrained')
    for arm in p_arms:
        rows = [v for v in spectra if v['arm'] == arm]
        steps = sorted({int(v['step']) for v in rows})
        for ax, key in zip(axes, ('sigma_max','singular_CV','eigen_max_abs','eigen_max_real')):
            means = [np.mean([float(v[key]) for v in rows if int(v['step']) == step]) for step in steps]
            ax.plot(steps, means, '-o', label=labels[arm], color=colors[arm], markersize=3)
    for ax, title in zip(axes, ('Largest singular value','Singular-value CV','Spectral radius','Largest real eigenvalue')):
        ax.set(xlabel='Training updates', title=title)
    axes[0].legend(fontsize=6.5)
    fig.suptitle('Mean over six physical P matrices; not a full-predictor stability bound')
    export(fig, 'P-trajectories')

    profiles = read('clock-profiles.csv')
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.7), sharex=True, sharey=True, layout='constrained')
    for ax, layer in zip(axes.flat, sorted({v['layer'] for v in profiles})):
        for arm, marker in (('feature_clock_zero','o'),('time_clock_zero','s')):
            rows = sorted((v for v in profiles if v['arm'] == arm and int(v['step']) == 20000
                           and v['layer'] == layer), key=lambda v: float(v['time']))
            assert len(rows) == 5
            t = np.array([float(v['time']) for v in rows])
            mean = np.array([float(v['mean']) for v in rows])
            std = np.array([float(v['std']) for v in rows])
            ax.plot(t, mean, '-'+marker, label=labels[arm], color=colors[arm], markersize=3)
            ax.fill_between(t, mean-std, mean+std, color=colors[arm], alpha=.15)
        ax.axhline(.6, color='#777777', ls='--', lw=.8, label='Initial 0.6')
        ax.set(xlabel='Corruption time t', ylabel='Local clock', title=layer)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('20k clock profiles; bands are within-panel SD, not confidence intervals')
    export(fig, 'clock-profiles')

    eigenvalues = read('P-eigenvalues.csv')
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7), sharex=True, sharey=True, layout='constrained')
    for ax, arm in zip(axes.flat, p_arms):
        for step, color, alpha, label in ((0,'#777777',.25,'Initial'),(20000,colors[arm],.55,'20k')):
            rows = [v for v in eigenvalues if v['arm'] == arm and int(v['step']) == step]
            assert len(rows) == 384
            ax.scatter([float(v['real']) for v in rows], [float(v['imag']) for v in rows],
                       s=8, color=color, alpha=alpha, label=label)
        ax.axhline(0, color='#888888', lw=.6)
        ax.axvline(0, color='#888888', lw=.6)
        ax.set(title=labels[arm], xlabel='Real part', ylabel='Imaginary part')
        ax.set_aspect('equal', adjustable='box')
        ax.legend(fontsize=7)
    fig.suptitle('All eigenvalues of six P matrices; initial versus fixed 20k endpoint')
    export(fig, 'P-eigenvalues')

    fig, axes = plt.subplots(len(cases), 8, figsize=(10, 9), squeeze=False, layout='constrained')
    with np.load(out/'case-images.npz') as images:
        for i, arm in enumerate(cases):
            values = np.rint((np.clip(images[arm], -1, 1)+np.float32(1))*np.float32(127.5)).astype(np.uint8)
            for j, value in enumerate(values):
                ax = axes[i, j]
                ax.imshow(value.transpose(1, 2, 0), interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                if j == 0:
                    ax.set_ylabel(labels[arm], rotation=0, ha='right', va='center', fontsize=7)
                if i == 0:
                    ax.set_title(f'Start {j}', fontsize=8)
    export(fig, 'fixed-start-cases')
    assert len(list(figures.glob('*.png'))) == len(list(figures.glob('*.pdf'))) == 7
    print('COLD_CLOCK_SOURCE_FIGURES_COMPLETE 7', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    main(parser.parse_args().root.resolve())
