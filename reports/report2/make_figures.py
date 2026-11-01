"""Rebuild Report2 figures from curated CSVs and unaltered fixed-source grids."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA, FIGURES = ROOT/'data', ROOT/'figures'
ARMS = ('isru','swiglu_matched','swiglu_wide')
COLORS = {'isru':'#0072B2','swiglu_matched':'#D55E00','swiglu_wide':'#009E73'}
LABELS = {'isru':'ISRU','swiglu_matched':'SwiGLU matched','swiglu_wide':'SwiGLU wide'}


def read(name):
    with (DATA/name).open() as handle:
        return list(csv.DictReader(handle))


def save(fig, name):
    fig.savefig(FIGURES/f'{name}.pdf', bbox_inches='tight', pad_inches=.025)
    fig.savefig(FIGURES/f'{name}.png', dpi=220, bbox_inches='tight', pad_inches=.025)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
        'grid.alpha':.18,'axes.axisbelow':True,'legend.frameon':False,
        'pdf.fonttype':42,'ps.fonttype':42})
    validation, scores = read('cifar-validation.csv'), read('cifar-fid.csv')
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 1.95), layout='constrained')
    for arm in ARMS:
        a = [r for r in validation if r['arm'] == arm]
        steps = [int(r['step'])/1000 for r in a]
        axes[0].plot(steps, [float(r['ema']) for r in a], color=COLORS[arm], label=LABELS[arm])
        axes[0].plot(steps, [float(r['raw']) for r in a], color=COLORS[arm], linestyle='--', linewidth=1.)
        a = [r for r in scores if r['arm'] == arm]
        axes[1].plot([int(r['step'])/1000 for r in a], [float(r['FID']) for r in a],
                     color=COLORS[arm], marker='o', label=LABELS[arm])
    axes[0].set(xlabel='Updates (thousands)', ylabel='Validation velocity loss', ylim=(.13,.78))
    axes[0].text(.98,.94,'Solid: EMA\nDashed: raw',ha='right',va='top',transform=axes[0].transAxes,fontsize=8)
    axes[1].set(xlabel='Updates (thousands)', ylabel='EMA FID (1,024 vs 5,000)', xticks=[5,10,25])
    axes[1].legend(fontsize=7.5,loc='upper right')
    save(fig,'cifar-dynamics')

    # Every row uses source indices0–3; no visual-quality selection or enlargement filtering.
    fig, axes = plt.subplots(3, 8, figsize=(6.85, 1.60),
                             gridspec_kw={'wspace':.025,'hspace':.06})
    for row, arm in enumerate(ARMS):
        for tag, offset in (('raw',0),('ema',4)):
            pixels = np.asarray(Image.open(DATA/f'{arm}-{tag}-25k.png').convert('RGB'))
            assert pixels.shape == (256,256,3)
            for index in range(4):
                ax = axes[row,index+offset]
                ax.imshow(pixels[:32,index*32:(index+1)*32],interpolation='nearest')
                ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(False)
        axes[row,0].set_ylabel(LABELS[arm].replace('SwiGLU ','Swi '),rotation=0,
                               ha='right',va='center',labelpad=4,fontsize=7.8)
    fig.text(.32,.995,'Raw weights: source0–3',ha='center',va='bottom',fontsize=8.5)
    fig.text(.72,.995,'EMA weights: same source0–3',ha='center',va='bottom',fontsize=8.5)
    save(fig,'cifar-cases')

    matrix = [r for r in read('cifar-matrices.csv') if r['arm']=='isru']
    def average(step, name, key):
        values = [float(r[key]) for r in matrix if int(r['step'])==step and r['matrix']==name]
        assert len(values)==6
        return float(np.mean(values))
    text = (f"\\textbf{{Observation.}} Mean largest singular values change from "
            f"{average(0,'matrix1','singular_max'):.2f} at initialization to "
            f"{average(25000,'matrix1','singular_max'):.2f} for W1 and "
            f"{average(25000,'matrix2','singular_max'):.2f} for W2 at 25k. "
            f"Mean effective ranks remain {average(25000,'matrix1','effective_rank'):.1f} "
            f"and {average(25000,'matrix2','effective_rank'):.1f} out of 192.\n")
    (ROOT/'matrix-observation.tex').write_text(text)
    fig, axes = plt.subplots(1,2,figsize=(6.85,1.65),layout='constrained')
    for name,color,marker in (('matrix1','#0072B2','o'),('matrix2','#D55E00','s')):
        steps = sorted({int(r['step']) for r in matrix})
        for ax,key in zip(axes,('singular_max','effective_rank')):
            values = np.array([[float(r[key]) for r in matrix if r['matrix']==name and int(r['step'])==s] for s in steps])
            assert values.shape == (5,6)
            x = np.array(steps)/1000
            ax.plot(x,values.mean(1),color=color,marker=marker,label='W1' if name=='matrix1' else 'W2',markersize=3)
            ax.fill_between(x,values.min(1),values.max(1),color=color,alpha=.12,linewidth=0)
    axes[0].set(xlabel='Updates (thousands)',ylabel='Largest singular value')
    axes[1].set(xlabel='Updates (thousands)',ylabel='Entropy effective rank / 192')
    # Display normalized tick labels without changing the actual recorded statistic.
    axes[1].set_ylabel('Entropy effective rank (max192)')
    axes[0].legend(fontsize=8,ncol=2)
    save(fig,'matrix-trajectories')

    synthetic = read('synthetic.csv')
    laws = list(dict.fromkeys(r['law'] for r in synthetic))
    values = {(r['law'],r['arm']):float(r['oracle_mse']) for r in synthetic}
    fig,ax = plt.subplots(figsize=(6.85,2.2),layout='constrained')
    x = np.arange(len(laws))
    for j,(baseline,color,label) in enumerate((('swiglu64','#D55E00','ISRU / SwiGLU64'),('swiglu160','#009E73','ISRU / SwiGLU160'))):
        ratios=[values[(law,'free_isru')]/values[(law,baseline)] for law in laws]
        ax.bar(x+(j-.5)*.32,ratios,width=.31,color=color,label=label)
    ax.axhline(1,color='#333333',linewidth=.8)
    ax.set(xticks=x,xticklabels=laws,ylabel='Oracle field MSE ratio (lower is better)')
    ax.legend(fontsize=8,ncol=2,loc='upper left')
    save(fig,'synthetic-task-boundaries')
    print('REPORT2_FIGURES_PASS',flush=True)


if __name__ == '__main__':
    main()
