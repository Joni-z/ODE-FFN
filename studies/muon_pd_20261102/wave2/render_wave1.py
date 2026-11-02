"""Reproduce descriptive wave1 figures from curated CSVs; no training."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[3]
DOC = ROOT/'docs/research-muon-pd-2026-11-02'
OUT = DOC/'wave2/analysis/figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'savefig.dpi': 180})


def rows(path):
    with path.open() as stream:
        result = list(csv.DictReader(stream))
    for row in result:
        for k, v in row.items():
            try:
                row[k] = float(v) if v else None
            except ValueError:
                pass
    return result


def save(fig, stem):
    for suffix in ('pdf', 'png'):
        fig.savefig(OUT/(stem+'.'+suffix), bbox_inches='tight')
    plt.close(fig)


endpoints = rows(DOC/'endpoints/all-configurations.csv')
curves = rows(DOC/'wave2/analysis/curves.csv')
updates = rows(DOC/'wave2/analysis/updates.csv')
styles = [('adamw', .0003, '#4477aa', 'AdamW 0.0003'),
          ('adamw', .001, '#66ccee', 'AdamW 0.001'),
          ('muon', .006, '#ee7733', 'Muon 0.006'),
          ('muon', .02, '#cc3311', 'Muon 0.02')]
markers = {'scalar_k2': 'o', 'swiglu128': 's', 'swiglu192': 'D',
           'vector_dp_k2': '^', 'vector_pd_k2': 'v'}
datasets = [('gaussian_aniso_2d', 'Gaussian'), ('gmm_ring8_2d', 'Ring8'),
            ('rotated_product_bimodal_16d', 'Product16')]

fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.4))
for ax, (dataset, title) in zip(axes, datasets):
    subset = [r for r in endpoints if r['dataset'] == dataset]
    for row in subset:
        color = next(c for o, lr, c, _ in styles if o == row['optimizer'] and lr == row['projection_lr'])
        ax.scatter(row['field_mse'], row['test_sw2'], c=color, marker=markers[row['family']],
                   s=45, edgecolors='white', linewidths=.45)
    ax.axhline(subset[0]['test_floor'], color='0.6', linestyle=':', linewidth=1)
    ax.set(xscale='log', xlabel='Validation field MSE (log scale)', title=title)
    ticks = {'gaussian_aniso_2d': [.003, .0035, .004],
             'gmm_ring8_2d': [.012, .014, .016],
             'rotated_product_bimodal_16d': [.02, .04, .08]}[dataset]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(alpha=.15)
axes[0].set_ylabel('Test sliced W2 (lower is better)')
fig.legend([Line2D([], [], color=c, marker='o', linestyle='') for _, _, c, _ in styles],
           [label for _, _, _, label in styles], loc='lower center', bbox_to_anchor=(.5, .08), ncol=4)
fig.legend([Line2D([], [], color='0.3', marker=m, linestyle='') for m in markers.values()],
           ['Scalar ODE', 'SwiGLU128', 'SwiGLU192', 'Vector DP', 'Vector PD'],
           loc='lower center', bbox_to_anchor=(.5, .01), ncol=5)
fig.suptitle('All 36 fixed-budget endpoints; lower-left is better')
fig.subplots_adjust(top=.85, bottom=.29, wspace=.35)
save(fig, '01-all-endpoints')

fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.5))
for axis_row, (dataset, title) in zip(axes, datasets[1:]):
    for method, lr, color, label in styles:
        subset = [r for r in curves if r['dataset'] == dataset and r['family'] == 'scalar_k2'
                  and r['optimizer'] == method and r['lr'] == lr]
        for ax, key in zip(axis_row, ['p_sigma_max', 'clock_outer_saturated_fraction', 'probe_jacobian_max']):
            values = [r for r in subset if r[key] is not None]
            ax.plot([r['step']/1000 for r in values], [r[key] for r in values], color=color, label=label)
            ax.set_xlabel('Updates (thousands)')
            ax.grid(alpha=.15)
    axis_row[0].set_ylabel(title)
    axis_row[0].set_yscale('log')
    axis_row[2].set_yscale('log')
    axis_row[1].set_ylim(0, 1)
for ax, title in zip(axes[0], ['Mean P spectral norm', 'Mean gate saturation fraction', 'Largest fixed-probe layer Jacobian norm']):
    ax.set_title(title)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4)
fig.suptitle('Original scalar ODE: learned scale, controller and local sensitivity')
fig.tight_layout(rect=(0, .06, 1, .96))
save(fig, '02-p-gates-jacobians')

fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.5))
for axis_row, (dataset, title) in zip(axes, datasets[1:]):
    for method, lr, color, label in styles:
        select = lambda r: r['dataset'] == dataset and r['family'] == 'scalar_k2' and r['optimizer'] == method and r['lr'] == lr
        subset = [r for r in curves if select(r) and r['step'] > 0]
        axis_row[0].plot([r['step']/1000 for r in subset], [r['field_mse'] for r in subset], color=color, label=label)
        subset = [r for r in updates if select(r) and r['delta_frobenius'] > 0]
        for ax, key in zip(axis_row[1:], ['delta_frobenius', 'delta_stable_rank']):
            ax.plot([r['step'] for r in subset], [r[key] for r in subset], color=color, label=label)
            ax.set_xscale('log')
            ax.set_xlabel('Updates (log scale)')
    for method, lr, color, label in [('adamw', .0003, '#228833', 'SwiGLU192 AdamW 0.0003'),
                                      ('muon', .02, '#aa3377', 'SwiGLU192 Muon 0.02')]:
        subset = [r for r in curves if r['dataset'] == dataset and r['family'] == 'swiglu192'
                  and r['optimizer'] == method and r['lr'] == lr and r['step'] > 0]
        axis_row[0].plot([r['step']/1000 for r in subset], [r['field_mse'] for r in subset],
                         color=color, linestyle='--', label=label)
    axis_row[0].set(xlabel='Updates (thousands)', ylabel=title, yscale='log')
    axis_row[1].set_yscale('log')
    for ax in axis_row:
        ax.grid(alpha=.15)
for ax, title in zip(axes[0], ['Validation field MSE', 'Mean actual Delta P Frobenius norm', 'Mean actual Delta P stable rank']):
    ax.set_title(title)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3)
fig.suptitle('Fitting and recorded parameter-update geometry')
fig.tight_layout(rect=(0, .10, 1, .96))
save(fig, '03-fitting-and-updates')
print(OUT)
