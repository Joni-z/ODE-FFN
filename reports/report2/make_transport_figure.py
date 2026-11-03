"""Bind completed transport diagnostics and plot the saved shape/mass scores."""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
BASE = 'docs/autoresearch/professor-20261103'
records, hashes = [], {}


def read(relative):
    path = REPO / relative
    hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as handle:
        return list(csv.DictReader(handle))


def bind(source, rows, selector, metric, digits=None):
    found = [r for r in rows if all(r[k] == v for k, v in selector.items())]
    assert len(found) == 1, (source, selector)
    value = float(found[0][metric])
    records.append(dict(source=source, selector=selector, metric=metric,
                        value=value, printed_digits=digits))
    return value


shape_file = BASE + '/cycle15-results/shape-mass/shape-and-mass.csv'
mass_file = BASE + '/cycle15-results/precision/component-summary.csv'
shape, mass = read(shape_file), read(mass_file)
families = [('pd_isru_k1', 'ISRU-D', '#0072B2', '-'),
            ('free_isru', 'Free ISRU', '#D55E00', '--'),
            ('swiglu64', 'Swi64 / Muon', '#009E73', '-.')]
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(6.85, 1.5), layout='constrained')
for family, label, color, style in families:
    native, residual, masses = [], [], []
    for step in ['15000', '20000']:
        selector = dict(family=family, checkpoint_step=step)
        digits = 5 if family == 'pd_isru_k1' else None
        native.append(bind(shape_file, shape, selector, 'native_sw2', digits))
        residual.append(bind(shape_file, shape, selector, 'centered_residual_sw2', digits))
        masses.append(bind(mass_file, mass, dict(label=f'pool_{family}_{step}'),
                           'l1_to_analytic_weights', digits))
    for ax, values in zip(axes, [native, residual, masses]):
        ax.plot([15, 20], values, color=color, linestyle=style, marker='o',
                markersize=3, linewidth=1.2, label=label)
        ax.set(xlabel='Updates (k)', xticks=[15, 20])
        ax.grid(alpha=.18)
for ax, title, ylabel in zip(axes,
        ['Native samples', 'Centered residuals', 'Sector probabilities'],
        ['Pooled SW2', 'Residual SW2', 'L1 to equal weights']):
    ax.set(title=title, ylabel=ylabel)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, 1.18),
           ncol=3, frameon=False, fontsize=7.5)
for ext in ['pdf', 'png']:
    fig.savefig(ROOT / f'figures/transport-checkpoints.{ext}', dpi=240,
                bbox_inches='tight', pad_inches=.03)
plt.close(fig)

product_file = BASE + '/cycle14-results-complete/sampling_summary.csv'
product = read(product_file)
for label in ['ISRU-D / adamw 0.0003', 'Free ISRU / adamw 0.0003',
              'Swi192 / adamw 0.0003']:
    bind(product_file, product, dict(dataset='rotated_product_bimodal_16d',
         label=label, checkpoint_step='20000', heun_steps='128'), 'mean', 5)

switch_file = BASE + '/cycle15-results/switches.csv'
switches = read(switch_file)
for mode, cut in [('learned', '0'), ('oracle_prefix', '96'),
                  ('learned_prefix', '96')]:
    bind(switch_file, switches, dict(family='pd_isru_k1', checkpoint_step='20000',
         mode=mode, cut_step=cut), 'endpoint_oracle_mse', 5)

event_file = BASE + '/cycle15-results/sectors/endpoint-sector-error.csv'
events = read(event_file)
crossing_percent = []
for row in events:
    selector = dict(cell=row['cell'])
    total = bind(event_file, events, selector, 'gross_crossings')
    late = bind(event_file, events, selector, 'crossings_after75')
    crossing_percent.append(100 * (1 - late / total))
derived = dict(crossings_by_t075_percent_min=min(crossing_percent),
               crossings_by_t075_percent_max=max(crossing_percent))
for relative in ['cycle15-results/verification.json',
                 'cycle15-results/sectors/verification.json',
                 'cycle14-results-complete/verification.json',
                 'mode-mass-derivation.md']:
    path = REPO / BASE / relative
    hashes[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
(ROOT / 'data/transport-evidence.json').write_text(json.dumps(dict(
    records=records, source_hashes=hashes, derived=derived,
    frozen_models=6, new_model_calls=0, new_training_runs=0,
    status='Editable Report2 draft; finalization belongs to the user.',
    source_commit='5075b3b312d00986648a19ad828bbc9bbfd51de0',
), indent=2) + '\n')
print(json.dumps(dict(source_bound_values=len(records), figures=1, derived=derived)))
