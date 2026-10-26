"""Export and plot every audited offset-placement outcome."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT, DEST = map(Path, sys.argv[1:3])
TEACHERS = ('flow_map', 'swiglu128')
FAMILIES = ('w_map', 'hidden_bias', 'output_bias', 'swiglu128')
LABELS = {'flow_map': 'Fixed-flow', 'swiglu128': 'SwiGLU128', 'w_map': 'No bias',
          'hidden_bias': 'Hidden bias', 'output_bias': 'Output bias'}
COLORS = dict(zip(FAMILIES, ('#777777', '#D55E00', '#009E73', '#0072B2')))


def read(path: Path) -> dict | list:
    return json.loads(path.read_text())


def csv_file(name: str, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (DEST / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, name: str) -> None:
    for suffix in ('png', 'pdf'):
        fig.savefig(DEST / 'figures' / f'{name}.{suffix}', dpi=180)
    plt.close(fig)


assert read(ROOT / 'audit.json')['status'] == 'PASS'
(DEST / 'figures').mkdir(exist_ok=True)
comparison, panels, curves, training, costs = [], [], [], [], []
for cell in read(ROOT / 'audit-cohort.json'):
    folder = Path(cell['folder'])
    result = read(folder / 'result.json')
    base = {k: cell[k] for k in ('teacher', 'student', 'position', 'reused')}
    comparison.append({**base, **{k: result[k] for k in (
        'mse', 'relative_mse', 'target_energy', 'jacobian_error', 'relative_jacobian_error')}})
    panels.extend({**base, **row} for row in result['panels'])
    for line in (folder / 'metrics.jsonl').read_text().splitlines():
        row = json.loads(line)
        training.append({**base, **row})
        if 'mse' in row:
            curves.append({**base, **row})
    if cell['teacher'] == 'flow_map' and cell['position'] == 0:
        costs.append({'student': cell['student'], **cell['costs']})

offsets, values, matrices, spectra = [], [], [], []
for record in read(ROOT / 'audit-statistics.json'):
    base = {k: record[k] for k in ('teacher', 'student', 'position', 'step')}
    stats = record['statistics']
    offsets.append({**base, **{k: v for k, v in stats.items() if k not in ('offset', 'matrices')}})
    values.extend({**base, 'coordinate': k, 'value': v} for k, v in enumerate(stats['offset']))
    for name, entry in stats['matrices'].items():
        sv = entry['singular_values']
        matrices.append({**base, 'matrix': name, 'maximum_singular_value': sv[0],
                         'minimum_singular_value': sv[-1], 'condition_number': sv[0] / sv[-1]})
        spectra.extend({**base, 'matrix': name, 'index': k, 'singular_value': v}
                       for k, v in enumerate(sv))


def endpoint(teacher: str, position: int, student: str) -> dict:
    return next(row for row in comparison if row['teacher'] == teacher
                and row['position'] == position and row['student'] == student)


decisions = []
for position in range(3):
    errors = {f: endpoint('swiglu128', position, f)['relative_mse'] for f in FAMILIES}
    decisions.append({'position': position, 'hidden_vs_no_bias': errors['hidden_bias'] / errors['w_map'],
                      'hidden_vs_output_bias': errors['hidden_bias'] / errors['output_bias'],
                      'hidden_relative_mse': errors['hidden_bias'],
                      'passes': errors['hidden_bias'] <= .01
                      and errors['hidden_bias'] <= .5 * errors['w_map']
                      and errors['hidden_bias'] <= .5 * errors['output_bias']})
decision = {'substantial_hidden_specific_pattern': sum(row['passes'] for row in decisions) >= 2,
            'positions': decisions, 'interpretation': 'descriptive preregistered criterion; not a significance test'}
(DEST / 'decision.json').write_text(json.dumps(decision, indent=2) + '\n')
exports = {'comparison.csv': comparison, 'time-panels.csv': panels, 'learning-curves.csv': curves,
           'training-metrics.csv': training, 'costs.csv': costs, 'offset-trajectories.csv': offsets,
           'offset-values.csv': values, 'matrix-trajectories.csv': matrices, 'matrix-spectra.csv': spectra}
for name, rows in exports.items():
    csv_file(name, rows)

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(2, 3, figsize=(13, 7.8), layout='constrained')
for i, teacher in enumerate(TEACHERS):
    for j in range(3):
        ax = axes[i, j]
        errors = [100 * endpoint(teacher, j, f)['relative_mse'] for f in FAMILIES]
        ax.bar(range(4), errors, color=[COLORS[f] for f in FAMILIES])
        for k, error in enumerate(errors):
            ax.annotate(f'{error:.4g}%', (k, error), xytext=(0, 4),
                        textcoords='offset points', ha='center', fontsize=9)
        ax.set_xticks(range(4), [LABELS[f] for f in FAMILIES], rotation=15)
        ax.set_ylabel('Relative squared error (%)')
        ax.set_title(f'{LABELS[teacher]} target · position {j + 1}')
        ax.set_ylim(0, max(errors) * 1.25)
        ax.grid(axis='y', alpha=.2)
fig.suptitle('Moving the bias does not close the SwiGLU-target gap\n'
             'All 24 outcomes: 12 new placement fits and 12 historical controls', fontsize=13)
save_figure(fig, 'placement-fitting')

fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), layout='constrained')
for i, teacher in enumerate(TEACHERS):
    for j in range(3):
        ax = axes[i, j]
        for family in FAMILIES:
            rows = sorted([r for r in curves if r['teacher'] == teacher
                           and r['position'] == j and r['student'] == family], key=lambda r: r['step'])
            ax.plot([r['step'] for r in rows], [100 * r['relative_mse'] for r in rows],
                    color=COLORS[family], marker='o', label=LABELS[family])
        ax.set_yscale('log')
        ax.set_title(f'{LABELS[teacher]} target · position {j + 1}')
        ax.set_xlabel('Update')
        ax.set_ylabel('Relative squared error (%)')
        ax.grid(alpha=.2)
axes[0, 0].legend(fontsize=9)
fig.suptitle('Every saved validation trajectory', fontsize=13)
save_figure(fig, 'learning-curves')

fig, axes = plt.subplots(2, 3, figsize=(13, 7.1), layout='constrained')
for i, teacher in enumerate(TEACHERS):
    for j in range(3):
        ax = axes[i, j]
        for family in ('hidden_bias', 'output_bias'):
            rows = sorted([r for r in offsets if r['teacher'] == teacher
                           and r['position'] == j and r['student'] == family], key=lambda r: r['step'])
            ax.plot([r['step'] for r in rows], [r['offset_norm'] for r in rows],
                    color=COLORS[family], marker='o', label=LABELS[family])
        ax.set_title(f'{LABELS[teacher]} target · position {j + 1}')
        ax.set_xlabel('Update')
        ax.set_ylabel('Bias vector L2 norm')
        ax.grid(alpha=.2)
axes[0, 0].legend(fontsize=9)
fig.suptitle('Both bias placements become active during training', fontsize=13)
save_figure(fig, 'offset-trajectories')

receipt = dict(status='COMPLETE', csv_rows={k: len(v) for k, v in exports.items()},
               figures=['placement-fitting', 'learning-curves', 'offset-trajectories'],
               cohort_size=24, new_cells=12, reused_cells=12)
(DEST / 'analysis-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
print(json.dumps(decision))
for teacher in TEACHERS:
    for j in range(3):
        print(teacher, j + 1, [round(100 * endpoint(teacher, j, f)['relative_mse'], 6)
                              for f in FAMILIES])
