"""Audited descriptive tables, orthogonal error decomposition and figures."""
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT, DEST = map(Path, sys.argv[1:3])
OUT, CACHE = ROOT/'output', ROOT/'output/cache'
FAMILIES = ('ode_k2', 'swiglu128', 'swiglu192')
TEACHERS = ('ode_k2', 'swiglu128')
LABELS = {'ode_k2':'ODE', 'swiglu128':'Swi128', 'swiglu192':'Swi192'}
COLORS = {'ode_k2':'#D55E00', 'swiglu128':'#0072B2', 'swiglu192':'#009E73'}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write_csv(name: str, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (DEST/name).open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def components(value: np.ndarray, q: np.ndarray) -> tuple[float, float, float]:
    affine = q@(q.T@value)
    remaining = value-affine
    first, second = float(np.mean(affine**2)), float(np.mean(remaining**2))
    cross = float(2*np.mean(affine*remaining))
    np.testing.assert_allclose(first+second, np.mean(value**2), rtol=2e-11, atol=2e-14)
    return first, second, cross


def save_figure(fig, name: str) -> None:
    fig.savefig(DEST/'figures'/f'{name}.png', dpi=180)
    fig.savefig(DEST/'figures'/f'{name}.pdf')
    plt.close(fig)


assert read(ROOT/'audit.json')['status'] == 'PASS'
DEST.mkdir(exist_ok=True)
(DEST/'figures').mkdir(exist_ok=True)
results, panels, curves, training, p_rows, costs = [], [], [], [], [], []
for folder in sorted((OUT/'runs').iterdir()):
    result = read(folder/'result.json')
    base = {key:result[key] for key in ('teacher','student','position')}
    results.append({**base, **{key:result[key] for key in ('mse','relative_mse','target_energy','jacobian_error','relative_jacobian_error')}})
    panels.extend({**base, **row} for row in result['panels'])
    for line in (folder/'metrics.jsonl').read_text().splitlines():
        row = json.loads(line)
        training.append({**base, **row})
        if 'mse' in row:
            curves.append({**base, **row})
    provenance = read(folder/'provenance.json')
    if result['teacher'] == 'ode_k2' and result['position'] == 0:
        costs.append(dict(student=result['student'], **provenance['costs']))
    for path in sorted(folder.glob('spectrum_*.json')):
        step = int(path.stem.split('_')[-1])
        for layer, data in read(path).items():
            p_rows.append({**base, 'step':step, 'layer':layer,
                           'maximum_singular_value':data['singular_values'][0],
                           'minimum_singular_value':data['singular_values'][-1],
                           'spectral_radius':data['spectral_radius'],
                           'symmetric_minimum':data['symmetric_minimum']})

decomposition, teachers, geometry = [], [], []
for j in range(3):
    z = np.load(CACHE/f'validation-z{j}.npy').astype(np.float64)
    design = np.concatenate([np.ones((len(z),1)), z], axis=1)
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    retained = singular > 1e-12*singular[0]
    q = u[:,retained]
    geometry.append(dict(position=j, rank=int(retained.sum()), columns=65,
                         minimum_retained_singular=float(singular[retained][-1]),
                         maximum_singular=float(singular[0]),
                         retained_condition=float(singular[0]/singular[retained][-1]),
                         orthonormal_max_abs=float(np.max(np.abs(q.T@q-np.eye(q.shape[1]))))))
    for teacher in TEACHERS:
        truth = np.load(CACHE/f'validation-{teacher}-y{j}.npy').astype(np.float64)
        first, second, cross = components(truth, q)
        teachers.append(dict(teacher=teacher, position=j, energy=float(np.mean(truth**2)),
                             affine_energy=first, remaining_energy=second, cross_term=cross,
                             remaining_fraction=second/float(np.mean(truth**2))))
        for family in FAMILIES:
            folder = OUT/'runs'/f'{teacher}--{family}--{j}'
            value = np.load(folder/'validation_0005000.npy').astype(np.float64)
            first, second, cross = components(value-truth, q)
            decomposition.append(dict(teacher=teacher, student=family, position=j,
                affine_error=first, remaining_error=second, cross_term=cross,
                total_error=first+second, affine_error_relative=first/float(np.mean(truth**2)),
                remaining_error_relative=second/float(np.mean(truth**2))))
contrasts = []
for teacher in TEACHERS:
    for j in range(3):
        ode = next(r for r in decomposition if r['teacher']==teacher and r['position']==j and r['student']=='ode_k2')
        for family in FAMILIES[1:]:
            other = next(r for r in decomposition if r['teacher']==teacher and r['position']==j and r['student']==family)
            excess = ode['total_error']-other['total_error']
            affine, remaining = ode['affine_error']-other['affine_error'], ode['remaining_error']-other['remaining_error']
            contrasts.append(dict(teacher=teacher, position=j, generic=family, total_ODE_excess=excess,
                affine_excess=affine, remaining_excess=remaining,
                remaining_fraction_of_positive_excess=remaining/excess if excess>0 else None,
                affine_fraction_of_positive_excess=affine/excess if excess>0 else None))

for name, rows in [('comparison.csv',results),('time-panels.csv',panels),('learning-curves.csv',curves),
                   ('training-metrics.csv',training),('P-trajectories.csv',p_rows),('costs.csv',costs),
                   ('error-components.csv',decomposition),('component-contrasts.csv',contrasts),
                   ('teacher-components.csv',teachers),('input-geometry.csv',geometry)]:
    write_csv(name, rows)

plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
fig, axes = plt.subplots(2,3,figsize=(12.5,7.6),layout='constrained')
for row, teacher in enumerate(TEACHERS):
    for j in range(3):
        ax = axes[row,j]
        for i, family in enumerate(FAMILIES):
            value = next(r for r in decomposition if r['teacher']==teacher and r['position']==j and r['student']==family)
            a, b = 100*value['affine_error_relative'], 100*value['remaining_error_relative']
            ax.bar(i,a,color='#999999',label='Affine error component' if i==0 else None)
            ax.bar(i,b,bottom=a,color='#0072B2',label='Remaining error' if i==0 else None)
            ax.annotate(f'{a+b:.3g}%',(i,a+b),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10)
        ax.set_xticks(range(3),[LABELS[f] for f in FAMILIES])
        ax.set_title(f'{LABELS[teacher]} teacher · position {j+1}')
        ax.set_ylabel('Relative squared error (%)')
        ax.set_ylim(top=max(ax.get_ylim()[1]*1.2,.02))
        ax.grid(axis='y',alpha=.2)
axes[0,0].legend(fontsize=8,loc='upper left')
fig.suptitle('Local FFN fitting on common inputs — all 18 cold students\nStacked components are an empirical projection; they are not error bars',fontsize=13)
save_figure(fig,'local-fitting')

fig, axes = plt.subplots(2,3,figsize=(12.5,7.4),layout='constrained')
for row, teacher in enumerate(TEACHERS):
    for j in range(3):
        ax=axes[row,j]
        for family in FAMILIES:
            rows=sorted([r for r in curves if r['teacher']==teacher and r['position']==j and r['student']==family],key=lambda r:r['step'])
            ax.plot([r['step'] for r in rows],[100*r['relative_mse'] for r in rows],marker='o',color=COLORS[family],label=LABELS[family])
        ax.set_yscale('log');ax.set_xlabel('Update');ax.set_ylabel('Relative squared error (%)')
        ax.set_title(f'{LABELS[teacher]} teacher · position {j+1}');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=9)
fig.suptitle('Every saved local validation trajectory',fontsize=13)
save_figure(fig,'learning-curves')

fig, axes = plt.subplots(2,3,figsize=(12.5,6.7),layout='constrained')
for row, teacher in enumerate(TEACHERS):
    for j in range(3):
        ax=axes[row,j]
        for layer,color in [('ode1','#0072B2'),('ode2','#D55E00')]:
            rows=sorted([r for r in p_rows if r['teacher']==teacher and r['position']==j and r['layer']==layer],key=lambda r:r['step'])
            ax.plot([r['step'] for r in rows],[r['maximum_singular_value'] for r in rows],marker='o',label=layer,color=color)
        ax.set_title(f'ODE student · {LABELS[teacher]} teacher · position {j+1}')
        ax.set_xlabel('Update');ax.set_ylabel('Largest singular value of P');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=9)
fig.suptitle('All twelve trained ODE projection matrices',fontsize=13)
save_figure(fig,'P-trajectories')

receipt = dict(status='COMPLETE',audit_sha256=hashlib.sha256((ROOT/'audit.json').read_bytes()).hexdigest(),
               csv_rows={name:sum(1 for _ in csv.DictReader((DEST/name).open())) for name in [
                   'comparison.csv','time-panels.csv','learning-curves.csv','training-metrics.csv','P-trajectories.csv',
                   'costs.csv','error-components.csv','component-contrasts.csv','teacher-components.csv','input-geometry.csv']},
               figures=['local-fitting','learning-curves','P-trajectories'])
(DEST/'analysis-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
print('Swi teacher excess decomposition',json.dumps([r for r in contrasts if r['teacher']=='swiglu128'],indent=2))
