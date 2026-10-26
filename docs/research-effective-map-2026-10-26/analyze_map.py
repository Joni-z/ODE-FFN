"""Complete descriptive analysis of the audited paired-matrix cohort."""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT,DEST=map(Path,sys.argv[1:3])
read=lambda p:json.loads(Path(p).read_text())
assert read(ROOT/'audit.json')['status']=='PASS'
TEACHERS=('flow_map','swiglu128');FAMILIES=('p_map','w_map','swiglu128')
LABELS={'flow_map':'Fixed-flow','swiglu128':'SwiGLU128','p_map':'P core','w_map':'W core'}
COLORS={'p_map':'#D55E00','w_map':'#009E73','swiglu128':'#0072B2'}


def csv_file(name,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (DEST/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def figure(fig,name):
    fig.savefig(DEST/'figures'/f'{name}.png',dpi=180);fig.savefig(DEST/'figures'/f'{name}.pdf');plt.close(fig)


(DEST/'figures').mkdir(exist_ok=True)
comparisons,panels,curves,training,costs=[],[],[],[],[]
cohort=read(ROOT/'audit-cohort.json');matrix_records=read(ROOT/'audit-matrices.json')
for cell in cohort:
    folder=Path(cell['folder']);result=read(folder/'result.json')
    base={k:cell[k] for k in ('teacher','student','position','reused')}
    comparisons.append({**base,**{k:result[k] for k in ['mse','relative_mse','target_energy','jacobian_error','relative_jacobian_error']}})
    panels.extend({**base,**r} for r in result['panels'])
    for line in (folder/'metrics.jsonl').read_text().splitlines():
        row=json.loads(line);training.append({**base,**row})
        if 'mse' in row:curves.append({**base,**row})
    if cell['teacher']=='flow_map' and cell['position']==0:costs.append(dict(student=cell['student'],**cell['costs']))
matrices,origin=[],[]
for r in matrix_records:
    base={k:r[k] for k in ['teacher','position','student','step','reused']}
    origin.append({**base,**r['stats']['origin_jacobian']})
    for name,entry in r['stats']['matrices'].items():
        matrices.append({**base,'matrix':name,'maximum_singular_value':entry['singular_values'][0],
            'minimum_singular_value':entry['singular_values'][-1],
            **{k:v for k,v in entry.items() if k not in ['shape','singular_values']}})
exports={'comparison.csv':comparisons,'time-panels.csv':panels,'learning-curves.csv':curves,
         'training-metrics.csv':training,'costs.csv':costs,'matrix-trajectories.csv':matrices,'origin-jacobians.csv':origin}
for name,rows in exports.items():csv_file(name,rows)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(12.5,7.2),layout='constrained')
for i,teacher in enumerate(TEACHERS):
    for j in range(3):
        ax=axes[i,j]
        values=[100*next(r['relative_mse'] for r in comparisons if r['teacher']==teacher and r['position']==j and r['student']==family) for family in FAMILIES]
        ax.bar(range(3),values,color=[COLORS[f] for f in FAMILIES])
        for k,value in enumerate(values):ax.annotate(f'{value:.4g}%',(k,value),xytext=(0,4),textcoords='offset points',ha='center')
        ax.set_xticks(range(3),[LABELS[f] for f in FAMILIES]);ax.set_ylabel('Relative squared error (%)')
        ax.set_title(f'{LABELS[teacher]} target · position {j+1}');ax.set_ylim(0,max(values)*1.25);ax.grid(axis='y',alpha=.2)
fig.suptitle('Same initial function: P versus direct effective matrices\nAll 18 cells shown; three historical SwiGLU-on-SwiGLU controls are explicitly reused',fontsize=13)
figure(fig,'paired-fitting')

fig,axes=plt.subplots(2,3,figsize=(12.5,7.2),layout='constrained')
for i,teacher in enumerate(TEACHERS):
    for j in range(3):
        ax=axes[i,j]
        for family in FAMILIES:
            rows=sorted([r for r in curves if r['teacher']==teacher and r['position']==j and r['student']==family],key=lambda r:r['step'])
            ax.plot([r['step'] for r in rows],[100*r['relative_mse'] for r in rows],marker='o',label=LABELS[family],color=COLORS[family])
        ax.set_yscale('log');ax.set_title(f'{LABELS[teacher]} target · position {j+1}')
        ax.set_xlabel('Update');ax.set_ylabel('Relative squared error (%)');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=9)
fig.suptitle('Every saved validation trajectory',fontsize=13)
figure(fig,'learning-curves')

fig,axes=plt.subplots(2,3,figsize=(12.5,6.9),layout='constrained')
for i,teacher in enumerate(TEACHERS):
    for j in range(3):
        ax=axes[i,j]
        for name,color in [('P1','#0072B2'),('P2','#D55E00')]:
            rows=sorted([r for r in matrices if r['teacher']==teacher and r['position']==j and r['student']=='p_map' and r['matrix']==name],key=lambda r:r['step'])
            ax.plot([r['step'] for r in rows],[r['maximum_singular_value'] for r in rows],marker='o',label=name,color=color)
        ax.set_title(f'P core · {LABELS[teacher]} target · position {j+1}')
        ax.set_xlabel('Update');ax.set_ylabel('Largest singular value of P');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=9);fig.suptitle('All twelve P trajectories; effective W matrices are also exported',fontsize=13)
figure(fig,'P-trajectories')

receipt=dict(status='COMPLETE',csv_rows={name:len(rows) for name,rows in exports.items()},
             figures=['paired-fitting','learning-curves','P-trajectories'],cohort_size=18,new_cells=15,reused_cells=3)
(DEST/'analysis-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt)
for teacher in TEACHERS:
    for j in range(3):
        rows=[next(r for r in comparisons if r['teacher']==teacher and r['position']==j and r['student']==f) for f in FAMILIES]
        print(teacher,j+1,[round(100*r['relative_mse'],6) for r in rows])
