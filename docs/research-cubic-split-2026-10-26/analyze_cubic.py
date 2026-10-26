"""All-arm descriptive report for the independently audited cubic-field screen."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT, DEST = map(Path, sys.argv[1:3])
read = lambda p: json.loads(Path(p).read_text())
assert read(ROOT/'audit.json')['status'] == 'PASS'
FAMILIES = ('ode_k2', 'cubic_split', 'silu_split', 'free_isru', 'swiglu64', 'swiglu128', 'swiglu160')
GENERIC = ('swiglu64', 'swiglu128', 'swiglu160')
DATASETS = ('gaussian_aniso_2d', 'gmm_ring8_2d', 'rotated_product_bimodal_16d')
LABELS = dict(zip(FAMILIES, ('Original ODE', 'Cubic split', 'SiLU split', 'Free ISRU', 'SwiGLU64', 'SwiGLU128', 'SwiGLU160')))
SHORT = dict(zip(DATASETS, ('Gaussian2D', 'Ring8', 'Product16')))
COLORS = dict(zip(FAMILIES, ('#888888', '#D55E00', '#CC79A7', '#009E73', '#E69F00', '#0072B2', '#000000')))


def csv_file(name: str, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (DEST/name).open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def figure(fig, name: str) -> None:
    fig.savefig(DEST/'figures'/f'{name}.png', dpi=180)
    fig.savefig(DEST/'figures'/f'{name}.pdf')
    plt.close(fig)


(DEST/'figures').mkdir(exist_ok=True)
comparisons, panels, curves, training, costs = [], [], [], [], []
for cell in read(ROOT/'audit-cohort.json'):
    folder=Path(cell['folder']); result=read(folder/'result.json')
    base={k:cell[k] for k in ('dataset','family','reused')}
    comparisons.append({**base, **{k:result[k] for k in ('oracle_mse','pair_mse','prediction_rms','jacobian_error','sw2','sw2_reference_floor')}})
    panels.extend({**base,**row} for row in result['panels'])
    costs.append({**base,**cell['costs']})
    for line in (folder/'metrics.jsonl').read_text().splitlines():
        row=json.loads(line);training.append({**base,**row})
        if 'oracle_mse' in row:curves.append({**base,**row})
spectra, trajectory=[] ,[]
for record in read(ROOT/'audit-statistics.json'):
    base={k:record[k] for k in ('dataset','family','step')}
    for row in record['statistics']:
        if 'P_singular_values' not in row:continue
        sv=row['P_singular_values'];ev=row['P_symmetric_eigenvalues']
        trajectory.append({**base,**{k:v for k,v in row.items() if k not in ('P_singular_values','P_symmetric_eigenvalues')},
            'maximum_singular_value':sv[0],'minimum_singular_value':sv[-1],
            'symmetric_minimum':ev[0],'symmetric_maximum':ev[-1]})
        spectra.extend({**base,'position':row['position'],'index':k,'singular_value':s,'symmetric_eigenvalue':e}
                       for k,(s,e) in enumerate(zip(sv,ev)))


def cell(dataset, family):
    return next(row for row in comparisons if row['dataset']==dataset and row['family']==family)


guards=[]
for dataset in DATASETS:
    candidate=cell(dataset,'cubic_split')
    best={metric:min(cell(dataset,f)[metric] for f in GENERIC) for metric in ('oracle_mse','jacobian_error','sw2')}
    row=dict(dataset=dataset,field_ratio=candidate['oracle_mse']/best['oracle_mse'],
             jacobian_ratio=candidate['jacobian_error']/best['jacobian_error'],
             sampling_ratio=candidate['sw2']/best['sw2'],
             original_ratio=candidate['oracle_mse']/cell(dataset,'ode_k2')['oracle_mse'],
             silu_ratio=candidate['oracle_mse']/cell(dataset,'silu_split')['oracle_mse'],
             free_isru_ratio=candidate['oracle_mse']/cell(dataset,'free_isru')['oracle_mse'])
    row['field_pass']=row['field_ratio']<=(.9 if dataset==DATASETS[2] else 1.05)
    row['jacobian_pass']=row['jacobian_ratio']<=1.10
    row['sampling_pass']=row['sampling_ratio']<=1.10
    row['mechanism_pass']=(row['original_ratio']<=.8 and row['silu_ratio']<=.9 and row['free_isru_ratio']<=.9) if dataset==DATASETS[2] else True
    guards.append(row)
decision=dict(eligible=all(row[k] for row in guards for k in ('field_pass','jacobian_pass','sampling_pass','mechanism_pass')),
              guards=guards,scope='fixed synthetic screen; descriptive operational gates')
(DEST/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
exports={'comparison.csv':comparisons,'time-panels.csv':panels,'learning-curves.csv':curves,
         'training-metrics.csv':training,'costs.csv':costs,'P-trajectories.csv':trajectory,'P-spectra.csv':spectra}
for name,rows in exports.items():csv_file(name,rows)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(3,3,figsize=(14,10.8),layout='constrained')
for j,dataset in enumerate(DATASETS):
    for i,(metric,label) in enumerate([('oracle_mse','Oracle field MSE'),('jacobian_error','Mean squared Jacobian error'),('sw2','Sampling SW2 (32 steps)')]):
        ax=axes[i,j];values=[cell(dataset,f)[metric] for f in FAMILIES]
        ax.bar(range(7),values,color=[COLORS[f] for f in FAMILIES]);ax.set_yscale('log')
        ax.set_xticks(range(7),[LABELS[f] for f in FAMILIES],rotation=55,ha='right',fontsize=8)
        ax.set_ylabel(label);ax.set_title(SHORT[dataset]);ax.grid(axis='y',alpha=.2)
fig.suptitle('Cubic-field screen: every endpoint and strong ordinary control\n15 new fits; original ODE and SwiGLU128 are six historical references',fontsize=13)
figure(fig,'all-endpoints')

fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
for ax,dataset in zip(axes,DATASETS):
    for f in FAMILIES:
        rows=sorted([r for r in curves if r['dataset']==dataset and r['family']==f],key=lambda r:r['step'])
        ax.plot([r['step'] for r in rows],[r['oracle_mse'] for r in rows],color=COLORS[f],marker='o',label=LABELS[f])
    ax.set_yscale('log');ax.set_title(SHORT[dataset]);ax.set_xlabel('Update');ax.set_ylabel('Oracle field MSE');ax.grid(alpha=.2)
axes[0].legend(fontsize=8,ncol=2);fig.suptitle('All saved validation trajectories',fontsize=13)
figure(fig,'learning-curves')

fig,axes=plt.subplots(3,3,figsize=(12,9),layout='constrained')
for i,dataset in enumerate(DATASETS):
    for j in range(3):
        ax=axes[i,j]
        for f in ('cubic_split','silu_split'):
            rows=sorted([r for r in trajectory if r['dataset']==dataset and r['position']==j and r['family']==f],key=lambda r:r['step'])
            ax.plot([r['step'] for r in rows],[r['maximum_singular_value'] for r in rows],color=COLORS[f],marker='o',label=LABELS[f])
        ax.set_title(f'{SHORT[dataset]} · block {j+1}');ax.set_xlabel('Update');ax.set_ylabel('Largest singular value of P');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=9);fig.suptitle('Every tied-generator trajectory; full spectra exported separately',fontsize=13)
figure(fig,'P-trajectories')

receipt=dict(status='COMPLETE',csv_rows={k:len(v) for k,v in exports.items()},
             figures=['all-endpoints','learning-curves','P-trajectories'],new_cells=15,reused_cells=6)
(DEST/'analysis-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));print(json.dumps(decision))
for d in DATASETS:
    for f in FAMILIES:
        r=cell(d,f);print(d,f,r['oracle_mse'],r['jacobian_error'],r['sw2'])
