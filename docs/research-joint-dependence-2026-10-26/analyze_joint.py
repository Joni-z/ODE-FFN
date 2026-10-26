"""Complete descriptive exports, target geometry and generation case studies."""
from pathlib import Path
import csv
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import torch

ROOT,DEST=map(Path,sys.argv[1:3])
read=lambda p:json.loads(Path(p).read_text())
assert read(ROOT/'audit.json')['status']=='PASS'
FAMILIES=('ode_k2','free_isru','swiglu64','swiglu128','swiglu160')
LAWS=('product','sign_code')
LABELS=dict(zip(FAMILIES,('Original ODE','Free ISRU','SwiGLU64','SwiGLU128','SwiGLU160')))
NAMES={'product':'Independent signs','sign_code':'Dependent sign code'}
COLORS=dict(zip(FAMILIES,('#888888','#009E73','#E69F00','#0072B2','#000000')))


def export(name,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (DEST/name).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def figure(fig,name):
    for suffix in ('png','pdf'):fig.savefig(DEST/'figures'/f'{name}.{suffix}',dpi=180)
    plt.close(fig)


(DEST/'figures').mkdir(exist_ok=True)
comparison=[];panels=[];curves=[];training=[];costs=[];p_records=[];p_values=[]
for cell in read(ROOT/'audit-cohort.json'):
    folder=Path(cell['folder']);result=read(folder/'result.json');base={k:cell[k] for k in ('law','family','reused')}
    comparison.append({**base,**{k:result[k] for k in ('oracle_mse','pair_mse','prediction_rms','jacobian_error','sw2','sw2_reference_floor')}})
    panels.extend({**base,**r} for r in result['panels']);costs.append({**base,**cell['costs']})
    for line in (folder/'metrics.jsonl').read_text().splitlines():
        row=json.loads(line);training.append({**base,**row})
        if 'oracle_mse' in row:curves.append({**base,**row})
    if cell['family']=='ode_k2':
        for step in (0,1000,2500,5000):
            state=torch.load(folder/f'step_{step:07d}.pt',map_location='cpu',weights_only=False)
            for name,value in state['model'].items():
                if not name.endswith('.proj.weight') or '.ffn.ode' not in name:continue
                p=value.double().numpy();sv=np.linalg.svd(p,compute_uv=False)
                np.testing.assert_allclose(sv,torch.linalg.svdvals(value.double()).numpy(),rtol=1e-9,atol=1e-10)
                ev=np.linalg.eigvalsh(.5*(p+p.T))
                b={**base,'step':step,'layer':name.removesuffix('.proj.weight')}
                p_records.append({**b,'maximum_singular_value':sv[0],'minimum_singular_value':sv[-1],
                    'symmetric_minimum':ev[0],'symmetric_maximum':ev[-1],'frobenius_norm':np.linalg.norm(p)})
                p_values.extend({**b,'index':i,'singular_value':s,'symmetric_eigenvalue':e} for i,(s,e) in enumerate(zip(sv,ev)))

geometry=read(ROOT/'target-geometry.json');geo_values=[]
for row in read(ROOT/'target-geometry-values.json'):
    for metric in ('off_diagonal_energy_fraction','normalized_pair_commutator'):
        geo_values.extend(dict(law=row['law'],time=row['time'],metric=metric,index=i,value=v) for i,v in enumerate(row[metric]))
case=read(ROOT/'joint-sample-case-details.json');assert case['status']=='PASS'
case_rows=case['results'];case_details=[]
for row in case['details']:
    case_details.extend(dict(family=row['family'],hamming_distance=i,count=n) for i,n in enumerate(row['hamming_histogram']))
jet=read(ROOT/'interaction/receipt.json');assert jet['status']=='PASS'
(DEST/'interaction-verification.json').write_text(json.dumps(jet,indent=2)+'\n')


def cell(law,family):return next(r for r in comparison if r['law']==law and r['family']==family)


ratios=[]
for law in LAWS:
    ordinary=[cell(law,f) for f in FAMILIES if f.startswith('swiglu')]
    isru=cell(law,'free_isru')
    ratios.append(dict(law=law,field_ratio=isru['oracle_mse']/min(r['oracle_mse'] for r in ordinary),
        jacobian_ratio=isru['jacobian_error']/min(r['jacobian_error'] for r in ordinary),
        sampling_ratio=isru['sw2']/min(r['sw2'] for r in ordinary)))
new=ratios[1]
decision=dict(ordinary_isru_transfer=new['field_ratio']<=1.05 and new['jacobian_ratio']<=1.10 and new['sampling_ratio']<=1.10,
    ratios=ratios,scope='architecture retrained from cold states on each law; no ODE advancement or checkpoint zero-shot claim')
(DEST/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
exports={'comparison.csv':comparison,'time-panels.csv':panels,'learning-curves.csv':curves,'training-metrics.csv':training,
    'costs.csv':costs,'target-geometry.csv':geometry,'target-geometry-values.csv':geo_values,
    'P-trajectories.csv':p_records,'P-spectra.csv':p_values,'joint-sample-case-study.csv':case_rows,'sample-hamming-counts.csv':case_details}
for name,rows in exports.items():export(name,rows)

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(14,7.2),layout='constrained')
for i,law in enumerate(LAWS):
    for j,(metric,label) in enumerate([('oracle_mse','Oracle field MSE'),('jacobian_error','Mean squared Jacobian error'),('sw2','32-step Heun SW2')]):
        ax=axes[i,j];values=[cell(law,f)[metric] for f in FAMILIES]
        ax.scatter(values,range(5),c=[COLORS[f] for f in FAMILIES],s=60)
        for k,v in enumerate(values):ax.annotate(f'{v:.4g}',(v,k),xytext=(5,0),textcoords='offset points',va='center',fontsize=9)
        ax.set_xscale('log');ax.set_xlim(min(values)*.65,max(values)*1.7);ax.set_yticks(range(5),[LABELS[f] for f in FAMILIES]);ax.invert_yaxis()
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(label+' (lower is better)');ax.set_title(NAMES[law]);ax.grid(axis='x',alpha=.2)
fig.suptitle('Changing joint dependence reverses the ordinary ISRU field-fit ordering\nFive new cold fits and five historical Product16 references',fontsize=13)
figure(fig,'all-endpoints')

fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for ax,law in zip(axes,LAWS):
    for family in FAMILIES:
        rows=sorted([r for r in curves if r['law']==law and r['family']==family],key=lambda r:r['step'])
        ax.plot([r['step'] for r in rows],[r['oracle_mse'] for r in rows],marker='o',color=COLORS[family],label=LABELS[family])
    ax.set_yscale('log');ax.set_title(NAMES[law]);ax.set_xlabel('Update');ax.set_ylabel('Oracle field MSE');ax.grid(alpha=.2)
axes[0].legend(fontsize=9);fig.suptitle('Every saved validation measurement',fontsize=13)
figure(fig,'learning-curves')

fig,axes=plt.subplots(1,2,figsize=(11.5,4),layout='constrained')
for law,color in [('product','#777777'),('sign_code','#D55E00')]:
    rows=sorted([r for r in geometry if r['law']==law],key=lambda r:r['time'])
    axes[0].plot([r['time'] for r in rows],[100*r['off_diagonal_mean'] for r in rows],color=color,marker='o',label=NAMES[law])
    axes[1].plot([r['time'] for r in rows],[r['commutator_mean'] for r in rows],color=color,marker='o',label=NAMES[law])
for ax in axes:ax.set_xlabel('FM time');ax.grid(alpha=.2);ax.set_ylim(bottom=0)
axes[0].set_ylabel('Latent off-diagonal energy (%)');axes[1].set_ylabel('Mean normalized pair commutator')
axes[0].legend(fontsize=9);fig.suptitle('Exact oracle geometry on actual interpolated validation states',fontsize=13)
figure(fig,'target-geometry')

fig,axes=plt.subplots(2,2,figsize=(11.5,7.1),layout='constrained')
for i,law in enumerate(LAWS):
    for j,(metric,label) in enumerate([('field_mse','Oracle field MSE'),('jacobian_error','Squared Jacobian error')]):
        ax=axes[i,j]
        for family in FAMILIES:
            rows=sorted([r for r in panels if r['law']==law and r['family']==family],key=lambda r:r['time'])
            ax.plot([r['time'] for r in rows],[r[metric] for r in rows],marker='o',color=COLORS[family],label=LABELS[family])
        ax.set_yscale('log');ax.set_title(NAMES[law]);ax.set_xlabel('FM time');ax.set_ylabel(label);ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8);fig.suptitle('All fixed-time errors',fontsize=13)
figure(fig,'time-panels')

fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
for ax,law in zip(axes,LAWS):
    for name in sorted({r['layer'] for r in p_records}):
        rows=sorted([r for r in p_records if r['law']==law and r['layer']==name],key=lambda r:r['step'])
        ax.plot([r['step'] for r in rows],[r['maximum_singular_value'] for r in rows],marker='o',label=name.replace('raw.blocks.','block ').replace('.ffn.','.'))
    ax.set_title(NAMES[law]);ax.set_xlabel('Update');ax.set_ylabel('Largest singular value of P');ax.grid(alpha=.2)
axes[0].legend(fontsize=8,ncol=2);fig.suptitle('All six original-ODE P trajectories under each law',fontsize=13)
figure(fig,'P-trajectories')

fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
names=[r['family'] for r in case_rows];colors=[COLORS.get(n,'#BBBBBB') for n in names]
labels=[LABELS.get(n,{'target':'Target A','target2':'Target B'}.get(n,n)) for n in names]
axes[0].bar(range(7),[100*r['valid_code_fraction'] for r in case_rows],color=colors);axes[0].set_ylim(0,105);axes[0].set_ylabel('Valid sign-code fraction (%)')
axes[1].bar(range(7),[r['nearest_mean_mse'] for r in case_rows],color=colors);axes[1].set_ylim(bottom=0);axes[1].set_ylabel('Nearest-mean MSE per coordinate')
for ax in axes:ax.set_xticks(range(7),labels,rotation=35,ha='right');ax.grid(axis='y',alpha=.2)
fig.suptitle('Post-outcome generation case study: joint structure beyond SW2\nGaussian noise permits rare sign errors; target batches are empirical references',fontsize=12)
figure(fig,'joint-sample-case')

receipt=dict(status='COMPLETE',new_cells=5,reused_cells=5,csv_rows={name:len(rows) for name,rows in exports.items()},
    figures=['all-endpoints','learning-curves','target-geometry','time-panels','P-trajectories','joint-sample-case'])
(DEST/'analysis-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));print(json.dumps(decision))
print('sign_code_t03',json.dumps([r for r in panels if r['law']=='sign_code' and abs(r['time']-.3)<1e-6]))
