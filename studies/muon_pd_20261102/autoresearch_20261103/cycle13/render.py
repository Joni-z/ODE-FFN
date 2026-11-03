"""Complete fixed-cohort comparisons after native replay, never partial figures."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--states',nargs=2,required=True);p.add_argument('--samples',nargs=2,required=True);p.add_argument('--runtime',nargs=2,required=True);p.add_argument('--receipts',nargs=2,required=True);p.add_argument('--runtime-commit',required=True);a=p.parse_args()
out=Path(a.results);root=Path(__file__).resolve().parents[4]
read=lambda f:list(csv.DictReader(Path(f).open()))
rows=read(out/'endpoints.csv');curves=read(out/'curves.csv');updates=read(out/'updates.csv')
verification=json.loads((out/'verification.json').read_text())
assert len(rows)==18 and verification['endpoint_steps']==[20000] and verification['max_generated_error']==0
states=[json.loads(Path(f).read_text()) for f in a.states];samples=[json.loads(Path(f).read_text()) for f in a.samples];runtime=[json.loads(Path(f).read_text()) for f in a.runtime]
assert all(s['passed'] and not s['partial'] and len(s['anchors'])==5 and not s['pending'] for s in states)
assert {r['cell'] for s in states for r in s['anchors']}=={r['id'] for r in rows if r['role']=='anchor'}
assert all(s['endpoint_step']==20000 and len(s['rows'])==9 for s in samples)
assert {r['cell'] for s in samples for r in s['rows']}=={r['id'] for r in rows}
costs={r['id']:r for s in runtime for r in s['recorded_endpoints']};assert set(costs)=={r['id'] for r in rows}
def save(name,records):
    keys=list(dict.fromkeys(k for r in records for k in r))
    with (out/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(records)
source_paths=['docs/research-muon-pd-2026-11-02/wave2/analysis/endpoints.csv','docs/research-muon-pd-2026-11-02/wave2/results/endpoints.csv']
source_paths += [f'docs/autoresearch/professor-20261103/cycle{i}-results/endpoints.csv' for i in [3,4,5]]
historical=[r for f in source_paths for r in read(root/f)]
tasks=['gmm_ring8_2d','rotated_product_bimodal_16d'];contrasts=[]
ordinary=lambda r:r['family'].startswith(('swiglu','bilinear')) or r['family'] in ['pd_constant_k1','free_isru']
for ds in tasks:
    cohort=[r for r in rows if r['dataset']==ds];frontier=min(float(r['replayed_field_mse']) for r in historical+rows if r['dataset']==ds and ordinary(r))
    for r in cohort:
        cost=costs[r['id']];assert abs(cost['field_mse']-float(r['replayed_field_mse']))<2e-6 and abs(cost['test_sw2']-float(r['numpy_test_sw2']))<2e-6
        if r['family']!='pd_isru_k1':continue
        same=lambda family:next(x for x in cohort if x['family']==family and x['optimizer']==r['optimizer'])
        constant=same('pd_constant_k1');free=same('free_isru');field=float(r['replayed_field_mse']);gain=1-field/float(constant['replayed_field_mse']);free_gain=1-field/float(free['replayed_field_mse'])
        contrasts.append(dict(cell=r['id'],dataset=ds,optimizer=r['optimizer'],lr=r['projection_lr'],validation_mse=field,test_sw2=float(r['numpy_test_sw2']),constant_field_gain=gain,free_field_gain=free_gain,constant_test_sw2_change=float(r['numpy_test_sw2'])/float(constant['numpy_test_sw2'])-1,free_test_sw2_change=float(r['numpy_test_sw2'])/float(free['numpy_test_sw2'])-1,strongest_ordinary_field=frontier,continuation_eligible=gain>=.05 and free_gain>=.05 and field<frontier))
save('matched-comparisons.csv',contrasts)
save('costs.csv',[dict(cell=r['id'],dataset=r['dataset'],family=r['family'],optimizer=r['optimizer'],lr=r['projection_lr'],native_process_wall_seconds=costs[r['id']]['wall_seconds'],config_sha256=costs[r['id']]['config_sha256']) for r in rows])
save('sample-readouts.csv',[{k:v for k,v in r.items() if k!='axis_gap_mass'} for s in samples for r in s['rows']])
labels={'pd_constant_k1':'Constant D','pd_isru_k1':'ISRU-D','free_isru':'Free ISRU','swiglu64':'Swi64','swiglu128':'Swi128','swiglu192':'Swi192'}
colors=dict(zip(labels,['#4477aa','#ee7733','#228833','#cc6677','#aa3377','#66ccee']))
def label(r):return labels[r['family']]+(' / A' if r['optimizer']=='adamw' else ' / M')
def figsave(fig,name):
    (out/'figures').mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/'figures'/f'{name}.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(13,8))
for col,ds in enumerate(tasks):
    cohort=[r for r in rows if r['dataset']==ds]
    for ax,key,title in [(axs[0,col],'replayed_field_mse','Validation oracle MSE'),(axs[1,col],'numpy_test_sw2','Separate test SW2')]:
        values=[float(r[key]) for r in cohort];ax.bar(np.arange(9),values,color=[colors[r['family']] for r in cohort]);ax.set_xticks(np.arange(9),[label(r) for r in cohort],rotation=60,ha='right',fontsize=8);ax.set(title=ds+' / '+title,yscale='log');ax.grid(axis='y',alpha=.2)
fig.tight_layout();figsave(fig,'endpoints')
for name,data,key,title in [('learning',curves,'field_mse','Validation oracle MSE'),('matrix-norm',updates,'weight_sigma_max','Mean FFN matrix spectral norm (P or ordinary W)')]:
    fig,axs=plt.subplots(1,2,figsize=(13,4.5))
    for ax,ds in zip(axs,tasks):
        for r in [r for r in rows if r['dataset']==ds]:
            cc=[c for c in data if c['cell']==r['id'] and int(c['step'])>=100 and c[key]]
            ax.plot([int(c['step']) for c in cc],[float(c[key]) for c in cc],color=colors[r['family']],linestyle='-' if r['optimizer']=='muon' else '--',label=label(r))
        ax.set(title=ds,xlabel='Native updates',ylabel=title,yscale='log');ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
    fig.tight_layout();figsave(fig,name)
# Fixed cases registered before outcomes: Ring Muon, Product Adam, plus all Swi.
for sample in samples:
    ds=sample['rows'][0]['dataset'];opt='muon' if ds==tasks[0] else 'adamw'
    cohort=[r for r in rows if r['dataset']==ds and (r['optimizer']==opt or r['family'].startswith('swiglu'))]
    assert len(cohort)==6
    if ds==tasks[0]:
        fig,axs=plt.subplots(2,3,figsize=(12,7))
        for ax,r in zip(axs.flat,cohort):
            ex=next(e for e in sample['examples'] if e['cell']==r['id']);readout=next(e for e in sample['rows'] if e['cell']==r['id']);target=np.array(ex['target']);x=np.array(ex['generated']);ax.scatter(*target.T,s=2,alpha=.25,c='black');ax.scatter(*x.T,s=2,alpha=.4,c=colors[r['family']]);ax.set(title=f'{label(r)}; outside {100*readout["outside_plot_fraction"]:.2f}%',xlim=(-4,4),ylim=(-4,4),aspect='equal');ax.grid(alpha=.15)
        fig.suptitle('Ring8: fixed1024 display indices; scores use4096');fig.tight_layout();figsave(fig,'ring-cases')
    else:
        fig,axs=plt.subplots(6,3,figsize=(12,16));ref=sample['reference']
        for axes,r in zip(axs,cohort):
            readout=next(e for e in sample['rows'] if e['cell']==r['id'])
            for ax,j in zip(axes,[0,7,15]):
                ex=next(e for e in sample['examples'] if e['cell']==r['id'] and e['axis']==j);edges=np.array(ex['edges']);ax.stairs(np.array(ex['counts'])/4096/np.diff(edges),edges,color=colors[r['family']]);m=ref['locations'][j];v=ref['variances'][j];x=np.linspace(edges[0],edges[-1],1000);pdf=(np.exp(-(x-m)**2/(2*v))+np.exp(-(x+m)**2/(2*v)))/(2*np.sqrt(2*np.pi*v));ax.plot(x,pdf,color='black',lw=1);ax.set(title=f'{label(r)} / axis {j}\ngap {100*readout["axis_gap_mass"][j]:.2f}% / {100*ref["population_gap_mass"][j]:.2f}% true\noutside {100*ex["outside_fraction"]:.2f}%',xlabel='Latent coordinate',ylabel='Density');ax.grid(alpha=.15)
        fig.suptitle('Product16: all4096 saved samples; fixed Adam/ordinary cases');fig.tight_layout();figsave(fig,'product-cases')
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
files=list(a.states)+list(a.samples)+list(a.runtime)+list(a.receipts)
bindings=dict(runtime_commit=a.runtime_commit,analysis_sha256=sha(__file__),analysis_inputs={str(f):sha(f) for f in files},historical_csvs={f:sha(root/f) for f in source_paths})
(out/'source-bindings.json').write_text(json.dumps(bindings,indent=2)+'\n')
verification.update(exact_native_anchors=10,finite_endpoints=18,continuation_eligible=sum(c['continuation_eligible'] for c in contrasts),all_registered_comparisons=contrasts)
(out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification))
