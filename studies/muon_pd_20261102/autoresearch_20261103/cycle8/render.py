"""All-arm learning, operator and registered sample comparisons after replay."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--states',nargs=2,required=True);p.add_argument('--samples',nargs=2,required=True);a=p.parse_args()
out=Path(a.results);rows=list(csv.DictReader((out/'endpoints.csv').open()));curves=list(csv.DictReader((out/'curves.csv').open()))
assert len(rows)==12 and json.loads((out/'verification.json').read_text())['endpoint_steps']==[20000]
states=[json.loads(Path(f).read_text()) for f in a.states];samples=[json.loads(Path(f).read_text()) for f in a.samples]
assert all(s['passed'] and len(s['rows'])==6 and len(s['anchors'])==2 for s in states)
assert all(len(s['rows'])==6 and s['endpoint_step']==20000 for s in samples)
assert {r['id'] for r in rows}=={r['cell'] for s in states for r in s['rows']}=={r['cell'] for s in samples for r in s['rows']}
def save(name,values):
    keys=list(dict.fromkeys(k for r in values for k in r))
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(values)
readouts=[r for s in samples for r in s['rows']]
save('sample-readouts.csv',[{k:v for k,v in r.items() if k!='axis_gap_mass'} for r in readouts])
save('update-verification.csv',[r for s in states for r in s['rows']])
for s in states:
    for r in s['anchors']:assert r['exact_numerical_state'] and r['exact_samples']
comparisons=[]
for r in rows:
    native=next(x for x in rows if x['dataset']==r['dataset'] and x['family']==r['family'] and x['relaxation']=='none')
    final=next(c for c in curves if c['cell']==r['id'] and c['step']=='20000')
    comparisons.append(dict(cell=r['id'],dataset=r['dataset'],family=r['family'],relaxation=r['relaxation'],field_relative_change=float(r['replayed_field_mse'])/float(native['replayed_field_mse'])-1,test_sw2_relative_change=float(r['numpy_test_sw2'])/float(native['numpy_test_sw2'])-1,p_sigma_max=float(final['p_sigma_max']),p_stable_rank=float(final['p_stable_rank']),conditional_pd_sigma_max=float(final['generator_sigma_max']) if final['generator_sigma_max'] else None,term1_rms_over_input=float(final['term1_rms_over_input'])))
save('matched-comparisons.csv',comparisons)
colors={'none':'#4477aa','shape':'#ee7733','scale':'#228833'}
tasks=['gmm_ring8_2d','rotated_product_bimodal_16d'];families=['scalar_k2','pd_direct_raw_k1']
for key,ylabel,name in [('field_mse','Validation field MSE','learning'),('p_sigma_max','Mean P spectral norm','p-norm'),('p_stable_rank','Mean P stable rank','p-shape'),('term1_rms_over_input','First-term RMS / hidden-input RMS','term-amplitudes')]:
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for axs,ds in zip(axes,tasks):
        for ax,family in zip(axs,families):
            for mode,color in colors.items():
                row=next(r for r in rows if r['dataset']==ds and r['family']==family and r['relaxation']==mode)
                cc=[c for c in curves if c['cell']==row['id'] and int(c['step'])>=1000]
                ax.plot([int(c['step']) for c in cc],[float(c[key]) for c in cc],color=color,label=mode)
            ax.set(title=f'{ds}: {family}',xlabel='Native updates',ylabel=ylabel);ax.grid(alpha=.2);ax.legend()
            if key!='p_stable_rank':ax.set_yscale('log')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
product=next(s for s in samples if s['reference'] is not None);ring=next(s for s in samples if s['reference'] is None);ref=product['reference']
for family in families:
    fig,axes=plt.subplots(3,3,figsize=(11,8))
    for axs,mode in zip(axes,colors):
        row=next(r for r in product['rows'] if r['family']==family and r['relaxation']==mode)
        for ax,j in zip(axs,[0,7,15]):
            e=next(e for e in product['examples'] if e['cell']==row['cell'] and e['axis']==j)
            edges=np.array(e['edges']);ax.stairs(np.array(e['counts'])/e['sample_count']/np.diff(edges),edges,color=colors[mode],label=mode)
            mean=ref['locations'][j];var=ref['variances'][j];x=np.linspace(edges[0],edges[-1],1000)
            pdf=(np.exp(-(x-mean)**2/(2*var))+np.exp(-(x+mean)**2/(2*var)))/(2*np.sqrt(2*np.pi*var))
            ax.plot(x,pdf,color='black',lw=1,label='Population')
            ax.set(title=f'{mode}, axis{j}, gap {100*row["axis_gap_mass"][j]:.2f}% / {100*ref["population_gap_mass"][j]:.2f}% true',xlabel='Latent coordinate',ylabel='Density');ax.legend(fontsize=8)
    fig.suptitle(f'Product16: {family}, all registered20k arms');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'product-{family}.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
fig,axes=plt.subplots(2,4,figsize=(13,6),sharex=True,sharey=True)
for axs,family in zip(axes,families):
    target=np.array(ring['examples'][0]['target']);axs[0].scatter(*target.T,s=2,alpha=.5,color='black');axs[0].set_title(f'Target / {family}')
    for ax,mode in zip(axs[1:],colors):
        row=next(r for r in ring['rows'] if r['family']==family and r['relaxation']==mode)
        e=next(e for e in ring['examples'] if e['cell']==row['cell']);x=np.array(e['generated']);ax.scatter(*x.T,s=2,alpha=.5,color=colors[mode]);ax.set_title(f'{mode}; outside {100*row["outside_plot_fraction"]:.2f}%')
for ax in axes.flat:ax.set(xlim=(-4,4),ylim=(-4,4),aspect='equal');ax.grid(alpha=.15)
fig.suptitle('Ring8 final20k: all arms, same1024display indices; scores use4096')
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'ring-examples.{ext}',dpi=160,bbox_inches='tight')
(out/'state-verification.json').write_text(json.dumps(dict(passed=True,exact_anchors=4,receipts=states),indent=2)+'\n')
(out/'sample-readouts.json').write_text(json.dumps(samples,indent=2)+'\n')
print(json.dumps(dict(endpoints=len(rows),exact_anchors=4,figures=7,comparisons=comparisons)))
