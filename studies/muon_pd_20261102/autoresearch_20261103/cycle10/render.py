"""Full registered radial/static/native comparisons, after endpoint replay."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--states',nargs=2,required=True);p.add_argument('--samples',nargs=2,required=True);p.add_argument('--diagonals',nargs=2,required=True);a=p.parse_args()
out=Path(a.results);rows=list(csv.DictReader((out/'endpoints.csv').open()));curves=list(csv.DictReader((out/'curves.csv').open()))
assert len(rows)==12 and json.loads((out/'verification.json').read_text())['endpoint_steps']==[20000]
states=[json.loads(Path(f).read_text()) for f in a.states];samples=[json.loads(Path(f).read_text()) for f in a.samples];diagonals=[json.loads(Path(f).read_text()) for f in a.diagonals]
assert all(s['passed'] and not s['partial'] and len(s['anchors'])==4 and not s['pending'] for s in states)
assert all(s['endpoint_step']==20000 and len(s['rows'])==6 for s in samples)
assert all(s['passed'] and len(s['bindings'])==6 for s in diagonals)
ids={r['id'] for r in rows};assert ids=={r['cell'] for s in samples for r in s['rows']}=={r['cell'] for s in diagonals for r in s['bindings']}
assert {r['cell'] for s in states for r in s['anchors']}=={r['id'] for r in rows if r['role']=='anchor'}
def save(name,records):
    keys=list(dict.fromkeys(k for r in records for k in r))
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
families=['pd_direct_raw_k1','pd_radial_k1','pd_scaled_raw_k1','swiglu64','swiglu128','swiglu192']
assert set(families)=={r['family'] for r in rows}
names=['Raw K1','Radial D','Static D','SwiGLU64','SwiGLU128','SwiGLU192'];labels=dict(zip(families,names));colors=dict(zip(families,['#4477aa','#ee7733','#228833','#cc6677','#aa3377','#66ccee']))
tasks=['gmm_ring8_2d','rotated_product_bimodal_16d'];comparisons=[]
for ds in tasks:
    native=next(r for r in rows if r['dataset']==ds and r['family']=='pd_direct_raw_k1')
    static=next(r for r in rows if r['dataset']==ds and r['family']=='pd_scaled_raw_k1')
    ordinary=min(float(r['replayed_field_mse']) for r in rows if r['dataset']==ds and r['family'].startswith('swiglu'))
    for r in [x for x in rows if x['dataset']==ds]:
        field=float(r['replayed_field_mse']);raw_gain=1-field/float(native['replayed_field_mse']);static_gain=1-field/float(static['replayed_field_mse'])
        comparisons.append(dict(cell=r['id'],dataset=ds,family=r['family'],validation_mse=field,test_sw2=float(r['numpy_test_sw2']),raw_field_gain=raw_gain,static_field_gain=static_gain,raw_sw2_relative_change=float(r['numpy_test_sw2'])/float(native['numpy_test_sw2'])-1,strongest_native_swiglu_field=ordinary,radial_continuation_eligible=r['family']=='pd_radial_k1' and raw_gain>=.05 and static_gain>=.05,better_field_than_all_native_swiglu=field<ordinary))
save('matched-comparisons.csv',comparisons)
save('diagonal-history.csv',[r for s in diagonals for r in s['rows']])
save('sample-readouts.csv',[{k:v for k,v in r.items() if k!='axis_gap_mass'} for s in samples for r in s['rows']])
for key,ylabel,name in [('field_mse','Validation oracle MSE','learning'),('p_sigma_max','Mean P spectral norm','p-norm'),('term1_rms_over_input','First-term RMS / hidden-input RMS','term-amplitudes')]:
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,ds in zip(axes,tasks):
        for family in families:
            r=next(r for r in rows if r['dataset']==ds and r['family']==family)
            cc=[c for c in curves if c['cell']==r['id'] and int(c['step'])>=1000 and c[key]]
            if cc:ax.plot([int(c['step']) for c in cc],[float(c[key]) for c in cc],label=labels[family],color=colors[family])
        ax.set(title=ds,xlabel='Native updates',ylabel=ylabel,yscale='log');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
product=next(s for s in samples if s['reference'] is not None);ring=next(s for s in samples if s['reference'] is None);ref=product['reference']
fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
for ax,family in zip(axes.flat,families):
    e=next(e for e in ring['examples'] if e['family']==family);r=next(r for r in ring['rows'] if r['family']==family);target=np.array(e['target']);x=np.array(e['generated'])
    ax.scatter(*target.T,s=2,alpha=.25,color='black',label='Target');ax.scatter(*x.T,s=2,alpha=.4,color=colors[family],label=labels[family]);ax.set(title=f'{labels[family]}; outside {100*r["outside_plot_fraction"]:.2f}%',xlim=(-4,4),ylim=(-4,4),aspect='equal');ax.grid(alpha=.15)
fig.suptitle('Ring8: same 1024 display indices; scores use all 4096 samples');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'ring-examples.{ext}',dpi=160,bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(6,3,figsize=(12,15))
for axs,family in zip(axes,families):
    r=next(r for r in product['rows'] if r['family']==family)
    for ax,j in zip(axs,[0,7,15]):
        e=next(e for e in product['examples'] if e['cell']==r['cell'] and e['axis']==j);edges=np.array(e['edges']);ax.stairs(np.array(e['counts'])/e['sample_count']/np.diff(edges),edges,color=colors[family])
        mean=ref['locations'][j];var=ref['variances'][j];x=np.linspace(edges[0],edges[-1],1000);pdf=(np.exp(-(x-mean)**2/(2*var))+np.exp(-(x+mean)**2/(2*var)))/(2*np.sqrt(2*np.pi*var));ax.plot(x,pdf,color='black',lw=1)
        ax.set(title=f'{labels[family]}, axis {j}: gap {100*r["axis_gap_mass"][j]:.2f}% / {100*ref["population_gap_mass"][j]:.2f}% true',xlabel='Latent coordinate',ylabel='Density');ax.grid(alpha=.15)
fig.suptitle('Product16: all registered arms, same 4096 samples');fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'product-examples.{ext}',dpi=160,bbox_inches='tight')
plt.close(fig)
(out/'state-verification.json').write_text(json.dumps(dict(passed=True,exact_anchors=8,receipts=states),indent=2)+'\n')
(out/'sample-readouts.json').write_text(json.dumps(samples,indent=2)+'\n')
(out/'diagonal-bindings.json').write_text(json.dumps([s['bindings'] for s in diagonals],indent=2)+'\n')
print(json.dumps(dict(endpoints=12,exact_anchors=8,comparisons=comparisons)))
