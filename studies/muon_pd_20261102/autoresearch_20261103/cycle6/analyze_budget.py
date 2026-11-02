"""Matched80k comparisons after endpoint replay and all exact20k prefixes."""
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--prefix',nargs=2,required=True);a=p.parse_args()
out=Path(a.results);study=Path(__file__).resolve().parents[4]/'docs/autoresearch/professor-20261103'
rows=list(csv.DictReader((out/'endpoints.csv').open()));assert len(rows)==8
assert json.loads((out/'verification.json').read_text())['endpoint_steps']==[80000]
prefix=[json.loads(Path(file).read_text()) for file in a.prefix];assert all(x['passed'] and not x['pending'] and len(x['rows'])==4 for x in prefix)
verified={r['cell']:r for receipt in prefix for r in receipt['rows']};assert set(verified)=={r['id'] for r in rows}
parents=[]
for file in [study/'cycle3-results/endpoints.csv',study.parents[1]/'research-muon-pd-2026-11-02/wave2/analysis/endpoints.csv']:
    parents+=list(csv.DictReader(file.open()))
comparisons=[];growth=[]
for r in rows:
    parent=next(x for x in parents if x['id']==r['parent_id'])
    assert r['parent_checkpoint_sha256']==parent['checkpoint_sha256']==verified[r['id']]['parent_checkpoint_sha256']
    growth.append(dict(cell=r['id'],dataset=r['dataset'],family=r['family'],optimizer=r['optimizer'],lr=r['projection_lr'],parent=parent['id'],field_20k=float(parent['replayed_field_mse']),field_80k=float(r['replayed_field_mse']),test_sw2_20k=float(parent['numpy_test_sw2']),test_sw2_80k=float(r['numpy_test_sw2']),exact_20k_prefix=True))
for ds in sorted({r['dataset'] for r in rows}):
    raw=next(r for r in rows if r['dataset']==ds and r['family']=='pd_direct_raw_k1')
    for control in [r for r in rows if r['dataset']==ds and r['family'].startswith('swiglu')]:
        comparisons.append(dict(dataset=ds,raw=raw['id'],ordinary=control['id'],budget=80000,field_relative_improvement=1-float(raw['replayed_field_mse'])/float(control['replayed_field_mse']),test_sw2_relative_change=float(raw['numpy_test_sw2'])/float(control['numpy_test_sw2'])-1))
def save(name,values):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
save('budget-growth.csv',growth);save('raw-versus-ordinary.csv',comparisons)
curves=list(csv.DictReader((out/'curves.csv').open()));milestones=[]
for r in rows:
    cc=[c for c in curves if c['cell']==r['id']]
    for step in [20000,40000,60000,80000]:
        c=next(c for c in cc if int(c['step'])==step);milestones.append(c)
save('registered-milestones.csv',milestones)
fig,axes=plt.subplots(2,2,figsize=(12,7));colors={'pd_direct_raw_k1':'#ee7733','swiglu64':'#228833','swiglu128':'#4477aa','swiglu192':'#aa3377'}
for axs,ds,title in zip(axes,['gmm_ring8_2d','rotated_product_bimodal_16d'],['Ring8','Product16']):
    for family,color in colors.items():
        cc=[c for c in curves if c['dataset']==ds and c['family']==family and int(c['step'])>=5000]
        axs[0].plot([int(c['step']) for c in cc],[float(c['field_mse']) for c in cc],label=family,color=color)
        cc=[c for c in cc if c['validation_sw2']]
        axs[1].plot([int(c['step']) for c in cc],[float(c['validation_sw2']) for c in cc],label=family,color=color)
    for ax in axs:ax.axvline(20000,c='gray',ls=':');ax.grid(alpha=.2);ax.legend(fontsize=8);ax.set_xlabel('Updates')
    axs[0].set(title=title+' — fixed validation field MSE',ylabel='Field MSE',yscale='log')
    axs[1].set(title=title+' — validation sampling trajectory',ylabel='Validation sliced W2')
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'learning.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,ds,title in zip(axes,['gmm_ring8_2d','rotated_product_bimodal_16d'],['Ring8','Product16']):
    cc=[c for c in curves if c['dataset']==ds and c['family']=='pd_direct_raw_k1']
    for key,label in [('p_sigma_max','Mean raw P sigma'),('generator_sigma_max','Mean conditional PD sigma')]:
        ax.plot([int(c['step']) for c in cc],[float(c[key]) for c in cc],label=label)
    ax.set(title=title+' — rawK1 local operators',xlabel='Updates',ylabel='Local matrix norm',yscale='log');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(out/f'raw-operator-history.{ext}',dpi=180,bbox_inches='tight')
(out/'prefix-verification.json').write_text(json.dumps(dict(passed=True,exact_prefixes=len(verified),receipts=prefix),indent=2)+'\n')
print(json.dumps(dict(endpoints=len(rows),exact_prefixes=len(verified),comparisons=comparisons)))
