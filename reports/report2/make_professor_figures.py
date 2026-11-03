"""Source-bound additions for the two professor suggestions; no model execution."""
from pathlib import Path
import csv,hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
BASE='docs/autoresearch/professor-20261103'
WAVE='docs/research-muon-pd-2026-11-02/wave2/analysis'
records=[];hashes={}

def read(relative):
    path=REPO/relative;hashes[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    return list(csv.DictReader(path.open()))

def bind(source,row,selector,metric,digits=None):
    records.append(dict(source=source,selector=selector,metric=metric,value=float(row[metric]),printed_digits=digits))
    return float(row[metric])

def unique(rows,selector):
    matches=[r for r in rows if all(r[k]==v for k,v in selector.items())]
    assert len(matches)==1,(selector,len(matches))
    return matches[0]

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
                     'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
source=WAVE+'/updates.csv';updates=read(source)
fig,axes=plt.subplots(1,2,figsize=(6.85,1.9),layout='constrained')
for opt,rate,color,label in [('adamw','0.001','#0072B2','AdamW .001'),('muon','0.02','#D55E00','Muon .02')]:
    cell='rotated_product_bimodal_16d__scalar_k2__'+opt+'_'+rate
    selected=[r for r in updates if r['cell']==cell and int(r['step'])>=10]
    assert len(selected)==9
    for ax,key,title in zip(axes,['delta_stable_rank','weight_sigma_max'],['Update stable rank','Mean P spectral norm']):
        values=[bind(source,r,dict(cell=cell,step=r['step']),key,2 if int(r['step'])==20000 else None) for r in selected]
        ax.plot([int(r['step'])/1000 for r in selected],values,color=color,label=label)
        ax.set(xlabel='Updates (thousands)',ylabel=title);ax.grid(alpha=.2)
for ax in axes:ax.set_ylim(bottom=0)
axes[0].legend(frameon=False,fontsize=7)
for ext in ['pdf','png']:fig.savefig(ROOT/f'figures/professor-muon.{ext}',dpi=240,bbox_inches='tight',pad_inches=.03)
plt.close(fig)
# Bind the two reported scalar quality endpoints and the ordinary reference.
source=WAVE+'/endpoints.csv';endpoints=read(source)
for fam,opt,rate in [('scalar_k2','adamw','0.001'),('scalar_k2','muon','0.02'),('swiglu192','adamw','0.0003')]:
    selector=dict(dataset='rotated_product_bimodal_16d',family=fam,optimizer=opt,projection_lr=rate)
    r=unique(endpoints,selector);bind(source,r,selector,'replayed_field_mse',5)
# Every new table entry, with fixed row ordering and all optimizer settings.
source=BASE+'/cycle13-results/endpoints.csv';endpoints=read(source)
table=[]
for fam,opt in [('pd_constant_k1','adamw'),('pd_constant_k1','muon'),('pd_isru_k1','adamw'),('pd_isru_k1','muon'),('free_isru','adamw'),('free_isru','muon'),('swiglu64',None),('swiglu128',None),('swiglu192',None)]:
    item=dict(family=fam,optimizer=opt)
    for ds,key in [('gmm_ring8_2d','ring'),('rotated_product_bimodal_16d','product')]:
        selector=dict(dataset=ds,family=fam)
        if opt:selector['optimizer']=opt
        r=unique(endpoints,selector)
        item[key]=dict(field=bind(source,r,selector,'replayed_field_mse',5),sw2=bind(source,r,selector,'numpy_test_sw2',5),optimizer=r['optimizer'],lr=r['projection_lr'])
    table.append(item)
# Retain the exact continuation margin instead of interpreting rounding.
source=BASE+'/cycle13-results/matched-comparisons.csv';matched=read(source)
for ds,digits in [('gmm_ring8_2d',2),('rotated_product_bimodal_16d',None)]:
    selector=dict(dataset=ds,optimizer='adamw');r=unique(matched,selector)
    if digits is not None:
        value=bind(source,r,selector,'free_field_gain');records[-1].update(printed_digits=2,printed_scale=100.)
source=BASE+'/cycle13-results/curves.csv';curves=read(source)
fig,axes=plt.subplots(1,2,figsize=(6.85,2.05),layout='constrained')
colors={'pd_constant_k1':'#0072B2','pd_isru_k1':'#D55E00','free_isru':'#009E73','swiglu192':'#333333'}
labels={'pd_constant_k1':'Constant','pd_isru_k1':'ISRU-D','free_isru':'Free ISRU','swiglu192':'Swi192'}
for ax,ds,title in zip(axes,['gmm_ring8_2d','rotated_product_bimodal_16d'],['Ring8','Product16']):
    for fam in colors:
        for opt in (['adamw','muon'] if fam!='swiglu192' else [None]):
            chosen=[r for r in curves if r['dataset']==ds and r['family']==fam and int(r['step'])>=1000 and (opt is None or r['optimizer']==opt)]
            assert len(chosen)==39
            values=[bind(source,r,dict(cell=r['cell'],step=r['step']),'field_mse') for r in chosen]
            ax.plot([int(r['step'])/1000 for r in chosen],values,color=colors[fam],linestyle=':' if opt is None else ('--' if opt=='adamw' else '-'),label=labels[fam]+(' / A' if opt=='adamw' else ' / M' if opt else ''))
    ax.set(title=title,xlabel='Updates (thousands)',ylabel='Validation MSE',yscale='log');ax.grid(alpha=.2)
axes[1].legend(frameon=False,fontsize=6.2,ncol=2)
for ext in ['pdf','png']:fig.savefig(ROOT/f'figures/professor-diagonal.{ext}',dpi=240,bbox_inches='tight',pad_inches=.03)
plt.close(fig)
(ROOT/'data/professor-evidence.json').write_text(json.dumps(dict(records=records,source_hashes=hashes,table=table),indent=2)+'\n')
print(json.dumps(dict(source_bound_values=len(records),figures=2,table_rows=len(table))))
