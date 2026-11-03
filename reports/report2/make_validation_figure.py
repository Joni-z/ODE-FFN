"""Bind the completed Ring8 validation and plot saved checkpoint summaries."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
BASE='docs/autoresearch/professor-20261103/cycle14-results-ring8'
records=[]
hashes={}


def read(relative):
    path=REPO/relative
    hashes[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as f:return list(csv.DictReader(f))


def bind(source,row,selector,metric,digits=None):
    value=float(row[metric])
    records.append(dict(source=source,selector=selector,metric=metric,value=value,printed_digits=digits))
    return value


summary_file=BASE+'/sampling_summary.csv'
field_file=BASE+'/field_trajectories.csv'
pool_file=BASE+'/precision/pooled.csv'
summary=read(summary_file)
fields=read(field_file)
pool=read(pool_file)
families=[('pd_isru_k1','adamw','0.0003','ISRU-D','#0072B2','-'),
          ('free_isru','adamw','0.0003','Free ISRU','#D55E00','--'),
          ('swiglu64','adamw','0.0003','Swi64 / A','#009E73',':'),
          ('swiglu64','muon','0.02','Swi64 / M','#CC79A7','-.'),
          ('swiglu192','muon','0.02','Swi192 / M','#555555','--')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.3,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(6.85,1.62),layout='constrained')
for family,opt,rate,label,color,style in families:
    candidates=[r for r in summary if r['label']=={'pd_isru_k1':'ISRU-D','free_isru':'Free ISRU','swiglu64':'Swi64','swiglu192':'Swi192'}[family]+' / '+opt+' '+f'{float(rate):g}' and r['heun_steps']=='128']
    candidates=sorted(candidates,key=lambda r:int(r['checkpoint_step']))
    assert [int(r['checkpoint_step']) for r in candidates]==[10000,15000,20000]
    cell=candidates[0]['cell']
    field_values=[];sample_values=[]
    for row in candidates:
        step=row['checkpoint_step']
        selector=dict(cell=cell,checkpoint_step=step,heun_steps='128')
        sample_values.append(bind(summary_file,row,selector,'mean',5 if step in ['15000','20000'] and family=='pd_isru_k1' or step=='20000' else None))
        if step=='20000':bind(summary_file,row,selector,'sd',5)
        field=next(r for r in fields if r['cell']==cell and r['checkpoint_step']==step)
        field_values.append(bind(field_file,field,dict(cell=cell,checkpoint_step=step),'numpy_oracle_mse',5 if step=='20000' or (step=='15000' and family=='pd_isru_k1') else None))
    pooled=next(r for r in pool if r['cell']==cell and r['omitted_bank']=='-1')
    bind(pool_file,pooled,dict(cell=cell,omitted_bank='-1'),'sliced_w2',5)
    for ax,values in zip(axes,[field_values,sample_values]):
        ax.plot([10,15,20],values,color=color,linestyle=style,marker='o',markersize=2.7,linewidth=1.1,label=label)
        ax.set(xlabel='Updates (thousands)',xticks=[10,15,20]);ax.grid(alpha=.18)
axes[0].set_ylabel('Oracle MSE')
axes[1].set_ylabel('Mean SW2')
axes[1].legend(frameon=False,fontsize=6.8,ncol=2,loc='upper center',columnspacing=.6,handlelength=1.8)
for ext in ['pdf','png']:
    fig.savefig(ROOT/f'figures/validation-checkpoints.{ext}',dpi=240,bbox_inches='tight',pad_inches=.03)
plt.close(fig)
cost_file=BASE+'/cost.json'
hashes[cost_file]=hashlib.sha256((REPO/cost_file).read_bytes()).hexdigest()
cost=json.loads((REPO/cost_file).read_text())
assert cost['passed']
(ROOT/'data/validation-evidence.json').write_text(json.dumps(dict(
    records=records,source_hashes=hashes,cost_source=cost_file,
    dataset='gmm_ring8_2d',checkpoint_steps=[10000,15000,20000],heun_steps=128,
    banks=8,samples_per_bank=4096,pooled_samples=32768,
    pooled_analysis='Follow-up precision diagnostic; separate from bank mean.',
    primary_result_source='7d4cac9b188632dd69fa10c395f31f0655a4c2b0'
),indent=2)+'\n')
print(json.dumps(dict(source_bound_values=len(records),figures=1,cost_rows=len(cost['rows']))))
