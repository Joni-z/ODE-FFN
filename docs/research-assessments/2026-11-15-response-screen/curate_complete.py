"""Curate the finished immutable screen without touching training or reports."""
import csv,hashlib,json,math,shutil
from pathlib import Path
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=Path('/Users/mr.z/ode-workspace-prep/response-screen-20261115')
RAW=ROOT/'analysis-v1'
OUT=HERE/'results'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(name):return list(csv.DictReader((RAW/name).open()))
def dump(path,x):path.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def main():
 done=json.loads((RAW/'COMPLETE.json').read_text());assert done['passed']
 terminal=json.loads((ROOT/'output/monitor/queue-finished.json').read_text())
 assert not terminal['failed'] and not terminal['unstarted'] and len(terminal['completed'])==36
 assert all(x['exit_code']==0 and x['complete'] for x in terminal['completed'])
 for name in ('verification.json','confirmation-verification.json'):
  assert json.loads((RAW/name).read_text())['passed']
 assert done['shortlist_sha256']==sha(RAW/'shortlist.json')
 OUT.mkdir(exist_ok=True)
 names=('selection-metrics.csv','selection-banks.csv','confirmation-metrics.csv','confirmation-banks.csv',
        'verification.json','confirmation-verification.json','latency.json','shortlist.json',
        'response-comparison.png','quality-cost.png','COMPLETE.json')
 for name in names:shutil.copyfile(RAW/name,OUT/name)
 sel=read('selection-metrics.csv');conf=read('confirmation-metrics.csv')
 model_names=('isru','tanh','arctan','softsign','erf','silu','gelu','linear')
 datasets=('gmm_ring8_2d','rotated_product_bimodal_16d')
 def row(rows,d,name,k):return next(r for r in rows if r['dataset']==d and r['response']==name and int(r['order'])==k and int(r['step'])==20000)
 lines=['# Completed response comparison','',
  'All 36 Mac mini CPU training/selection pipelines and the registered independent',
  'confirmation completed. Each trajectory uses 20k updates; every sampling score',
  'uses eight common banks of 4,096 samples, 256 directions and Heun128. Lower is better.',
  'Selection and confirmation use different prebound arrays.','',
  '[CelebA decision and interpretation](analysis-report.md) |',
  '[Paired-bank analysis](stats-appendix.md) | [Figure catalog](figure-catalog.md)','',
  '## All response families: selection pooled SW2','',
  '| Response | Ring8 K1 | Ring8 K8 | Product16 K1 | Product16 K8 |',
  '|---|---:|---:|---:|---:|']
 for n in model_names:
  values=[float(row(sel,d,n,k)['pooled_sw2']) for d in datasets for k in (1,8)]
  lines.append('| '+n+' | '+' | '.join(f'{x:.5f}' for x in values)+' |')
 for n in ('swiglu','free_isru'):
  values=[]
  for d in datasets:
   label=('swiglu64' if d==datasets[0] else 'swiglu192') if n=='swiglu' else n
   value=float(row(sel,d,label,0)['pooled_sw2']);values.extend([value,value])
  lines.append('| '+n+' (ordinary) | '+' | '.join(f'{x:.5f}' for x in values)+' |')
 lines+=['','Ordinary references have no Taylor order; their values repeat in both columns',
         'to show the common reference for K1 and K8.']
 lines+=['','## Ordinary references and frozen shortlist: independent confirmation','',
  '| Model | Ring8 MSE | Ring8 SW2 | Product16 MSE | Product16 SW2 |',
  '|---|---:|---:|---:|---:|']
 selected=json.loads((RAW/'shortlist.json').read_text())['selected_responses']
 for n,k in [('swiglu',0),('free_isru',0)]+[(n,k) for n in selected for k in (1,8)]:
  values=[]
  for d in datasets:
   label=('swiglu64' if d==datasets[0] else 'swiglu192') if n=='swiglu' else n
   r=row(conf,d,label,k);values += [float(r['field_mse']),float(r['pooled_sw2'])]
  lines.append('| '+n+(f'-K{k}' if k else '')+' | '+' | '.join(f'{x:.6f}' for x in values)+' |')
 lines+=['','Ring8 ordinary SwiGLU is width64/Muon.02; Product16 is width192/AdamW.0003.',
  'All focal responses use AdamW.0003. Their configuration labels remain explicit.',
  'The shortlist was frozen before confirmation and is preserved in shortlist.json.',
  'Responses outside it have selection results only.','',
  '## Same-host forward cost','',
  '| Dataset | Model | Parameters | Batch512 forward, ms | Relative to SwiGLU |',
  '|---|---|---:|---:|---:|']
 lat=json.loads((RAW/'latency.json').read_text())['rows']
 for d in datasets:
  baseline=next(x for x in lat if x['dataset']==d and x['response'].startswith('swiglu'))
  for x in lat:
   if x['dataset']!=d or x['response'] not in selected+['swiglu64','swiglu192','free_isru']:continue
   name=x['response']+(f"-K{x['order']}" if x['order'] else '')
   ratio=x['median_seconds_per_forward']/baseline['median_seconds_per_forward']
   lines.append(f"| {d} | {name} | {x['parameters']} | {1000*x['median_seconds_per_forward']:.3f} | {ratio:.2f}x |")
 lines+=['','These are M1 CPU timings of the complete small velocity network, warmed up,',
  'one CPU thread, median of three 20-forward repeats. They are not image-model',
  'GPU timings. The corresponding image architecture needs its own benchmark.','',
  '## Figures','',
  '![All responses at 20k](results/response-comparison.png)','',
  '![Quality versus measured CPU cost](results/quality-cost.png)','',
  '![Selection and independent confirmation](figures/confirmation-comparison.png)','',
  '[All selection metrics](results/selection-metrics.csv),',
  '[confirmation metrics](results/confirmation-metrics.csv),',
  '[selection rule and shortlist](results/shortlist.json),',
  '[verification](results/verification.json),',
  '[confirmation verification](results/confirmation-verification.json).']
 (HERE/'current-comparison.md').write_text('\n'.join(lines)+'\n')
 bankrows=read('confirmation-banks.csv');effects=[]
 for d in datasets:
  baseline=row(conf,d,'swiglu64' if d==datasets[0] else 'swiglu192',0)
  baseline_banks=np.array([float(x['sw2']) for x in sorted(bankrows,key=lambda x:int(x['bank'])) if x['id']==baseline['id']])
  for n in selected:
   for k in (1,8):
    r=row(conf,d,n,k);v=np.array([float(x['sw2']) for x in sorted(bankrows,key=lambda x:int(x['bank'])) if x['id']==r['id']])
    assert len(v)==len(baseline_banks)==8;delta=v-baseline_banks
    mean=float(delta.mean());sd=float(delta.std(ddof=1));half=float(t.ppf(.975,7)*sd/math.sqrt(8))
    effects.append(dict(dataset=d,response=n,order=k,bank_sw2_delta=delta.tolist(),improved_banks=int((delta<0).sum()),mean=mean,sd=sd,conditional_mean_ci95=[mean-half,mean+half],pooled_sw2_relative_change=float(r['pooled_sw2'])/float(baseline['pooled_sw2'])-1))
 dump(OUT/'paired-confirmation.json',effects)
 figdir=HERE/'figures';figdir.mkdir(exist_ok=True)
 for name in ('response-comparison.png','quality-cost.png'):
  shutil.copyfile(RAW/name,figdir/name)
 fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
 models=[('swiglu',0),('free_isru',0)]+[(n,k) for n in selected for k in (1,8)]
 labels=['SwiGLU','Free ISRU']+[f'{n} K{k}' for n,k in models[2:]]
 for col,d in enumerate(datasets):
  for ri,metric in enumerate(('field_mse','pooled_sw2')):
   ax=axes[ri,col]
   for j,(rows,marker,color,label) in enumerate(((sel,'o','#8a8a8a','Selection'),(conf,'D','#126e82','Confirmation'))):
    values=[]
    for n,k in models:
     nn=('swiglu64' if d==datasets[0] else 'swiglu192') if n=='swiglu' else n
     values.append(float(row(rows,d,nn,k)[metric]))
    ax.plot(np.arange(len(models))+(-.10 if j==0 else .10),values,linestyle='none',marker=marker,color=color,label=label,markersize=6)
   ax.set_xticks(np.arange(len(models)),labels,rotation=25,ha='right')
   ax.set_ylabel('Field MSE (lower is better)' if ri==0 else 'Pooled SW2 (lower is better)')
   ax.set_title(('Ring8' if col==0 else 'Product16')+' — 20k checkpoint')
   ax.grid(axis='y',alpha=.2)
   if ri==0 and col==0:ax.legend(frameon=False)
 fig.savefig(figdir/'confirmation-comparison.png',dpi=180);plt.close(fig)
 dump(HERE/'completion.json',dict(passed=True,training_source='b5fe7aa',analysis_source='0e78ce7',
     queue_terminal=terminal,analysis=done,artifacts={n:sha(OUT/n) for n in names},
     derived_artifacts={p.name:sha(p) for p in figdir.glob('*.png')},
     paired_comparison_sha256=sha(OUT/'paired-confirmation.json'),curator_sha256=sha(__file__)))
 print(json.dumps(dict(passed=True,selected=selected,confirmation_cases=done['confirmation_cases'],output=str(HERE))))

if __name__=='__main__':main()
