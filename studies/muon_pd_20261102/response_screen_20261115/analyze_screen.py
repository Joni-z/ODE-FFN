"""Independent NumPy recomputation and preregistered response selection."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):
 x=np.ascontiguousarray(x);return hashlib.sha256(str((x.shape,x.dtype.str)).encode()+x.tobytes()).hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def sw2(x,y,d):
 a=np.sort(np.asarray(x,dtype=float)@np.asarray(d,dtype=float).T,axis=0)
 b=np.sort(np.asarray(y,dtype=float)@np.asarray(d,dtype=float).T,axis=0)
 return float(np.sqrt(np.mean((a-b)**2)))
def table(path,rows):
 with Path(path).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def collect(root,out,split="selection",expected_cases=36):
 rows=[];banks=[];maximum=0.;checks=[]
 for file in sorted((root/split).glob('*/binding.json')):
  folder=file.parent;b=json.loads(file.read_text());cfg=b['source_config'];c=cfg['cell'];asset=b['assets']
  done=json.loads((folder/'COMPLETE.json').read_text());assert done['passed'] and not done['witness']
  assert b['split']==split and asset['split']==split
  assert sha(asset['file'])==asset['sha256']
  with np.load(asset['file']) as z:arrays={k:z[k].copy() for k in z.files}
  assert {k:ah(v) for k,v in arrays.items()}==asset['arrays']
  anchor=json.loads((folder/'native-anchor.json').read_text());assert anchor['exact']
  for s in json.loads((folder/'summary.json').read_text()):
   assert sha(Path(b['source_folder'])/'checkpoints'/f"step_{s['step']:07d}.pt")==s['checkpoint_sha256']
   assert sha(Path(b['source_folder'])/'config.json')==s['config_sha256']
   pred=np.load(folder/f"field-{s['step']}.npy");mse=float(np.mean((pred.astype(float)-arrays['field_oracle'].astype(float))**2))
   samples=[np.load(folder/f"{s['step']}-b{i}.npy") for i in range(8)]
   scores=[sw2(x,arrays[f'target_{i}'],arrays['directions']) for i,x in enumerate(samples)]
   pool=sw2(np.concatenate(samples),np.concatenate([arrays[f'target_{i}'] for i in range(8)]),arrays['directions'])
   error=max(abs(mse-s['field_mse']),abs(pool-s['pooled_sw2']),float(np.max(np.abs(np.array(scores)-s['bank_sw2']))))
   maximum=max(maximum,error);assert error<1e-10
   rows.append(dict(id=c['id'],dataset=c['dataset'],response=c['response'],order=c['order'],role=c['role'],step=s['step'],field_mse=mse,pooled_sw2=pool,sw2_mean=float(np.mean(scores)),sw2_sd=float(np.std(scores,ddof=1)),sector_l1=s.get('sector_l1'),gap_error_max=s.get('gap_error_max'),checkpoint_sha256=s['checkpoint_sha256']))
   for i,score in enumerate(scores):banks.append(dict(id=c['id'],step=s['step'],bank=i,sw2=score,sample_sha256=sha(folder/f"{s['step']}-b{i}.npy")))
  checks.append(dict(id=c['id'],binding_sha256=sha(file),completion_sha256=sha(folder/'COMPLETE.json')))
 assert len(checks)==expected_cases and len(rows)==expected_cases*(2 if split=='selection' else 1)
 table(out/(split+'-metrics.csv'),rows);table(out/(split+'-banks.csv'),banks)
 dump(out/('verification.json' if split=='selection' else 'confirmation-verification.json'),dict(passed=True,cells=len(checks),rows=len(rows),max_numpy_difference=maximum,bindings=checks,source_sha256=sha(__file__)))
 return rows

def shortlist(rows,latency):
 data=('gmm_ring8_2d','rotated_product_bimodal_16d');names=('isru','tanh','arctan','softsign','erf','silu','gelu','linear');rankings=[]
 at=lambda name,k,d:next(x for x in rows if x['response']==name and x['order']==k and x['dataset']==d and x['step']==20000)
 for k in (1,8):
  for name in names:
   eligible=True;ratios=[]
   for d in data:
    r=at(name,k,d);old=[at(n,k,d) for n in ('isru','tanh')]
    eligible &= r['field_mse']<=max(x['field_mse'] for x in old)
    ratios.append(r['pooled_sw2']/min(x['pooled_sw2'] for x in old))
   times=[x['median_seconds_per_forward'] for x in latency if x['response']==name and x['order']==k]
   rankings.append(dict(response=name,order=k,eligible=bool(eligible),worst_relative_sw2=max(ratios),dataset_ratios=dict(zip(data,ratios)),latency_mean=float(np.mean(times))))
 eligible=sorted([r for r in rankings if r['order']==8 and r['eligible']],key=lambda r:(r['worst_relative_sw2'],r['latency_mean'],names.index(r['response'])))
 # Reorder numerical ties only; use registered latency/family tie breaks.
 for i in range(len(eligible)):
  for j in range(i+1,len(eligible)):
   if abs(eligible[j]['worst_relative_sw2']-eligible[i]['worst_relative_sw2'])<1e-10 and (eligible[j]['latency_mean'],names.index(eligible[j]['response']))<(eligible[i]['latency_mean'],names.index(eligible[i]['response'])):
    eligible[i],eligible[j]=eligible[j],eligible[i]
 selected=[x['response'] for x in eligible[:2]]
 return dict(selected_responses=selected,ranking=rankings,rule='registered worst-task pooled SW2 ratio, MSE eligibility; selection arrays only',selected_before_confirmation=True)

def render(rows,out,latency):
 names=('isru','tanh','arctan','softsign','erf','silu','gelu','linear')
 fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
 for i,d in enumerate(('gmm_ring8_2d','rotated_product_bimodal_16d')):
  for j,metric in enumerate(('field_mse','pooled_sw2')):
   ax=axes[i,j]
   for order,offset,color in [(1,-.18,'#0072B2'),(8,.18,'#D55E00')]:
    y=[next(r[metric] for r in rows if r['dataset']==d and r['response']==n and r['order']==order and r['step']==20000) for n in names]
    ax.bar(np.arange(8)+offset,y,width=.36,color=color,label=f'K{order}')
   ax.set(xticks=np.arange(8),xticklabels=names,ylabel=metric,title=('Ring8' if i==0 else 'Product16'))
   ax.tick_params(axis='x',rotation=35);ax.legend()
 fig.savefig(out/'response-comparison.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for ax,d in zip(axes,('gmm_ring8_2d','rotated_product_bimodal_16d')):
  for n in names:
   points=[(x['median_seconds_per_forward']*1000,next(r['pooled_sw2'] for r in rows if r['id']==x['id'] and r['step']==20000),x['order']) for x in latency if x['dataset']==d and x['response']==n]
   points.sort(key=lambda x:x[2]);ax.plot([x[0] for x in points],[x[1] for x in points],marker='o',label=n)
   for x,y,k in points:ax.annotate(str(k),(x,y),xytext=(3,3),textcoords='offset points',fontsize=7)
  for x in latency:
   if x['dataset']==d and x['order']==0:
    y=next(r['pooled_sw2'] for r in rows if r['id']==x['id'] and r['step']==20000)
    ax.scatter(x['median_seconds_per_forward']*1000,y,marker='x',s=60,color='black')
    ax.annotate(x['response'],(x['median_seconds_per_forward']*1000,y),fontsize=7)
  ax.set(xlabel='CPU milliseconds per batch512 forward',ylabel='Pooled SW2',title='Ring8' if d=='gmm_ring8_2d' else 'Product16')
 axes[1].legend(fontsize=7,ncol=2);fig.savefig(out/'quality-cost.png',dpi=180);plt.close(fig)


def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True);p.add_argument('--output',required=True);p.add_argument('--latency');p.add_argument('--confirmation-only',action='store_true');p.add_argument('--expected-cases',type=int,default=36)
 a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if a.confirmation_only:
  collect(Path(a.runtime),out,split='confirmation',expected_cases=a.expected_cases);print(json.dumps(dict(passed=True,confirmation_cases=a.expected_cases)));return
 assert a.latency is not None
 rows=collect(Path(a.runtime),out);latency=json.loads(Path(a.latency).read_text())['rows']
 selected=shortlist(rows,latency);selected['verification_sha256']=sha(out/'verification.json');selected['latency_sha256']=sha(a.latency)
 path=out/'shortlist.json';assert not path.exists(),'Do not overwrite a frozen shortlist'
 dump(path,selected);render(rows,out,latency);print(json.dumps(dict(passed=True,selected=selected['selected_responses'])))

if __name__=='__main__':main()
