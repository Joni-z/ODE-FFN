"""Recompute sampled metrics and compare response-by-order effects at fixed budgets."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--previous',required=True);p.add_argument('--output',required=True)
a=p.parse_args();root=Path(a.input);previous=Path(a.previous);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[3]
# Independent NumPy formula copied from followup_20261115/analyze.py.
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):
 x=np.ascontiguousarray(x);return hashlib.sha256(str((x.shape,x.dtype.str)).encode()+x.tobytes()).hexdigest()
def sw2(x,y,d):
 u=np.asarray(x,dtype=float)@np.asarray(d,dtype=float).T
 v=np.asarray(y,dtype=float)@np.asarray(d,dtype=float).T
 return float(np.sqrt(np.mean((np.sort(u,axis=0)-np.sort(v,axis=0))**2)))
def dump(name,value):(out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def table(name,rows):
 with (out/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def row(dataset,family,order,step,variant,s,checkpoint):
 eq=s.get('equivariance') or {}
 return dict(dataset=dataset,response='ISRU' if 'isru' in family else 'tanh',order=order,step=step,variant=variant,field_mse=s['field_mse'],sw2_mean=s['sw2_mean'],sw2_sd=s['sw2_sd'],pooled_sw2=s['pooled_sw2'],sector_l1=s.get('sector_l1'),equivariance_relative_rms=eq.get('relative_rms'),checkpoint_sha256=checkpoint)

rows=[];banks=[];truncations=[];layerrows=[];exp_rows=[];bindings=[];bankhashes={};error=0.
for bfile in sorted((root/'new').glob('gpu*/evaluations/*/binding.json')):
 folder=bfile.parent;b=json.loads(bfile.read_text());cell=b['source_config']['cell'];dataset=cell['dataset'];family=cell['family'];gpu=folder.parents[1].name
 assert json.loads((folder/'COMPLETE.json').read_text())['passed']
 assert json.loads((folder/'native-anchor.json').read_text())['exact']
 with np.load(folder/'banks.npz') as z:arrays={k:z[k].copy() for k in z.files}
 hashes={k:ah(v) for k,v in arrays.items()};assert hashes==b['bank_hashes']
 if dataset in bankhashes:assert hashes==bankhashes[dataset]
 bankhashes[dataset]=hashes;directions=arrays['directions'];target=np.concatenate([arrays[f'target_{i}'] for i in range(8)])
 for s in json.loads((folder/'summary.json').read_text()):
  values=[];scores=[]
  for bank in range(8):
   file=folder/f"{s['step']}-{s['variant']}-b{bank}.npy";x=np.load(file);values.append(x)
   value=sw2(x,arrays[f'target_{bank}'],directions);scores.append(value)
   banks.append(dict(dataset=dataset,response='ISRU' if 'isru' in family else 'tanh',order=8,step=s['step'],variant=s['variant'],bank=bank,sw2=value,sha256=sha(file)))
  pool=sw2(np.concatenate(values),target,directions)
  error=max(error,abs(pool-s['pooled_sw2']),abs(np.mean(scores)-s['sw2_mean']))
  assert abs(pool-s['pooled_sw2'])<1e-10 and abs(np.mean(scores)-s['sw2_mean'])<1e-10
  rows.append(row(dataset,family,8,s['step'],s['variant'],s,s['checkpoint_sha256']))
 diag=root/'new'/gpu/'diagnostics'/cell['id'];done=json.loads((diag/'COMPLETE.json').read_text());assert done['passed'] and done['all_parameters_buffers_unchanged']
 checkpoint=done['checkpoint']['checkpoint_sha256']
 for s in json.loads((diag/'truncation.json').read_text()):
  values=[np.load(diag/f"order{s['order']}-b{i}.npy") for i in range(8)]
  pool=sw2(np.concatenate(values),target,directions);assert abs(pool-s['pooled_sw2'])<1e-10
  error=max(error,abs(pool-s['pooled_sw2']))
  truncations.append(dict(dataset=dataset,response='ISRU' if 'isru' in family else 'tanh',trained_order=8,evaluation_order=s['order'],field_mse=s['field_mse'],pooled_sw2=pool,equivariance_relative_rms=(s['equivariance'] or {}).get('relative_rms'),checkpoint_sha256=checkpoint))
 native=next(r for r in rows if r['dataset']==dataset and r['response']==('ISRU' if 'isru' in family else 'tanh') and r['step']==20000 and r['variant']=='native' and r['order']==8)
 truncations.append(dict(dataset=dataset,response=native['response'],trained_order=8,evaluation_order=8,field_mse=native['field_mse'],pooled_sw2=native['pooled_sw2'],equivariance_relative_rms=native['equivariance_relative_rms'],checkpoint_sha256=checkpoint))
 term=json.loads((diag/'terms.json').read_text());assert term['checkpoint_sha256']==checkpoint
 for x in term['layers']:layerrows.append(dict(dataset=dataset,response=native['response'],**x))
 for x in term['exponential']:exp_rows.append(dict(dataset=dataset,response=native['response'],**x))
 bindings.append(dict(cell=cell,binding_sha256=sha(bfile),diagnostic_sha256=sha(diag/'COMPLETE.json'),term_sha256=sha(diag/'terms.json')))
assert len(bindings)==4
# Reuse three recent K1 cases; verify exact common bank arrays and all pooled metrics.
for bfile in sorted((previous/'evaluations').glob('gpu*/*/binding.json')):
 folder=bfile.parent;b=json.loads(bfile.read_text());cell=b['source_config']['cell']
 if cell['family'] not in ('pd_isru_k1','pd_tanh_k1'):continue
 dataset=cell['dataset'];assert b['bank_hashes']==bankhashes[dataset]
 with np.load(folder/'banks.npz') as z:arrays={k:z[k].copy() for k in z.files}
 assert {k:ah(v) for k,v in arrays.items()}==bankhashes[dataset]
 for s in json.loads((folder/'summary.json').read_text()):
  samples=[np.load(folder/f"{s['step']}-{s['variant']}-b{i}.npy") for i in range(8)]
  pool=sw2(np.concatenate(samples),np.concatenate([arrays[f'target_{i}'] for i in range(8)]),arrays['directions']);assert abs(pool-s['pooled_sw2'])<1e-10
  rows.append(row(dataset,cell['family'],1,s['step'],s['variant'],s,s['checkpoint_sha256']))
  for i,x in enumerate(samples):banks.append(dict(dataset=dataset,response='ISRU' if 'isru' in cell['family'] else 'tanh',order=1,step=s['step'],variant=s['variant'],bank=i,sw2=sw2(x,arrays[f'target_{i}'],arrays['directions']),sha256=sha(folder/f"{s['step']}-{s['variant']}-b{i}.npy")))
# Original Product16 ISRU K1: restore both15k/20k measurements from hashed archives.
folder=root/'product-isru-k1';b=json.loads((folder/'binding.json').read_text());dataset=b['case']['dataset'];assert b['bank_identity']==bankhashes[dataset]
with np.load(folder/'banks.npz') as z:arrays={k:z[k].copy() for k in z.files}
assert {k:ah(v) for k,v in arrays.items()}==bankhashes[dataset]
archive=repo/'docs/autoresearch/professor-20261103/cycle14-results-complete'
sources=list(csv.DictReader((archive/'recomputed.csv').open()));fields=list(csv.DictReader((archive/'field_trajectories.csv').open()))
for step in (15000,20000):
 values=[];scores=[]
 for i in range(8):
  file=folder/f'generated_{step}_b{i}_h128.npy'
  source=next(r for r in sources if r['cell']==b['case']['id'] and int(r['checkpoint_step'])==step and int(r['bank'])==i and int(r['heun_steps'])==128)
  assert sha(file)==source['generated_sha256'];x=np.load(file);values.append(x);s=sw2(x,arrays[f'target_{i}'],arrays['directions']);scores.append(s)
  assert abs(s-float(source['numpy_sw2']))<1e-10
  banks.append(dict(dataset=dataset,response='ISRU',order=1,step=step,variant='native',bank=i,sw2=s,sha256=sha(file)))
 field=next(r for r in fields if r['cell']==b['case']['id'] and int(r['checkpoint_step'])==step)
 s=dict(field_mse=float(field['numpy_oracle_mse']),sw2_mean=float(np.mean(scores)),sw2_sd=float(np.std(scores,ddof=1)),pooled_sw2=sw2(np.concatenate(values),np.concatenate([arrays[f'target_{i}'] for i in range(8)]),arrays['directions']))
 rows.append(row(dataset,'pd_isru_k1',1,step,'native',s,field['checkpoint_sha256']))
assert len(rows)==24
# All observed source checkpoints must match the registered K1 source identity.
anchors=json.loads(Path(__file__).with_name('anchors.json').read_text())
for r in rows:
 if r['order']==1:
  anchor=next(x for x in anchors if x['dataset']==r['dataset'] and x['family']==('pd_isru_k1' if r['response']=='ISRU' else 'pd_tanh_k1'))
  assert anchor['checkpoint_sha256'][str(r['step'])]==r['checkpoint_sha256']
contrasts=[];interaction=[]
for dataset in bankhashes:
 for step in (15000,20000):
  for response in ('ISRU','tanh'):
   one=next(r for r in rows if r['dataset']==dataset and r['step']==step and r['response']==response and r['variant']=='native' and r['order']==1)
   eight=next(r for r in rows if r['dataset']==dataset and r['step']==step and r['response']==response and r['variant']=='native' and r['order']==8)
   vector={k:np.array([r['sw2'] for r in sorted(banks,key=lambda r:r['bank']) if r['dataset']==dataset and r['step']==step and r['response']==response and r['order']==k and r['variant']=='native']) for k in (1,8)}
   delta=vector[8]-vector[1];assert len(delta)==8
   contrasts.append(dict(dataset=dataset,step=step,response=response,mse_k8_minus_k1=eight['field_mse']-one['field_mse'],pool_k8_minus_k1=eight['pooled_sw2']-one['pooled_sw2'],paired_bank_sw2_delta=delta.tolist(),improved_banks=int(np.sum(delta<0))))
  i=next(r for r in contrasts if r['dataset']==dataset and r['step']==step and r['response']=='ISRU');t=next(r for r in contrasts if r['dataset']==dataset and r['step']==step and r['response']=='tanh')
  interaction.append(dict(dataset=dataset,step=step,definition='(K8-K1)_ISRU - (K8-K1)_tanh; negative favors ISRU order benefit',mse=i['mse_k8_minus_k1']-t['mse_k8_minus_k1'],pooled_sw2=i['pool_k8_minus_k1']-t['pool_k8_minus_k1']))
table('metrics.csv',rows);table('banks.csv',banks);table('truncation.csv',truncations)
dump('order-contrasts.json',contrasts);dump('interaction.json',interaction);dump('layer-terms.json',layerrows);dump('exponential.json',exp_rows)
dump('verification.json',dict(passed=True,metric_rows=len(rows),bank_rows=len(banks),new_cells=4,matched_k1_cases=4,max_gpu_numpy_error=error,k1_anchor_hashes_match=True,all_bank_hashes_match=True,source_sha256=sha(__file__),bindings=bindings))
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
for i,dataset in enumerate(('gmm_ring8_2d','rotated_product_bimodal_16d')):
 for j,metric in enumerate(('field_mse','pooled_sw2')):
  ax=axes[i,j]
  for response,color in [('ISRU','#0072B2'),('tanh','#D55E00')]:
   for step,style in [(15000,'--'),(20000,'-')]:
    y=[next(r[metric] for r in rows if r['dataset']==dataset and r['response']==response and r['order']==k and r['step']==step and r['variant']=='native') for k in (1,8)]
    ax.plot([1,8],y,style,marker='o',color=color,label=f'{response}/{step//1000}k')
  ax.set(xticks=[1,8],xlabel='Taylor order',ylabel=metric,title='Ring8' if i==0 else 'Product16')
axes[0,0].legend(fontsize=7);fig.savefig(out/'factorial.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
for i,dataset in enumerate(('gmm_ring8_2d','rotated_product_bimodal_16d')):
 for j,response in enumerate(('ISRU','tanh')):
  ax=axes[i,j]
  for r in layerrows:
   if r['dataset']==dataset and r['response']==response:ax.semilogy(range(1,9),r['term_rms_over_input'],marker='.',label=r['layer'].replace('blocks.','b').replace('.ffn.','.'))
  ax.set(title=('Ring8' if i==0 else 'Product16')+'/'+response,xlabel='Taylor term',ylabel='RMS(term) / RMS(input)',xticks=range(1,9))
axes[0,0].legend(fontsize=6,ncol=2);fig.savefig(out/'taylor-terms.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
for ax,dataset in zip(axes,('gmm_ring8_2d','rotated_product_bimodal_16d')):
 for response in ('ISRU','tanh'):
  sub=sorted([r for r in truncations if r['dataset']==dataset and r['response']==response],key=lambda r:r['evaluation_order'])
  ax.plot([r['evaluation_order'] for r in sub],[r['pooled_sw2'] for r in sub],marker='o',label=response)
 ax.set(title='Ring8' if dataset=='gmm_ring8_2d' else 'Product16',xlabel='Evaluation order, K8-trained weights',ylabel='Pooled SW2',xticks=[1,2,4,8]);ax.legend()
fig.savefig(out/'truncation.png',dpi=180);plt.close(fig)
# Ring8 case study: the same bank and first2048 source samples for each panel.
fig,axes=plt.subplots(2,3,figsize=(9,6),layout='constrained')
for i,response in enumerate(('ISRU','tanh')):
 family='isru' if response=='ISRU' else 'tanh'
 old=next(f.parent for f in (previous/'evaluations').glob('gpu*/*/binding.json') if json.loads(f.read_text())['source_config']['cell']['dataset']=='gmm_ring8_2d' and json.loads(f.read_text())['source_config']['cell']['family']==f'pd_{family}_k1')
 new=next(f.parent for f in (root/'new').glob('gpu*/evaluations/*/binding.json') if json.loads(f.read_text())['source_config']['cell']['dataset']=='gmm_ring8_2d' and json.loads(f.read_text())['source_config']['cell']['family']==f'pd_{family}_k8')
 with np.load(new/'banks.npz') as z:limit=float(np.max(np.abs(z['target_0'])))*1.1
 for j,(folder,order,variant) in enumerate(((old,1,'native'),(new,8,'native'),(new,8,'c8'))):
  x=np.load(folder/f'20000-{variant}-b0.npy')[:2048]
  metric=next(r for r in rows if r['dataset']=='gmm_ring8_2d' and r['response']==response and r['order']==order and r['step']==20000 and r['variant']==variant)
  ax=axes[i,j];ax.scatter(x[:,0],x[:,1],s=2,alpha=.35,color='#0072B2' if response=='ISRU' else '#D55E00')
  ax.set(xlim=(-limit,limit),ylim=(-limit,limit),aspect='equal',title=f"{response} K{order} {variant}\npooled SW2={metric['pooled_sw2']:.4f}")
fig.savefig(out/'ring-samples.png',dpi=180);plt.close(fig)
print(json.dumps(dict(passed=True,rows=len(rows),max_error=error,output=str(out))))
