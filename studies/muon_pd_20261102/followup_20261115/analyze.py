"""Independent NumPy metric checks and curated figures for the15November wave."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
a=p.parse_args();root=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
repo=Path(__file__).resolve().parents[3]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x):
 x=np.ascontiguousarray(x);return hashlib.sha256(str((x.shape,x.dtype.str)).encode()+x.tobytes()).hexdigest()
def sw2(x,y,d):
 u=np.asarray(x,dtype=float)@np.asarray(d,dtype=float).T
 v=np.asarray(y,dtype=float)@np.asarray(d,dtype=float).T
 return float(np.sqrt(np.mean((np.sort(u,axis=0)-np.sort(v,axis=0))**2)))
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def table(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=[];bankrows=[];bindings=[];folders={};banksets={};maxerr=0
for f in sorted((root/'evaluations').glob('gpu*/*/binding.json')):
 folder=f.parent;b=json.loads(f.read_text());cell=b['source_config']['cell'];dataset=cell['dataset'];family=cell['family']
 assert json.loads((folder/'COMPLETE.json').read_text())['passed']
 assert json.loads((folder/'native-anchor.json').read_text())['exact']
 with np.load(folder/'banks.npz') as z:arrays={k:z[k].copy() for k in z.files}
 hashes={k:ah(v) for k,v in arrays.items()};assert hashes==b['bank_hashes']
 if dataset in banksets:assert banksets[dataset]==hashes
 banksets[dataset]=hashes;folders[family,dataset]=folder
 bindings.append(dict(folder=str(folder),binding_sha256=sha(f),summary_sha256=sha(folder/'summary.json'),bank_hashes=hashes,cell=cell))
 recorded=[json.loads(line) for line in (folder/'banks.jsonl').read_text().splitlines()]
 for s in json.loads((folder/'summary.json').read_text()):
  vals=[];samples=[]
  for bank in range(8):
   file=folder/f"{s['step']}-{s['variant']}-b{bank}.npy";x=np.load(file)
   value=sw2(x,arrays[f'target_{bank}'],arrays['directions']);vals.append(value);samples.append(x)
   old=next(r for r in recorded if r['step']==s['step'] and r['variant']==s['variant'] and r['bank']==bank)
   maxerr=max(maxerr,abs(value-old['sw2']));assert abs(value-old['sw2'])<1e-10
   bankrows.append(dict(dataset=dataset,family=family,step=s['step'],variant=s['variant'],bank=bank,sw2=value,generated_sha256=sha(file)))
  pool=sw2(np.concatenate(samples),np.concatenate([arrays[f'target_{i}'] for i in range(8)]),arrays['directions'])
  maxerr=max(maxerr,abs(pool-s['pooled_sw2']));assert abs(pool-s['pooled_sw2'])<1e-10
  row=dict(dataset=dataset,family=family,optimizer=cell['optimizer'],lr=cell['projection_lr'],step=s['step'],variant=s['variant'],field_mse=s['field_mse'],sw2_mean=float(np.mean(vals)),sw2_sd=float(np.std(vals,ddof=1)),pooled_sw2=pool,sector_l1=s.get('sector_l1'),equivariance_relative_rms=(s['equivariance'] or {}).get('relative_rms'),trajectory_pair_rms=s.get('rotated_trajectory_final_rms'),checkpoint_sha256=s['checkpoint_sha256'],cohort='145_seed0_20261115')
  rows.append(row)
# Reuse five completed Product16controls, verifying their source and identical banks.
oldrows=list(csv.DictReader((repo/'docs/autoresearch/professor-20261103/cycle14-results-complete/recomputed.csv').open()))
oldfields=list(csv.DictReader((repo/'docs/autoresearch/professor-20261103/cycle14-results-complete/field_trajectories.csv').open()))
for f in sorted((root/'product16-historical').glob('*/seed000/binding.json')):
 folder=f.parent;b=json.loads(f.read_text());cell=b['case'];dataset=cell['dataset']
 assert b['bank_identity']==banksets[dataset]
 with np.load(folder/'banks.npz') as z:arrays={k:z[k].copy() for k in z.files}
 assert {k:ah(v) for k,v in arrays.items()}==banksets[dataset]
 values=[];samples=[]
 for bank in range(8):
  record=next(r for r in oldrows if r['cell']==cell['id'] and int(r['checkpoint_step'])==20000 and int(r['heun_steps'])==128 and int(r['bank'])==bank)
  file=folder/record['generated_file'];assert sha(file)==record['generated_sha256']
  x=np.load(file);samples.append(x);value=sw2(x,arrays[f'target_{bank}'],arrays['directions']);values.append(value)
  assert abs(value-float(record['numpy_sw2']))<1e-10
  bankrows.append(dict(dataset=dataset,family=cell['family']+'/'+str(cell['projection_lr']),step=20000,variant='historical_native',bank=bank,sw2=value,generated_sha256=sha(file)))
 pool=sw2(np.concatenate(samples),np.concatenate([arrays[f'target_{i}'] for i in range(8)]),arrays['directions'])
 row=dict(dataset=dataset,family=cell['family'],optimizer=cell['optimizer'],lr=cell['projection_lr'],step=20000,variant='historical_native',field_mse=cell['native_field_mse'],sw2_mean=float(np.mean(values)),sw2_sd=float(np.std(values,ddof=1)),pooled_sw2=pool,sector_l1=None,equivariance_relative_rms=None,trajectory_pair_rms=None,checkpoint_sha256=cell['final_checkpoint_sha256'],cohort='145_seed0_20261103')
 rows.append(row);bindings.append(dict(folder=str(folder),binding_sha256=sha(f),bank_hashes=banksets[dataset],cell=cell))
assert len(bindings)==10
paired=[]
for family in ('pd_isru_k1','pd_tanh_k1','free_isru','swiglu64'):
 for step in (15000,20000):
  native=np.array([r['sw2'] for r in bankrows if r['dataset']=='gmm_ring8_2d' and r['family']==family and r['step']==step and r['variant']=='native'])
  c8=np.array([r['sw2'] for r in bankrows if r['dataset']=='gmm_ring8_2d' and r['family']==family and r['step']==step and r['variant']=='c8'])
  assert len(native)==len(c8)==8
  d=c8-native;paired.append(dict(family=family,step=step,c8_minus_native=d.tolist(),mean=float(d.mean()),sd=float(d.std(ddof=1)),improved_banks=int(np.sum(d<0))))
old=list(csv.DictReader((repo/'docs/autoresearch/professor-20261103/cycle13-results/endpoints.csv').open()));replay=[]
for family in ('pd_isru_k1','free_isru','swiglu64'):
 r=next(x for x in rows if x['dataset']=='gmm_ring8_2d' and x['family']==family and x['step']==20000 and x['variant']=='native')
 h=next(x for x in old if x['dataset']=='gmm_ring8_2d' and x['family']==family and x['optimizer']==r['optimizer'] and float(x['projection_lr'])==r['lr'])
 replay.append(dict(family=family,old_field_mse=float(h['replayed_field_mse']),new_field_mse=r['field_mse'],absolute_difference=r['field_mse']-float(h['replayed_field_mse']),old_checkpoint_sha256=h['checkpoint_sha256'],new_checkpoint_sha256=r['checkpoint_sha256']))
table(out/'metrics.csv',rows);table(out/'banks.csv',bankrows);dump(out/'paired-c8.json',paired);dump(out/'replay.json',replay)
dump(out/'verification.json',dict(passed=True,models=len(bindings),metric_rows=len(rows),bank_rows=len(bankrows),max_numpy_gpu_metric_error=maxerr,all_bank_hashes_match=True,all_historical_sample_hashes_match=True,source_sha256=sha(__file__),bindings=bindings))
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
families=['pd_isru_k1','pd_tanh_k1','free_isru','swiglu64'];labels=['ISRU-D','Tanh-D','Free ISRU','SwiGLU/Muon']
fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
for ax,metric,title in zip(axes,['field_mse','pooled_sw2','sector_l1'],['Field MSE','Pooled SW2','Sector mass L1']):
 for offset,variant,color in [(-.18,'native','#0072B2'),(.18,'c8','#D55E00')]:
  y=[next(r[metric] for r in rows if r['dataset']=='gmm_ring8_2d' and r['family']==f and r['step']==20000 and r['variant']==variant) for f in families]
  ax.bar(np.arange(4)+offset,y,.36,label=variant,color=color)
 ax.set(xticks=np.arange(4),xticklabels=labels,title=title);ax.tick_params(axis='x',rotation=20)
axes[0].legend();fig.savefig(out/'c8-comparison.png',dpi=180);plt.close(fig)
folder=folders['pd_isru_k1','gmm_ring8_2d']
fig,axes=plt.subplots(2,2,figsize=(7,6.4),layout='constrained')
for i,step in enumerate((15000,20000)):
 for j,variant in enumerate(('native','c8')):
  x=np.load(folder/f'{step}-{variant}-b0.npy');ax=axes[i,j]
  ax.scatter(x[:,0],x[:,1],s=2,alpha=.35,rasterized=True)
  ax.set(xlim=(-4,4),ylim=(-4,4),aspect='equal',title=f'ISRU-D {step//1000}k / {variant}',xlabel='x1',ylabel='x2')
fig.savefig(out/'isru-samples.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
for ax,variant in zip(axes[:2],('native','c8')):
 with np.load(folder/f'20000-{variant}-paths.npz') as z:
  path=z['path'];pr=z['rotated_start_path'];err=z['pair_rms']
  for i in range(32):ax.plot(path[:,i,0],path[:,i,1],lw=.6,alpha=.65)
 ax.set(title=f'ISRU-D20k / {variant}',aspect='equal',xlim=(-4,4),ylim=(-4,4))
 axes[2].semilogy(np.linspace(0,1,129),np.maximum(err,1e-10),label=variant)
axes[2].set(xlabel='Flow time',ylabel='Rotated-pair trajectory RMS');axes[2].legend()
fig.savefig(out/'isru-trajectories.png',dpi=180);plt.close(fig)
print(json.dumps({'passed':True,'rows':len(rows),'max_metric_error':maxerr,'output':str(out)}))
# Additional read-only interpretation, computed against the same bank identities.
interpretation=root/'interpretation'
if (interpretation/'COMPLETE.json').exists():
 assert json.loads((interpretation/'COMPLETE.json').read_text())['passed']
 oracle=json.loads((interpretation/'oracle.json').read_text())
 assert oracle['bank_hashes']==banksets['gmm_ring8_2d']
 decomp=json.loads((interpretation/'decomposition.json').read_text())
 assert len(decomp)==8 and max(abs(r['identity_residual']) for r in decomp)<1e-10
 for name in ('oracle.json','decomposition.json','COMPLETE.json','self-test.json'):
  (out/('interpretation-'+name)).write_bytes((interpretation/name).read_bytes())
 fig,axes=plt.subplots(1,4,figsize=(11.5,3.2),layout='constrained',sharey=True)
 for ax,family,label in zip(axes,families,labels):
  r=sorted([r for r in decomp if r['family']==family],key=lambda r:r['step'])
  sym=[x['symmetric_orbit_mse'] for x in r];asym=[x['asymmetric_orbit_mse'] for x in r]
  ax.bar([0,1],sym,color='#0072B2',label='Symmetric error')
  ax.bar([0,1],asym,bottom=sym,color='#D55E00',label='Asymmetric error')
  ax.set(xticks=[0,1],xticklabels=['15k','20k'],title=label)
 axes[0].set_ylabel('MSE on complete C8 orbits');axes[0].legend(fontsize=7)
 fig.savefig(out/'error-decomposition.png',dpi=180);plt.close(fig)
 dump(out/'interpretation-binding.json',dict(passed=True,complete_sha256=sha(interpretation/'COMPLETE.json'),decomposition_sha256=sha(interpretation/'decomposition.json'),oracle_sha256=sha(interpretation/'oracle.json'),scope='Same sample banks; complete-orbit error differs from original finite validation probe MSE'))
