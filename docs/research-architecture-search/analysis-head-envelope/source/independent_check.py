"""Independent Gram-eigendecomposition and safeguarded-Newton saved-head audit."""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import torch
ROOT=Path('/Users/mr.z/ode-workspace-prep/architecture-search/shared_features_20261024_m1')
OUT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary=json.loads((OUT/'summary.json').read_text())
for name,digest in summary['input_sha256'].items():assert sha(Path(name))==digest,name
panels=[];totals=[];prediction_max=0.;compared=[]
def project_distance(y,axes2):
 radius=np.sum(y*y/axes2,axis=-1);distance=np.zeros(len(y))
 for i in np.flatnonzero(radius>1):
  yi=y[i]
  equation=lambda lam:np.sum(axes2*yi*yi/(axes2+lam)**2)-1
  upper=1.
  while equation(upper)>0:upper*=2
  lower=0.;lam=upper/2
  for _ in range(100):
   value=equation(lam)
   if abs(value)<1e-14:break
   if value>0:lower=lam
   else:upper=lam
   derivative=-2*np.sum(axes2*yi*yi/(axes2+lam)**3)
   candidate=lam-value/derivative
   lam=candidate if lower<candidate<upper else (lower+upper)/2
  projected=axes2*yi/(axes2+lam)
  assert abs(np.sum(projected*projected/axes2)-1)<1e-11
  distance[i]=np.sum((yi-projected)**2)
 return distance,radius
sphere,_=project_distance(np.array([[0.,0.],[3.,4.]]),np.array([4.,4.]));np.testing.assert_allclose(sphere,[0.,9.],atol=1e-12,rtol=1e-12)
for dataset,suffix in [('gmm_ring8_2d','v1'),('rotated_product_bimodal_16d','product16')]:
 for arm in ('vector_parent','swiglu_budget','swiglu'):
  study=ROOT/'output'/f'shared_features_{suffix}_20261024_m1';folder=study/'diagnostics'/dataset/arm
  native=json.loads((folder/'summary.json').read_text());cfg=json.loads((study/'configs'/f'{arm}.json').read_text())
  path=study/'runs'/cfg['experiment_id']/dataset/native['model']/'seed000/checkpoints/step_0005000.pt';assert sha(path)==native['checkpoint_sha256']
  state=torch.load(path,weights_only=False,map_location='cpu')['model'];get=lambda k:state[k].double().numpy()
  W,gamma,beta,b=get('output.weight'),get('norm.weight'),get('norm.bias'),get('output.bias');w=len(gamma);center=W@beta+b
  scaled=W*gamma;sum_columns=scaled.sum(axis=1)
  # BB^T, independently expanded using the mean-zero projector identity.
  gram=w*(scaled@scaled.T)-np.outer(sum_columns,sum_columns)
  axes2,rotation=np.linalg.eigh(gram);assert axes2.min()>0
  current=[]
  for t in (.1,.3,.5,.7,.9):
   path=folder/f'common-{t:.1f}.npz';assert sha(path)==native['arrays_sha256'][path.name]
   with np.load(path,allow_pickle=False) as data:
    target=data['oracle'];prediction=data['prediction'];assert target.dtype==np.float64
   distance,radius=project_distance((target-center)@rotation,axes2)
   prad=np.sum(((prediction-center)@rotation)**2/axes2,axis=1);assert prad.max()<1+2e-11;prediction_max=max(prediction_max,float(prad.max()))
   mse=float(np.mean((prediction-target)**2));floor=float(distance.mean()/target.shape[-1]);assert floor<=mse+1e-12
   row=dict(dataset=dataset,arm=arm,time=t,field_mse=mse,envelope_floor=floor,floor_fraction=floor/mse,target_outside_fraction=float(np.mean(radius>1)),prediction_radius_max=float(prad.max()),minimum_semiaxis=float(np.sqrt(axes2.min())),maximum_semiaxis=float(np.sqrt(axes2.max())))
   panels.append(row);current.append(row)
  mse=float(np.mean([r['field_mse'] for r in current]));floor=float(np.mean([r['envelope_floor'] for r in current]));np.testing.assert_allclose(mse,native['field_mse'],atol=1e-12,rtol=1e-12)
  totals.append(dict(dataset=dataset,arm=arm,field_mse=mse,envelope_floor=floor,floor_fraction=floor/mse,target_outside_fraction=float(np.mean([r['target_outside_fraction'] for r in current]))))
maximum=0.
for filename,own in [('per-panel.csv',panels),('aggregate.csv',totals)]:
 rows=list(csv.DictReader((OUT/filename).open()));assert len(rows)==len(own)
 for r in own:
  match=[v for v in rows if v['dataset']==r['dataset'] and v['arm']==r['arm'] and ('time' not in r or float(v['time'])==r['time'])];assert len(match)==1
  for key,value in r.items():
   if key in ('dataset','arm','time'):continue
   delta=abs(float(match[0][key])-value);maximum=max(maximum,delta);np.testing.assert_allclose(float(match[0][key]),value,atol=2e-11,rtol=2e-11)
for name,digest in summary['input_sha256'].items():assert sha(Path(name))==digest,name
result=dict(state='PASS',method='Independent Gram matrix eigendecomposition and safeguarded Newton root; no import of diagnostic implementation.',reviewed_source_sha256=sha(OUT/'source/diagnose_head_envelope.py'),reviewed_protocol_sha256=sha(OUT/'source/protocol-head-envelope.md'),independent_source_sha256=sha(Path(__file__)),panels_checked=30,aggregates=totals,designated_2D_16D_panels=[r for r in panels if r['arm']=='vector_parent' and r['time']==.5],maximum_numeric_difference=maximum,prediction_radius_max=prediction_max,sphere_squared_distances=sphere.tolist(),input_hashes_verified=len(summary['input_sha256']),review=dict(math='Mean-zero normalized z has norm<=sqrt(width); center Wbeta+b includes both affine biases. BB^T=w AA^T-(A1)(A1)^T for A=Wdiag(gamma).',numerics='Saved states promoted to double; positive singular axes verified. Diagnostic bisection upper=max(1,sum(axis^2*y^2)) brackets the root. Independent safeguarded Newton solve confirms all panel/aggregate values.',scope='Conditional fixed-trained-head outer-envelope distance only. Network need not realize the whole ellipsoid; retraining changes the head. Not a global architecture approximation bound or proof of a cause.'))
(OUT/'independent-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(state=result['state'],maximum_numeric_difference=maximum,prediction_radius_max=prediction_max,aggregates=totals),indent=2))
