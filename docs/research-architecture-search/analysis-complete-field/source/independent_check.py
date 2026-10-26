"""Independent NumPy audit of all predeclared saved panels; no model evaluation."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path('/Users/mr.z/ode-workspace-prep/architecture-search/shared_features_20261024_m1')
OUT=Path(__file__).resolve().parents[1]
PROTOCOL=Path('/Users/mr.z/ODE-FFN-report-work/docs/research-architecture-search/protocol-complete-field-diagnostic.md')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inputs={str(PROTOCOL):sha(PROTOCOL)};panels=[];groups=[];references={}
for suffix,dataset,dimension in [('v1','gmm_ring8_2d',2),('product16','rotated_product_bimodal_16d',16)]:
 for arm in ('vector_parent','swiglu_budget','swiglu'):
  folder=ROOT/'output'/f'shared_features_{suffix}_20261024_m1'/'diagnostics'/dataset/arm
  summary=json.loads((folder/'summary.json').read_text());inputs[str(folder/'summary.json')]=sha(folder/'summary.json');values=[]
  for time in (.1,.3,.5,.7,.9):
   path=folder/f'common-{time:.1f}.npz';digest=sha(path);assert digest==summary['arrays_sha256'][path.name];inputs[str(path)]=digest
   with np.load(path) as record:arrays={k:record[k] for k in record.files}
   assert arrays['x'].shape==(512,dimension) and arrays['jacobian'].dtype==np.float64
   key=dataset,time
   if key not in references:references[key]={k:arrays[k] for k in ('x','oracle','oracle_jacobian')}
   for k in references[key]:np.testing.assert_array_equal(arrays[k],references[key][k])
   J=arrays['jacobian'];Jtrue=arrays['oracle_jacobian'];a=1-time;A=np.eye(dimension)+a*J;T=np.eye(dimension)+a*Jtrue
   S=(A+A.swapaxes(-1,-2))/2;K=(A-A.swapaxes(-1,-2))/2
   eigen,basis=np.linalg.eigh(S);Q=(basis*np.maximum(eigen,0)[:,None,:])@basis.swapaxes(-1,-2)
   square=lambda z:np.sum(z*z,axis=(-2,-1))
   total=square(J-Jtrue);skew=square(K)/(a*a);negative=np.sum(np.minimum(eigen,0)**2,axis=-1)/(a*a)
   structural=skew+negative;projected=square((Q-np.eye(dimension))/a-Jtrue)
   cross=2*np.sum((A-Q)*(Q-T),axis=(-2,-1))/(a*a)
   np.testing.assert_allclose(total,structural+projected+cross,atol=2e-11,rtol=2e-11)
   np.testing.assert_allclose(structural,square(A-Q)/(a*a),atol=2e-11,rtol=2e-11)
   assert cross.min()>-2e-10
   target_eigen=np.linalg.eigvalsh((T+T.swapaxes(-1,-2))/2)
   assert np.abs(T-T.swapaxes(-1,-2)).max()<1e-10 and target_eigen.min()>-1e-10
   tolerance=1e-10*np.maximum(1,np.max(np.abs(eigen),axis=-1));violations=eigen[:,0]<-tolerance
   native=next(row['jacobian_error'] for row in summary['rows'] if row['time']==time)
   np.testing.assert_allclose(total.mean(),native,atol=2e-12,rtol=2e-12)
   row=dict(dataset=dataset,arm=arm,time=time,anchors=512,velocity_total=float(total.mean()),velocity_skew=float(skew.mean()),velocity_negative=float(negative.mean()),velocity_structural=float(structural.mean()),velocity_projected=float(projected.mean()),velocity_cross=float(cross.mean()),velocity_projected_gain=float(np.mean(total-projected)),structural_fraction=float(structural.sum()/total.sum()),PSD_violation_fraction=float(violations.mean()),denoiser_total=float(total.mean()*a*a),denoiser_structural=float(structural.mean()*a*a),target_min_eigenvalue=float(target_eigen.min()),target_max_asymmetry=float(np.abs(T-T.swapaxes(-1,-2)).max()),decomposition_max_residual=float(np.max(np.abs(total-structural-projected-cross))))
   panels.append(row);values.append(row)
  aggregate={key:float(np.mean([v[key] for v in values])) for key in ('velocity_total','velocity_skew','velocity_negative','velocity_structural','velocity_projected','velocity_cross','velocity_projected_gain','PSD_violation_fraction','denoiser_total','denoiser_structural')}
  aggregate.update(dataset=dataset,arm=arm,structural_fraction=aggregate['velocity_structural']/aggregate['velocity_total'],projected_gain_fraction=aggregate['velocity_projected_gain']/aggregate['velocity_total'],native_jacobian_error=summary['jacobian_error'])
  np.testing.assert_allclose(aggregate['velocity_total'],summary['jacobian_error'],atol=2e-12,rtol=2e-12);groups.append(aggregate)
result=dict(state='INDEPENDENT_NUMERICAL_PASS_PENDING_CODE_FIGURE_REVIEW',method='Independent NumPy eigh on all30 raw float64 panels. Velocity distances divided by (1-t)^2 before equal-panel averaging; ratio of sums, not average of ratios.',input_sha256=inputs,script_sha256=sha(Path(__file__)),panels=panels,aggregates=groups,designated_panel_checks=[r for r in panels if r['arm']=='vector_parent' and r['time']==.5],limitations='Pointwise PSD projection only; not an integrable Jacobian map, a trained-model gain, a velocity-loss decomposition, or an architectural approximation lower bound.')
# Code and figures independently inspected; hashes bind that exact review.
import csv
reviewed_script=Path('/Users/mr.z/ODE-FFN-report-work/scripts/diagnose_complete_field.py')
assert sha(reviewed_script)=='420dd871df5417d24b8d2be8d1c074b60ee59007c171c584046e3728c0799fd3'
assert sha(OUT/'source/diagnose_complete_field.py')==sha(reviewed_script)
expected_figures={'complete-field-error-components.png': 'edd10f7a51c13c02e5fa79176d13621aa16631bea9fd05c9f7eff22b0d9f5003', 'complete-field-time-profiles.png': 'b688b370b5c94bfacf740ac430dd92e1e48444f2748f194fce2ef8bb05e6a4cf'}
for name,digest in expected_figures.items():assert sha(OUT/'figures'/name)==digest
mapping={'velocity_total':'mean_velocity_jacobian_error','velocity_skew':'mean_velocity_skew_distance',
 'velocity_negative':'mean_velocity_negative_distance','velocity_structural':'mean_velocity_structural_distance',
 'velocity_projected':'mean_velocity_projected_error','velocity_cross':'mean_velocity_projection_slack',
 'PSD_violation_fraction':'psd_violation_fraction','structural_fraction':'velocity_structural_distance_fraction',
 'denoiser_total':'mean_denoiser_jacobian_error','denoiser_structural':'mean_denoiser_structural_distance'}
maximum=0.
for filename,ours in [('aggregate.csv',groups),('per-panel.csv',panels)]:
 theirs=list(csv.DictReader((OUT/filename).open()));assert len(theirs)==len(ours)
 for r in ours:
  matches=[v for v in theirs if v['dataset']==r['dataset'] and v['arm']==r['arm'] and ('time' not in r or float(v['time'])==r['time'])]
  assert len(matches)==1
  for own_key,other_key in mapping.items():
   actual=float(matches[0][other_key]);np.testing.assert_allclose(actual,r[own_key],atol=2e-11,rtol=2e-11)
   maximum=max(maximum,abs(actual-r[own_key]))
result.update(state='PASS',script_sha256=sha(Path(__file__)),independent_aggregate_and_panel_comparison_max_absolute_difference=maximum,
 reviewed_diagnostic_sha256=sha(reviewed_script),reviewed_figures_sha256=expected_figures,
 review=dict(math='A=I+aJv; Q is nearest symmetric PSD matrix. E=D+Eprojected+cross, cross=-2 trace(N*T)>=0 for negative PSD N=S-Q and PSD T.',
 units='All denoiser squared errors divided by a^2 before averaging; every native velocity-Jacobian summary reproduced.',
 code='No blocking issue in source geometry, eigenvector reconstruction, scalar tolerance, equal-anchor/time aggregation, input-hash/common-oracle checks or scope claims.',
 figures='Both PNGs inspected: readable axes, complete arms/times, correct velocity units and geometry-only interpretation. No statistical uncertainty implied.',
 interpretation='Negative eigenvalue mass accounts for only0.118% Ring and0.210% Product parent Jacobian error. Violation counts are not the dominant gap; no constrained-model lower bound or trained improvement follows.'))

(OUT/'independent-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(groups,indent=2))
