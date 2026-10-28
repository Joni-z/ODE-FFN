"""Curate every deterministic probe and the fixed closed-form counterexamples."""
from pathlib import Path
import argparse,csv,json,sys,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args();root=a.root
sys.path.insert(0,str(root/'source'));import study
study.verify(root);assert study.load(root/'orchestrator/status.json')['state']=='COMPLETE'
formal=study.load(root/'output/result.json');audit=study.load(root/'audit/result.json');assert formal['status']==audit['status']=='PASS'
assert len(formal['cases'])==120 and audit['finite_difference_cases']==240
out=root/'curation';out.mkdir();fig=out/'figures';fig.mkdir()
def save(name,rows):
    with (out/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
save('point-cases.csv',formal['cases']);save('finite-differences.csv',audit['records'])
controls=study.load(root/'output/controls.json');study.write(out/'controls.json',controls)
applications=[]
for name,parent in study.paths(root).items():
    distribution=study.load(parent/name/'distribution.json')['distribution']
    if 'locations' in distribution:
        location=np.array(distribution['locations']);variance=np.array(distribution['variances']);rotation=np.array(distribution['rotation'])
        assert np.isfinite(location).all() and np.isfinite(variance).all() and (variance>0).all()
        error=float(np.max(np.abs(rotation.T@rotation-np.eye(len(rotation)))));assert error<=1e-12
        row=dict(dataset=name,dim=len(location),finite_GMM_components=2**len(location),minimum_component_variance=float(variance.min()),positive_normalized_weights=True,rotation_orthogonality_maxerror=error,distribution_sha256=study.sha(parent/name/'distribution.json'),validation_sha256=study.sha(parent/name/'validation.npz'))
    else:
        means=np.array(distribution['means']);variance=np.array(distribution['variances']);weights=np.array(distribution['weights'])
        assert np.isfinite(means).all() and np.isfinite(variance).all() and np.isfinite(weights).all() and (variance>0).all() and (weights>0).all()
        row=dict(dataset=name,dim=means.shape[1],finite_GMM_components=len(weights),minimum_component_variance=float(variance.min()),positive_normalized_weights=bool(abs(weights.sum()-1)<1e-12),rotation_orthogonality_maxerror=0.,distribution_sha256=study.sha(parent/name/'distribution.json'),validation_sha256=study.sha(parent/name/'validation.npz'))
        assert row['positive_normalized_weights']
    applications.append(row)
study.write(out/'assumption-discharge.json',dict(status='PASS',scope='Prior-class application to the three existing analytic laws; numerical rotation tolerance, not exact-real orthogonality certification',rows=applications))
summary=[]
for name in study.paths(root):
    rows=[r for r in formal['cases'] if r['dataset']==name]
    summary.append(dict(dataset=name,points=len(rows),mean_oracle_maxerror=max(r['mean_oracle_maxerror'] for r in rows),variance_PDE_maxerror=max(r['variance_PDE_maxerror'] for r in rows),FM_PDE_maxerror=max(r['FM_PDE_maxerror'] for r in rows),jacobian_covariance_maxerror=max(r['jacobian_covariance_maxerror'] for r in rows),jacobian_min_eigenvalue=min(r['jacobian_min_eigenvalue'] for r in rows)))
save('identity-summary.csv',summary)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
def export(f,name):
    for ext in ('png','pdf'):f.savefig(fig/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(f)
t=np.linspace(.05,.95,101);v=t*t+(1-t)**2
f,axes=plt.subplots(2,2,figsize=(12,8))
axes[0,0].plot(t,t/v,label='Posterior mean D: slope t/V');axes[0,0].plot(t,1/np.sqrt(v),label='Endpoint map F: slope 1/sqrt(V)')
axes[0,0].set_ylabel('Prediction slope');axes[0,0].set_xlabel('FM time');axes[0,0].legend(fontsize=8)
axes[0,1].plot(t,(1-t)/v**2,label='Posterior mean D');axes[0,1].plot(t,np.zeros_like(t),label='Endpoint map F')
axes[0,1].set_xlabel('FM time');axes[0,1].set_ylabel('Material derivative at input z=1');axes[0,1].legend(fontsize=8)
g=controls['gaussian_at_t_half_z_one']
axes[1,0].bar(['Invariant F','Wrong-prior D2'],[g['endpoint_excess_weighted_risk'],g['wrong_prior_excess_weighted_risk']],color=['#377eb8','#ff7f00'])
axes[1,0].set_ylabel('Excess supervised risk at t=.5');axes[1,0].set_title('Both appealing constraints can miss the target')
x=np.arange(2);width=.32;atomic=controls['atomic']
axes[1,1].bar(x-width/2,[r['precision_derivative'] for r in atomic],width,label='True precision derivative',color='#e41a1c')
axes[1,1].bar(x+width/2,[r['covariance_only_drift'] for r in atomic],width,label='Covariance-only closure',color='#4daf4a')
axes[1,1].set_xticks(x,['Symmetric posterior','Skew posterior']);axes[1,1].set_ylabel('Derivative at y=0, precision=1');axes[1,1].set_title('Both have mean 0 and variance 1');axes[1,1].legend(fontsize=8)
for ax in axes.flat:ax.grid(axis='y',alpha=.2)
f.suptitle('Closed-form counterexamples, not trained-model results');f.tight_layout();export(f,'counterexamples')
f,axes=plt.subplots(3,2,figsize=(12,10))
names={'gaussian_aniso_2d':'Gaussian 2D','gmm_ring8_2d':'Ring8 2D','rotated_product_bimodal_16d':'Product16'}
for i,name in enumerate(study.paths(root)):
    for j,key in enumerate(('laplacian_RMS','material_drift_RMS')):
        ax=axes[i,j];means=[]
        for time_value in study.TIMES:
            rows=[r for r in formal['cases'] if r['dataset']==name and r['time']==time_value];values=[r[key] for r in rows]
            ax.scatter(time_value+np.linspace(-.008,.008,8),values,color='#999999',s=20,alpha=.7);means.append(np.mean(values))
        ax.plot(study.TIMES,means,'-o',color='#377eb8',label='Mean of eight fixed probes')
        ax.set_title(names[name]);ax.set_xlabel('FM time');ax.set_ylabel('Laplacian RMS in y coordinates' if j==0 else 'D material-drift RMS in z coordinates');ax.grid(alpha=.2);ax.legend(fontsize=8)
f.suptitle('All fixed probes; columns have different quantities/units, no quality ranking');f.tight_layout();export(f,'all-probes')
for name,source in [('numerical-audit.json',root/'audit/result.json')]:shutil.copyfile(source,out/name)
study.write(out/'decision.json',dict(new_training=False,new_ODE_architecture_nominated=False,scope='Necessary target identities and bounded counterexamples; no PDE regularizer, invariant target or covariance rescue selected'))
table='| Existing law | Cases | Mean/oracle max error | Variance PDE max residual | FM PDE max residual |\n|---|---:|---:|---:|---:|\n'
for r in summary:table+=f"| {names[r['dataset']]} | {r['points']} | {r['mean_oracle_maxerror']:.3g} | {r['variance_PDE_maxerror']:.3g} | {r['FM_PDE_maxerror']:.3g} |\n"
(out/'numeric-table.md').write_text(table)
with (out/'results.tsv').open('w') as f:
    f.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    commit=study.load(root/'settings.json')['source_commit'][:7]
    resources=[json.loads(line) for line in (root/'orchestrator/resources.jsonl').read_text().splitlines()]
    peak=max(r['rss'] for r in resources)/2**30
    for r in summary:f.write(f"{commit}\t{r['FM_PDE_maxerror']:.12g}\t{peak:.3f}\tkeep\t{r['dataset']}: target-identity fixture only, not model quality or a new architecture\n")
print(json.dumps(dict(status='CURATED',point_cases=120,finite_difference_cases=240,figures=2)),flush=True)
