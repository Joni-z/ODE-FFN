"""Post-outcome reporting only: paired outputs in fixed training PCA coordinates."""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np

def main(root):
    sys.path.insert(0,str(root/'source'));import gaussian_reference as g
    assert g.load(root/'orchestrator/status.json')['state']=='COMPLETE';g.verify(root)
    out=root/'reporting-subspaces';out.mkdir();before={str(p):g.sha(p) for name in ('fit','sampling','metric-output','curation') for p in (root/name).rglob('*') if p.is_file()}
    p=g.gm.load_parameters(root/'fit/ppca59.npz');mean,u=p['mean'],p['basis'];d=len(mean)
    reference=np.load(root/'sampling/ppca59-generated.npy').reshape(1024,d).astype(np.float64)-mean
    reference_coeff=reference@u;rows=[];errors=[]
    def energies(x):
        coefficients=x@u;remainder=x-coefficients@u.T
        retained=float(np.mean(np.sum(coefficients*coefficients,axis=1))/d)
        discarded=float(np.mean(remainder*remainder));total=float(np.mean(x*x))
        np.testing.assert_allclose(retained+discarded,total,rtol=1e-12,atol=1e-12)
        errors.append(abs(retained+discarded-total));return coefficients,retained,discarded,total
    def record(arm,images,paired):
        x=images.reshape(len(images),d).astype(np.float64)-mean
        a,retained,discarded,total=energies(x)
        row=dict(arm=arm,images=len(images),retained_energy_per_coordinate=retained,discarded_energy_per_coordinate=discarded,
            total_energy_per_coordinate=total,retained_RMS=float(np.sqrt(retained)),discarded_RMS=float(np.sqrt(discarded)),
            coupled_difference_total=None,coupled_difference_retained=None,coupled_difference_discarded=None,
            discarded_fraction_of_coupled_difference=None,retained_centered_correlation_to_PPCA=None)
        if paired:
            delta=x-reference;_,part1,part2,full=energies(delta)
            left=a-a.mean(0);right=reference_coeff-reference_coeff.mean(0);denom=np.linalg.norm(left)*np.linalg.norm(right)
            row.update(coupled_difference_total=full,coupled_difference_retained=part1,coupled_difference_discarded=part2,
                discarded_fraction_of_coupled_difference=part2/full if full>0 else None,
                retained_centered_correlation_to_PPCA=float(np.sum(left*right)/denom) if denom>0 else None)
        rows.append(row)
    real=g.gm.normalize_pixels(np.load(g.NATIVE/'banks/fid-reference.uint8.npy'))
    # record() expects raw image shape but only reshapes; these are already native normalized pixels.
    record('real_reference',real,False)
    formal=g.load(root/'metric-output/result.json')['rows'];primary=[r for r in formal if r['step']==20000 or r['arm'] in g.ARMS];assert len(primary)==16
    order=('ode_k1','fixed06','matched_silu64','ppca59','isotropic','native_posthoc06','ode_k1_time','ode_k1_pooled','ode_k1_constant','ode_k2','ode_k2_constant','swiglu86','swiglu129','silu64','silu131','silu193')
    for r in sorted(primary,key=lambda v:order.index(v['arm'])):record(r['arm'],np.load(r['sample_path']),True)
    with (out/'subspaces.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for path,digest in before.items():assert g.sha(path)==digest,path
    g.verify(root);g.write(out/'summary.json',dict(status='PASS',reporting_added_after_outcomes=True,trained_PCA_rank=59,
        fitted_or_generated_or_scored_new_samples=False,model_rows=16,real_reference_rows=2048,paired_generated_rows=1024,
        maximum_Pythagorean_error=max(errors),formal_files_unchanged=len(before),neural_mechanism_identified=False,
        FID_decomposition=False,rank_or_covariance_setting_changed=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    labels={'real_reference':'Real reference','ode_k1':'K1 native','fixed06':'Fixed 0.6 P','matched_silu64':'Matched-init SiLU64','ppca59':'PPCA59','isotropic':'Isotropic','native_posthoc06':'Posthoc0.6','ode_k1_time':'Old K1 time','ode_k1_pooled':'Old K1 pooled','ode_k1_constant':'Old K1 constant','ode_k2':'K2 native','ode_k2_constant':'Old K2 constant','swiglu86':'Swi86','swiglu129':'Swi129','silu64':'Default SiLU64','silu131':'SiLU131','silu193':'SiLU193'}
    fig,axes=plt.subplots(1,2,figsize=(16,5),layout='constrained')
    x=np.arange(len(rows));axes[0].bar(x-.18,[v['retained_RMS'] for v in rows],.36,label='Retained-subspace RMS')
    axes[0].bar(x+.18,[v['discarded_RMS'] for v in rows],.36,label='Complement RMS')
    axes[0].set_xticks(x,[labels[v['arm']] for v in rows],rotation=65,ha='right',fontsize=6)
    axes[0].set(ylabel='RMS per original pixel coordinate',title='Original outputs in one fixed training PCA coordinate system');axes[0].legend(fontsize=7)
    paired=[v for v in rows if v['discarded_fraction_of_coupled_difference'] is not None]
    axes[1].bar(range(len(paired)),[100*v['discarded_fraction_of_coupled_difference'] for v in paired],color='#8172b2')
    axes[1].set_xticks(range(len(paired)),[labels[v['arm']] for v in paired],rotation=65,ha='right',fontsize=6)
    axes[1].set(ylabel='Complement share of paired squared output difference (%)',ylim=(0,100),title='Difference from PPCA on common starts; not a FID decomposition')
    for ext in ('png','pdf'):fig.savefig(out/f'subspace-comparison.{ext}',dpi=180)
    plt.close(fig);print(json.dumps(g.load(out/'summary.json')),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);main(parser.parse_args().root.resolve())
