"""Descriptive finite-response analysis; no architecture or quality claim."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ARMS=('ode_k1','fixed06','matched_silu64','swiglu86','silu131')
DOMAINS=('paired_validation','paired_training','generated_K1','generated_Swi86')
load=lambda p:json.loads(Path(p).read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def table(p,rows):
    if not rows:return
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def analyze(root,out):
    tool=load(root/'run-tool.json');assert tool['exit_code']==0 and tool['actual_terminal_exit_consumed']
    audit=load(root/'audit/result.json');assert audit['status']=='PASS';assert all(v['exit_code']==0 for v in load(root/'controller/launches.json'))
    formal=load(root/'output/result.json');settings=load(root/'settings.json');rows=formal['rows'];assert len(rows)==settings['expected_rows']
    out.mkdir(exist_ok=False);fig=out/'figures';fig.mkdir();table(out/'all-responses.csv',rows);summaries=[];shared=[]
    for domain in DOMAINS:
        for arm in ARMS:
            v=[p for p in rows if p['domain']==domain and p['arm']==arm and p['frame']==arm];assert len(v)==10
            summaries.append(dict(domain=domain,arm=arm,mean_negative_hinge=float(np.mean([p['negative_hinge'] for p in v])),
                mean_skew_fraction=float(np.mean([p['skew_fraction'] for p in v])),minimum_observed_ritz=min(p['min_ritz'] for p in v),
                negative_witness_count=sum(p['rayleigh']<0 for p in v),material_witness_count=sum(p['rayleigh']<-.05 for p in v),count=len(v)))
        for arm in ('ode_k1','swiglu86'):
            v=[p for p in rows if p['domain']==domain and p['arm']==arm and p['frame']=='common']
            if v:
                assert len(v)==10;shared.append(dict(domain=domain,arm=arm,mean_negative_hinge=float(np.mean([p['negative_hinge'] for p in v])),
                    minimum_observed_ritz=min(p['min_ritz'] for p in v),material_witness_count=sum(p['rayleigh']<-.05 for p in v),count=len(v)))
    table(out/'own-space-summaries.csv',summaries);table(out/'common-space-summaries.csv',shared);checks=[]
    for domain in DOMAINS[2:]:
        if shared:
            k=next(v for v in shared if v['domain']==domain and v['arm']=='ode_k1');s=next(v for v in shared if v['domain']==domain and v['arm']=='swiglu86')
            checks.append(dict(domain=domain,Swi_mean_hinge=s['mean_negative_hinge'],K1_mean_hinge=k['mean_negative_hinge'],Swi_count_below_minus_005=s['material_witness_count'],
                Swi_mean_at_least_005=s['mean_negative_hinge']>=.05,Swi_at_least_three_witnesses=s['material_witness_count']>=3,K1_hinge_25_percent_lower=k['mean_negative_hinge']<=.75*s['mean_negative_hinge']))
    passed=len(checks)==2 and all(v[k] for v in checks for k in ('Swi_mean_at_least_005','Swi_at_least_three_witnesses','K1_hinge_25_percent_lower'))
    decision=dict(diagnostic_premise_pass=passed,common_patch_rank=settings['common_patch_rank'],checks=checks,new_architecture_selected=False,new_training=False,new_FID=False,
        full_ambient_Jacobian_reconstructed=False,global_PSD_certified=False);write(out/'decision.json',decision)
    figure,axes=plt.subplots(1,4,figsize=(14,3.6))
    for ax,domain in zip(axes,DOMAINS):
        for arm,color in (('ode_k1','tab:blue'),('swiglu86','tab:orange')):
            points=[p for p in rows if p['domain']==domain and p['arm']==arm and p['frame']=='common']
            if not points:continue
            times=sorted({p['time'] for p in points});means=[]
            for time in times:
                v=[p['min_ritz'] for p in points if p['time']==time];ax.scatter([time]*len(v),v,color=color,alpha=.4,s=12);means.append(np.mean(v))
            ax.plot(times,means,'.-',label=arm,color=color)
        ax.axhline(0,color='black',linewidth=.7);ax.axhline(-.05,color='red',linestyle='--',linewidth=.7)
        ax.set_title(domain);ax.set_xlabel('FM time');ax.set_ylabel('Smallest Ritz value in shared probe space');ax.grid(alpha=.15)
    axes[-1].legend(fontsize=8);figure.tight_layout();figure.savefig(fig/'01-common-response.png',dpi=180);figure.savefig(fig/'01-common-response.pdf');plt.close(figure)
    figure,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key,title in zip(axes,('mean_negative_hinge','mean_skew_fraction'),('Negative hinge in each own probe space','Skew fraction in each own probe space')):
        values=np.array([[next(p[key] for p in summaries if p['arm']==a and p['domain']==d) for d in DOMAINS] for a in ARMS])
        im=ax.imshow(values,aspect='auto',cmap='viridis',vmin=0);ax.set_yticks(range(5),ARMS);ax.set_xticks(range(4),('Val pairs','Train pairs','K1 path','Swi path'));ax.set_title(title,fontsize=10)
        for i in range(5):
            for j in range(4):ax.text(j,i,f'{values[i,j]:.3g}',ha='center',va='center',color='white' if values[i,j]<values.max()*.6 else 'black',fontsize=8)
        figure.colorbar(im,ax=ax,shrink=.8)
    figure.tight_layout();figure.savefig(fig/'02-ordinary-controls.png',dpi=180);figure.savefig(fig/'02-ordinary-controls.pdf');plt.close(figure)
    lines=['# Image-response geometry','','The registered common-direction premise '+('passes.' if passed else 'fails.')+' No new model is trained or nominated.','',
        '| State origin | K1 mean negative hinge | Swi86 mean negative hinge | K1 witnesses below -.05 | Swi86 witnesses below -.05 |','|---|---:|---:|---:|---:|']
    for domain in DOMAINS:
        if not shared:break
        k=next(v for v in shared if v['domain']==domain and v['arm']=='ode_k1');s=next(v for v in shared if v['domain']==domain and v['arm']=='swiglu86')
        lines.append(f'| {domain} | {k["mean_negative_hinge"]:.6g} | {s["mean_negative_hinge"]:.6g} | {k["material_witness_count"]}/10 | {s["material_witness_count"]}/10 |')
    lines+=['','The main contrast uses the same24directions for both models inside their shared observed input space. The five-model own-space summaries retain all controls, but their learned observation spaces and response-dependent probes differ. They are context, not a controlled decomposition of the FID gap.','',
        'A negative Rayleigh value is a verified direction-specific violation of the ideal observed-coordinate response. A nonnegative finite sketch cannot certify the complete model PSD. A smallest Ritz value is not a globally accurate minimum eigenvalue.','',
        '## Decision','',''+('Retain this necessary-property contrast as a hypothesis requiring ordinary-control attribution and a separately registered quality experiment. It does not demonstrate that a PSD architecture improves generation.' if passed else 'Do not prioritize a new PSD/monotonicity restriction from this finite comparison. It does not supply the prespecified model contrast. No larger Krylov size, new state cohort or changed threshold is selected as a rescue.'),
        'The result concerns the complete predictor after its actual lossy image observation. It imposes no theorem on hidden P and does not reopen old covariance-head, conservative-readout, spectral, frame or rotation settings.','',
        '## Verification','','The original native outputs and a literal float32 norm/attention implementation replay bitwise. The cast-free real extension agrees with native forwards before measurement. Every stored compression and witness is checked by central forward differences; no full ambient Jacobian was reconstructed. The actual controller and audit exits are zero.','',
        '## Claim candidate','','- Claim: the fixed shared-direction comparison '+('satisfies' if passed else 'does not satisfy')+' its registered priority rule.',
        '  - Evidence: all-responses.csv, common-space-summaries.csv, decision.json and complete finite-difference audit.',
        '  - Allowed wording: a finite observed-coordinate response result at these frozen checkpoints.',
        '  - Forbidden stronger wording: global PSD, a causal explanation of FID, a hidden-P stability theorem, or a new method gain.',
        '  - Uncertainty: two fixed images per time/domain and a24-dimensional response-dependent subspace.',
        '  - Next check: '+('controlled attribution and a task-facing hypothesis before any training.' if passed else 'choose a different identifiable deficit from completed image evidence before changing the architecture.'),
        '  - Decision: '+('retain the hypothesis only.' if passed else 'stop this priority setting.'),'']
    (out/'analysis-report.md').write_text('\n'.join(lines))
    (out/'stats-appendix.md').write_text('# Scope and arithmetic\n\nEach model has40own-space anchor responses. K1 and Swi86 share40additional anchor responses each in identical paired probe spaces. These are fixed states sharing images, times and trajectories, not independent population samples. The priority rule uses mean negative hinge on both generated domains and counts within the fixed10rows/domain/model; no p-values or population violation rates are reported. The eigenvalues belong to symmetric24-by24compressions, not full3072-dimensional Jacobians.\n')
    (out/'figure-catalog.md').write_text('# Figure interpretation\n\n1. Shared response: points retain images0/1at every time. Lines are their means; dashed red is the predeclared-.05material-witness threshold. Compare both generated origins. No confidence band or global spectral claim.\n2. Ordinary context: mean negative hinge and skew fraction on each model\'s own observed space. Probe spaces differ, so this cannot isolate a training mechanism or rank whole-model stability. Exact row values and all response records are retained.\n')
    with (out/'results.tsv').open('w',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['commit','metric','memory_gb','status','description']);w.writerow([settings['source_commit'],'NA','NA','keep' if passed else 'discard','complete-predictor finite-response diagnostic; no training, FID or accepted architecture'])
    write(out/'provenance.json',dict(runtime=str(root),source_commit=settings['source_commit'],audit_sha256=hashlib.file_digest((root/'audit/result.json').open('rb'),'sha256').hexdigest(),tool=tool))
    print(json.dumps(decision),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.root,a.output)
