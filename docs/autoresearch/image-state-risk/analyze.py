"""Descriptive shared-state analysis, only after complete independent audit."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ARMS=('ode_k1','fixed06','matched_silu64','swiglu86','silu131')
DOMAINS=('paired_validation','paired_training','generated_K1','generated_Swi86');INDICES=(13,38,64,90,115)
load=lambda p:json.loads(Path(p).read_text())
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def table(p,rows):
    keys=list(dict.fromkeys(k for v in rows for k in v))
    with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def analyze(root,out):
    tool=load(root/'run-tool.json');assert tool['exit_code']==0 and tool['actual_terminal_exit_consumed']
    audit=load(root/'audit/result.json');assert audit['status']=='PASS'
    assert all(v['exit_code']==0 for v in load(root/'controller/launches.json'))
    result=load(root/'output/result.json');rows=result['rows'];refs=result['reference_rows'];assert len(rows)==100 and len(refs)==20
    out.mkdir(exist_ok=False);fig=out/'figures';fig.mkdir();table(out/'all-model-rows.csv',rows);table(out/'all-reference-rows.csv',refs)
    summaries=[]
    for domain in DOMAINS:
        for arm in ARMS:
            v=[p for p in rows if p['arm']==arm and p['domain']==domain];assert len(v)==5
            summaries.append(dict(domain=domain,arm=arm,**{m:float(np.mean([p[m] for p in v])) for m in ('bank_reference_clean_mse','bank_reference_velocity_mse')},
                paired_target_mse=float(np.mean([p['paired_target_mse'] for p in v])) if domain.startswith('paired_') else None))
    table(out/'equal-time-summary.csv',summaries)
    ratios={domain:next(p['bank_reference_clean_mse'] for p in summaries if p['domain']==domain and p['arm']=='ode_k1')/next(p['bank_reference_clean_mse'] for p in summaries if p['domain']==domain and p['arm']=='swiglu86') for domain in DOMAINS}
    reversal=ratios['paired_validation']>=1 and all(ratios[d]<=.9 for d in DOMAINS[2:]);validity=[]
    for j in INDICES:
        r=next(v for v in refs if v['domain']=='paired_validation' and v['index']==j)
        k=next(v for v in rows if v['domain']=='paired_validation' and v['index']==j and v['arm']=='ode_k1')
        s=next(v for v in rows if v['domain']=='paired_validation' and v['index']==j and v['arm']=='swiglu86')
        validity.append(dict(index=j,time=j/128,reference_paired_target_mse=r['paired_target_mse'],K1_paired_target_mse=k['paired_target_mse'],Swi86_paired_target_mse=s['paired_target_mse'],
            reference_ratio_to_K1=r['paired_target_mse']/k['paired_target_mse'],reference_ratio_to_Swi86=r['paired_target_mse']/s['paired_target_mse'],
            exceeds_both_by_10_percent=r['paired_target_mse']>=1.1*k['paired_target_mse'] and r['paired_target_mse']>=1.1*s['paired_target_mse']))
    unsuitable=sum(v['exceeds_both_by_10_percent'] for v in validity)>=3;table(out/'reference-validity.csv',validity)
    decision=dict(empirical_risk_reversal_pass=reversal,reference_unsuitable_for_population_design=unsuitable,K1_over_Swi86_equal_time_clean_reference_ratios=ratios,
        unsuitable_time_count=sum(v['exceeds_both_by_10_percent'] for v in validity),new_architecture_selected=False,new_FID=False,new_training=False)
    write(out/'decision.json',decision)
    figure,axes=plt.subplots(1,4,figsize=(15,3.8));colors={a:plt.get_cmap('tab10')(i) for i,a in enumerate(ARMS)}
    for ax,domain in zip(axes,DOMAINS):
        for arm in ARMS:
            v=[p for p in rows if p['domain']==domain and p['arm']==arm];ax.plot([p['time'] for p in v],[p['bank_reference_clean_mse'] for p in v],'.-',label=arm,color=colors[arm])
        ax.set_title(domain);ax.set_xlabel('FM time');ax.set_ylabel('MSE to finite-bank conditional mean');ax.set_yscale('log');ax.grid(alpha=.15)
    axes[-1].legend(fontsize=7,bbox_to_anchor=(1.03,1));figure.tight_layout();figure.savefig(fig/'01-common-state-risk.png',dpi=180);figure.savefig(fig/'01-common-state-risk.pdf');plt.close(figure)
    figure,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,domain in zip(axes[:2],DOMAINS[:2]):
        for arm in ARMS:
            v=[p for p in rows if p['domain']==domain and p['arm']==arm];ax.plot([p['time'] for p in v],[p['paired_target_mse'] for p in v],'.-',label=arm,color=colors[arm])
        v=[p for p in refs if p['domain']==domain];ax.plot([p['time'] for p in v],[p['paired_target_mse'] for p in v],'x--',label='bank conditional mean',color='black');ax.set_title(domain);ax.set_ylabel('MSE to actual clean image');ax.set_yscale('log');ax.grid(alpha=.15)
    for domain in DOMAINS:
        v=[p for p in refs if p['domain']==domain];axes[2].plot([p['time'] for p in v],[p['effective_support_median'] for p in v],'.-',label=domain)
    axes[2].set_title('Finite-bank posterior support');axes[2].set_yscale('log');axes[2].set_ylabel('Median effective support');axes[2].grid(alpha=.15)
    axes[0].legend(fontsize=6);axes[2].legend(fontsize=6)
    for ax in axes:ax.set_xlabel('FM time')
    figure.tight_layout();figure.savefig(fig/'02-reference-scope.png',dpi=180);figure.savefig(fig/'02-reference-scope.pdf');plt.close(figure)
    with np.load(root/'targets.npz') as f:target=f['paired_validation'][0]
    figure,axes=plt.subplots(4,5,figsize=(10,8));labels=('Held-out clean image','Bank conditional mean','K1 prediction','Swi86 prediction')
    for col,j in enumerate(INDICES):
        key=f'paired_validation_{j}'
        with np.load(root/'output/posterior'/f'{key}.npz') as f:mean=f['mean'][0].reshape(3,32,32)
        images=[target,mean,np.load(root/'output/predictions'/f'ode_k1_{key}.npy')[0],np.load(root/'output/predictions'/f'swiglu86_{key}.npy')[0]]
        for row,value in enumerate(images):
            axes[row,col].imshow(np.clip((value.transpose(1,2,0)+1)/2,0,1));axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if col==0:axes[row,col].set_ylabel(labels[row],fontsize=8)
        axes[0,col].set_title(f't={j}/128')
    figure.tight_layout();figure.savefig(fig/'03-fixed-validation-image0.png',dpi=180);figure.savefig(fig/'03-fixed-validation-image0.pdf');plt.close(figure)
    body=['# Shared-state image risk','','No new model is trained or nominated. The registered empirical-risk reversal '+('passes.' if reversal else 'fails.'),
        'The finite-bank reference '+('fails' if unsuitable else 'does not fail')+' its prespecified held-out-use guard.','',
        '| Domain | K1 / Swi86, MSE to bank mean |','|---|---:|']
    body += [f'| {d} | {ratios[d]:.6f} |' for d in DOMAINS]
    body += ['','These are equal-time clean-risk ratios on the same states; they are neither FID nor integrated native training risk. All100model rows and all20reference rows are retained.','',
        '| Time | Bank reference / K1, actual-image MSE | Bank reference / Swi86 |','|---|---:|---:|']
    body += [f'| {v["index"]}/128 | {v["reference_ratio_to_K1"]:.5f} | {v["reference_ratio_to_Swi86"]:.5f} |' for v in validity]
    body += ['','## Interpretation','','The exact conditional mean of a uniform finite training bank is not automatically a population denoiser. Its training/held-out behavior and posterior support must accompany any distance-to-reference ranking. '
        +('This guard blocks using the measured bank-reference ranking to justify a population-quality architecture.' if unsuitable else 'Passing this guard alone still cannot establish that the bank mean is the unknown population optimum.'),
        'A cross-state ranking does not causally explain the earlier FID gap, identify a hidden P mechanism, or demonstrate neural memorization. No new loss, prior smoothing, bandwidth, sampling intervention or training rescue is selected.','',
        '## Claim candidates','','- Claim: the exact bank conditional mean '+('is unsuitable under the declared guard' if unsuitable else 'does not trigger the declared unsuitability guard')+' on these fixed held-out probes.',
        '  - Evidence: reference-validity.csv, raw reference means, direct-distance audit.',
        '  - Allowed wording: the finite empirical reference requires the reported scope.',
        '  - Forbidden stronger wording: true population error, memorization of a neural model, or an explanation of FID.',
        '  - Uncertainty: the population conditional mean is unknown; these paths and Gaussian starts are shared fixed probes.',
        '  - Next check: use the result to choose an identifiable missing premise before another architectural experiment.',
        '  - Decision: '+('discard the proposed population-reference interpretation.' if unsuitable else 'retain only the scoped diagnostic.'),'' ]
    (out/'analysis-report.md').write_text('\n'.join(body))
    (out/'stats-appendix.md').write_text('# Statistical scope\n\nEvery model sees the same64images per state batch. Domains share starts and paired targets; time points, pixels and models are not independent trials. Report descriptive MSEs and fixed ratios, without p-values or population confidence intervals. The reversal criterion uses equal-time clean-reference MSE; velocity-scaled errors and actual-image paired errors remain separate. No held-out test batch is accessed.\n')
    (out/'figure-catalog.md').write_text('# Figures\n\n1. Common-state risk compares all five models on every domain/time. Inspect whether the ranking changes with states; no curve is a sampling-quality score.\n2. Reference scope contrasts actual-image errors on training and held-out pairs and the effective posterior support. It checks whether exact empirical conditioning supplies the intended population reference. Zero training error is shown on a log axis only where positive; exact values remain in the CSV.\n3. The fixed first held-out image is shown at all five times, with empirical-bank, K1 and Swi86 predictions. Only display values are clipped; risk uses raw values. This is a diagnostic case, not generated image quality.\n')
    with (out/'results.tsv').open('w',newline='') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['commit','metric','memory_gb','status','description']);w.writerow([load(root/'settings.json')['source_commit'],'NA','NA','discard' if unsuitable or not reversal else 'keep','read-only empirical-reference diagnostic; no trained successor, no new FID'])
    write(out/'provenance.json',dict(runtime=str(root),source_commit=load(root/'settings.json')['source_commit'],audit_sha256=hashlib.file_digest((root/'audit/result.json').open('rb'),'sha256').hexdigest(),tool=tool))
    print(json.dumps(decision),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.root,a.output)
