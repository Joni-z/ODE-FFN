"""Generate only audited complete-cohort image evidence; no candidate selection."""
from __future__ import annotations
import argparse,csv,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
load=lambda p:json.loads(p.read_text())
assert load(root/'postprocess-v2/status.json')['state']=='COMPLETE'
assert load(root/'audit-training/audit.json')['status']=='PASS'
assert load(root/'audit-features-v2/audit.json')['status']=='PASS'
out=root/'curation';out.mkdir(exist_ok=False);figdir=out/'figures';figdir.mkdir()
read=lambda p:list(csv.DictReader(p.open()))
def table(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
arms=['ode_k1','ode_k2','swiglu86','swiglu129'];names=['ODE K1','ODE K2','Swi86','Swi129'];colors=['#6699bb','#d17a28','#6a9f58','#805da8']
comparison=read(root/'audit-training/comparison.csv');curves=read(root/'audit-training/curves.csv')
fids={r['family']:r['fid'] for r in load(root/'audit-features-v2/audit.json')['results']}
parent=Path(load(root/'settings.json')['parent']);costs={r['family']:r for r in load(parent/'preflight.json')['rows'] if r['device']=='cpu'}
macs={r['family']:r for r in load(root/'preflight.json')['arms']}
monitor=[json.loads(s) for s in (root/'output/monitor.jsonl').read_text().splitlines()]
launches=load(root/'output/launches.json');memory={}
for r in launches:
    samples=[p['rss_bytes'] for row in monitor for p in row['processes'] if p['pid']==r['pid']]
    assert samples;memory[r['family']]=max(samples)/2**30
for r in comparison:
    for k in list(r):
        if k!='family':r[k]=float(r[k])
    arm=r['family'];r.update(fid=fids[arm],parameters=costs[arm]['trainable_parameters'],
        FFN_forward_MACs=macs[arm]['dense_FFN_forward_MACs_per_token'],FFN_training_MACs=macs[arm]['dense_FFN_training_MACs_per_token'],
        sampled_peak_RSS_GiB=memory[arm],FID_generated=1024,FID_reference=2048)
table(out/'comparison.csv',comparison);shutil.copyfile(root/'audit-training/curves.csv',out/'curves.csv')
# Post-result accounting of the already registered FID, not another metric.
real_features=np.load(root/'metric-inputs/reference-features.npy').astype(np.float64)
real_mean=real_features.mean(0);real_trace=float(np.sum((real_features-real_mean)**2)/(len(real_features)-1));fid_parts=[]
for arm in arms:
    features=np.load(root/'metric-output'/f'{arm}-features.npy').astype(np.float64);mean=features.mean(0)
    mean_part=float(np.sum((mean-real_mean)**2));trace=float(np.sum((features-mean)**2)/(len(features)-1))
    total=fids[arm];covariance_part=total-mean_part
    assert covariance_part>=-1e-6
    fid_parts.append(dict(family=arm,FID=total,mean_squared_distance=mean_part,covariance_mismatch=covariance_part,
        generated_covariance_trace=trace,reference_covariance_trace=real_trace,implied_cross_trace=(mean_part+trace+real_trace-total)/2))
table(out/'FID-accounting.csv',fid_parts)

for src,dst in [('audit-training/audit.json','training-audit.json'),('audit-features-v2/audit.json','feature-audit.json'),
    ('preflight.json','trainer-preflight.json'),('metric-ready.json','metric-reference.json')]:shutil.copyfile(root/src,out/dst)
by={r['family']:r for r in comparison};ref=[by['swiglu86'],by['swiglu129']];candidate=by['ode_k2'];k1=by['ode_k1']
checks=[]
for r in ref:checks.append(dict(condition='K2 weighted MSE <= .9 ordinary',reference=r['family'],ratio=candidate['weighted_mse']/r['weighted_mse'],limit=.9,passed=candidate['weighted_mse']<=.9*r['weighted_mse']))
for metric in ('clean_mse','fid'):
    best=min(r[metric] for r in ref);checks.append(dict(condition=f'K2 {metric} guard',reference='better ordinary',ratio=candidate[metric]/best,limit=1.1,passed=candidate[metric]<=1.1*best))
base=all(r['passed'] for r in checks)
order=[dict(condition='K2 weighted MSE <= .95 K1',reference='ode_k1',ratio=candidate['weighted_mse']/k1['weighted_mse'],limit=.95,passed=candidate['weighted_mse']<=.95*k1['weighted_mse'])]
for metric in ('clean_mse','fid'):order.append(dict(condition=f'K2 {metric} guard against K1',reference='ode_k1',ratio=candidate[metric]/k1[metric],limit=1.1,passed=candidate[metric]<=1.1*k1[metric]))
decision=dict(original_K2_image_lead=base,higher_order_support=base and all(r['passed'] for r in order),
    ordinary_checks=checks,order_checks=order,new_architecture_nominated=False,
    note='A positive original baseline is not a new method or a synthetic transfer result; keep all stopped families stopped.')
(out/'decision.json').write_text(json.dumps(decision,indent=2)+'\n');table(out/'decision-checks.csv',checks+order)
P=[];clocks=[];telemetry=[]
for arm in arms:
    folder=root/'output/runs'/arm
    for line in (folder/'metrics.jsonl').read_text().splitlines():
        r=json.loads(line);telemetry.append({k:dict(family=arm,**r).get(k) for k in ('family','step','train_loss','grad_norm','clip_fraction','weighted_mse','clean_mse')})
    for step in (0,1000,2500,5000):
        for r in load(folder/f'P_{step:07d}.json'):
            sv=np.array(r['singular_values']);re=np.array(r['eigenvalues_real'])
            row=dict(family=arm,step=step,layer=r['layer'],sigma_max=sv[0],sigma_min=sv[-1],singular_cv=np.std(sv)/np.mean(sv),
                frobenius=r['frobenius'],eigen_real_max=np.max(re),eigen_real_min=np.min(re))
            for i,v in enumerate(sv):row[f'sigma_{i:02d}']=v
            for prefix,key in [('eigen_real','eigenvalues_real'),('eigen_imag','eigenvalues_imag')]:
                for i,v in enumerate(r[key]):row[f'{prefix}_{i:02d}']=v
            P.append(row)
        for r in load(folder/f'clocks_{step:07d}.json'):clocks.append(dict(family=arm,step=step,**{k:v for k,v in r.items() if k!='probes'}))
table(out/'P-history.csv',P);table(out/'clocks.csv',clocks);table(out/'training-telemetry.csv',telemetry)
with np.load(root/'banks/validation.npz') as b:target=b['target'].astype(np.float64);delta=b['delta'].astype(np.float64)
per_image=[]
for arm in arms:
    with np.load(root/'output/runs'/arm/'validation_0005000.npz') as f:prediction=f['prediction'].astype(np.float64)
    clean=np.mean((prediction-target)**2,axis=(1,2,3));weighted=clean/delta**2
    per_image.append(dict(family=arm,images=1024,weighted_mean=weighted.mean(),weighted_SD=np.std(weighted,ddof=1),
        weighted_q10=np.quantile(weighted,.1),weighted_median=np.median(weighted),weighted_q90=np.quantile(weighted,.9),clean_mean=clean.mean(),clean_SD=np.std(clean,ddof=1)))
table(out/'per-image-distribution.csv',per_image)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
x=np.arange(4);fig,axes=plt.subplots(1,3,figsize=(11,3.6))
for ax,key,label in zip(axes,('weighted_mse','clean_mse','fid'),('Weighted validation MSE','Clean-pixel MSE','FID: 1024 generated / 2048 real')):
    values=[by[arm][key] for arm in arms];bars=ax.bar(x,values,color=colors);ax.set_xticks(x,names,rotation=25,ha='right');ax.set_title(label,fontsize=10)
    ax.bar_label(bars,labels=[f'{v:.4f}' if key!='fid' else f'{v:.2f}' for v in values],fontsize=8,padding=2);ax.set_ylim(0,max(values)*1.2)
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(figdir/f'fixed-comparison.{ext}',dpi=170)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10,3.5))
for arm,name,color in zip(arms,names,colors):
    rows=[r for r in curves if r['family']==arm];steps=[int(r['step']) for r in rows]
    for ax,key in zip(axes,('weighted_mse','clean_mse')):ax.plot(steps,[float(r[key]) for r in rows],label=name,color=color)
axes[0].set_ylabel('Weighted validation MSE');axes[1].set_ylabel('Clean-pixel MSE')
for ax in axes:ax.set_xlabel('Update');ax.set_yscale('log');ax.legend(fontsize=8)
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(figdir/f'learning-curves.{ext}',dpi=170)
plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(10,6),sharex=True)
for j,arm in enumerate(arms[:2]):
    layers=list(dict.fromkeys(r['layer'] for r in P if r['family']==arm))
    for layer in layers:
        rows=[r for r in P if r['family']==arm and r['layer']==layer];steps=[r['step'] for r in rows]
        axes[0,j].plot(steps,[r['sigma_max'] for r in rows],marker='.',label=layer.replace('blocks.','b'))
        means=[np.mean([r['s_mean'] for r in clocks if r['family']==arm and r['layer']==layer and r['step']==step]) for step in steps]
        axes[1,j].plot(steps,means,marker='.')
    axes[0,j].set_title(arm);axes[0,j].legend(ncol=2,fontsize=6);axes[1,j].set_ylim(.2,1);axes[1,j].set_xlabel('Update')
axes[0,0].set_ylabel('P maximum singular value');axes[1,0].set_ylabel('Mean clock over five fixed time panels')
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(figdir/f'P-and-clocks.{ext}',dpi=170)
plt.close(fig)
fig,axes=plt.subplots(4,8,figsize=(8.5,4.4))
for row,(arm,name) in enumerate(zip(arms,names)):
    raw=np.load(root/'output/runs'/arm/'generated.npy')
    for col in range(8):
        ax=axes[row,col];ax.imshow(np.clip((raw[col].transpose(1,2,0)+1)/2,0,1),interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
        if col==0:ax.set_ylabel(name,fontsize=8)
fig.suptitle('Generated images: same first eight noise starts, Heun32, no CFG',fontsize=10);fig.tight_layout(rect=(0,0,1,.95),pad=.3)
for ext in ('png','pdf'):fig.savefig(figdir/f'generation-cases.{ext}',dpi=170)
plt.close(fig)
case_rows=[]
fig,axes=plt.subplots(5,6,figsize=(8,6.8))
for row,t in enumerate((.1,.3,.5,.7,.9)):
    with np.load(root/'banks'/f'panel_{t:.1f}.npz') as b:images=[b['target'][0],b['x'][0]]
    for arm in arms:
        with np.load(root/'output/runs'/arm/f'panel_0005000_{t:.1f}.npz') as f:images.append(f['prediction'][0])
        error=images[-1].astype(np.float64)-images[0].astype(np.float64)
        case_rows.append(dict(family=arm,validation_row=0,time=t,clean_mse=float(np.mean(error**2))))
    for col,image in enumerate(images):
        ax=axes[row,col];ax.imshow(np.clip((image.transpose(1,2,0)+1)/2,0,1),interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
        if row==0:ax.set_title(['Target','Noisy',*names][col],fontsize=9)
        if col==0:ax.set_ylabel(f't={t:.1f}',fontsize=8)
fig.suptitle('First validation target at all five predeclared noise levels',fontsize=10);fig.tight_layout(rect=(0,0,1,.96),pad=.3)
for ext in ('png','pdf'):fig.savefig(figdir/f'denoising-case.{ext}',dpi=170)
plt.close(fig)
table(out/'first-case.csv',case_rows)
header='| Model | Params | Weighted MSE | Clean MSE | FID (1k/2k) | Raw pixels outside range % |\n|---|---:|---:|---:|---:|---:|\n'
rows=''.join(f"| {r['family']} | {r['parameters']} | {r['weighted_mse']:.7f} | {r['clean_mse']:.7f} | {r['fid']:.4f} | {100*r['out_of_range_fraction']:.3f} |\n" for r in comparison)
(out/'numeric-table.md').write_text(header+rows)
with (out/'results.tsv').open('w') as f:
    f.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    for r in comparison:
        status='discard' if r['family']=='ode_k2' and not base else 'keep'
        f.write(f"{load(root/'settings.json')['registration_commit'][:7]}\t{r['weighted_mse']:.12g}\t{r['sampled_peak_RSS_GiB']:.3f}\t{status}\t{r['family']}: image baseline only, FID {r['fid']:.6f}; no new method nominated\n")
print(json.dumps(dict(status='CURATED',models=4,checkpoints=44,P_records=len(P),clock_rows=len(clocks),figures=5,decision=decision)),flush=True)
