"""Curate every registered grid, not a chosen solver or chosen example."""
from pathlib import Path
import argparse,csv,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
sys.path.insert(0,str(root/'source'));import study
study.verify(root);p=study.parent(root)
assert study.load(root/'orchestrator-v2/status.json')['state']=='COMPLETE'
assert study.load(root/'audit-sampling/result.json')['status']==study.load(root/'audit-features/result.json')['status']=='PASS'
assert len(study.load(root/'audit-sampling/result.json')['rows'])==8
fid_audit=study.load(root/'audit-features/result.json')['rows'];assert len(fid_audit)==12
formal=study.load(root/'metric-output/result.json')['rows'];assert len(formal)==12
out=root/'curation';out.mkdir();fig=out/'figures';fig.mkdir()
def save(name,rows):
    with (out/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
for row in formal:
    independent=next(r for r in fid_audit if r['family']==row['family'] and r['N']==row['N'])
    assert abs(independent['fid']-row['fid'])<1e-8
save('comparison.csv',formal)
def raw(arm,n):return np.load(p/'output/runs'/arm/'generated.npy') if n==32 else np.load(root/'output/runs'/arm/f'generated-{n}.npy')
def score(arm,n):return next(r['fid'] for r in formal if r['family']==arm and r['N']==n)
def pair(a,b):
    e=a.astype(np.float64)-b.astype(np.float64)
    return float(np.sqrt(np.mean(e**2))),float(np.max(np.abs(e)))
drifts=[]
for arm in study.ARMS:
    values={n:raw(arm,n) for n in study.GRIDS};rms={}
    for left,right in ((32,64),(64,128),(32,128)):
        r,m=pair(values[left],values[right]);rms[(left,right)]=r
        drifts.append(dict(family=arm,left_N=left,right_N=right,pixel_RMS=r,pixel_max=m))
    ratio=rms[(32,64)]/rms[(64,128)] if rms[(64,128)] else None
    for row in drifts[-3:]:row['successive_RMS_ratio']=ratio
save('sample-differences.csv',drifts)
contrasts=[]
for ode in ('ode_k1','ode_k2'):
    for baseline in ('swiglu86','swiglu129'):
        for n in study.GRIDS:
            delta=score(ode,n)-score(baseline,n)
            contrasts.append(dict(family=ode,reference=baseline,N=n,FID_difference=delta,ODE_lower=delta<0))
for n in study.GRIDS:
    contrasts.append(dict(family='ode_k2',reference='ode_k1',N=n,FID_difference=score('ode_k2',n)-score('ode_k1',n),ODE_lower=score('ode_k2',n)<score('ode_k1',n)))
save('contrasts.csv',contrasts)
ordinary=[r for r in contrasts if r['reference'].startswith('swiglu')]
reversal=any(r['FID_difference']>=0 for r in ordinary)
decision=dict(ODE_ordinary_ranking_reversal=reversal,all_ODE_ordinary_leads_persist=not reversal,
    cycle17_original_fitting_gate=False,cycle17_original_order_gate=False,new_architecture_nominated=False,
    no_training=True,finest_grid_is_not_ground_truth=True,
    note='Report all meshes; no best solver selected. Persistence only concerns these fixed numerical distributions.')
study.write(out/'decision.json',decision)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=dict(zip(study.ARMS,('#377eb8','#ff7f00','#4daf4a','#984ea3')))
names={'ode_k1':'ODE K1','ode_k2':'ODE K2','swiglu86':'Swi86','swiglu129':'Swi129'}
def export(figure,name):
    for ext in ('png','pdf'):figure.savefig(fig/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(figure)
f,ax=plt.subplots(1,2,figsize=(11,4))
for arm in study.ARMS:
    ax[0].plot(study.GRIDS,[score(arm,n) for n in study.GRIDS],'-o',color=colors[arm],label=names[arm])
    for other in ('swiglu86','swiglu129'):
        if arm.startswith('ode'):ax[1].plot(study.GRIDS,[score(arm,n)-score(other,n) for n in study.GRIDS],'-o' if other=='swiglu86' else '--s',color=colors[arm],label=names[arm]+' - '+names[other])
ax[0].set_ylabel('FID (1,024 generated / 2,048 reference)');ax[1].set_ylabel('FID difference (negative favors ODE)');ax[1].axhline(0,color='black',linewidth=.8)
for pane in ax:pane.set_xticks(study.GRIDS);pane.set_xlabel('Uniform Heun steps');pane.grid(alpha=.2);pane.legend(fontsize=8)
f.suptitle('Frozen models and identical Gaussian starts; no solver selected');f.tight_layout();export(f,'FID-resolution')
f,ax=plt.subplots(1,2,figsize=(11,4))
for arm in study.ARMS:
    rows=[r for r in drifts if r['family']==arm]
    ax[0].plot(range(3),[r['pixel_RMS'] for r in rows],'-o',color=colors[arm],label=names[arm])
    ax[1].plot(study.GRIDS,[next(r['out_of_range_fraction'] for r in formal if r['family']==arm and r['N']==n)*100 for n in study.GRIDS],'-o',color=colors[arm],label=names[arm])
ax[0].set_xticks(range(3),['32 vs 64','64 vs 128','32 vs 128']);ax[0].set_ylabel('Raw endpoint pixel RMS difference');ax[0].set_yscale('log')
ax[1].set_xticks(study.GRIDS);ax[1].set_xlabel('Uniform Heun steps');ax[1].set_ylabel('Raw pixels outside [-1, 1] (%)')
for pane in ax:pane.grid(alpha=.2);pane.legend(fontsize=8)
f.suptitle('Finite-grid endpoint changes; N=128 is not exact flow');f.tight_layout();export(f,'sample-resolution')
f,axs=plt.subplots(4,3,figsize=(12,9))
for i,arm in enumerate(study.ARMS):
    for j,n in enumerate(study.GRIDS):
        images=raw(arm,n)[:8];pixels=np.rint((np.clip(images,-1,1)+1)*np.float32(127.5)).astype(np.uint8).transpose(0,2,3,1)
        tiled=np.concatenate([np.concatenate(pixels[k:k+4],axis=1) for k in (0,4)],axis=0)
        axs[i,j].imshow(tiled);axs[i,j].axis('off');axs[i,j].set_title(f'{names[arm]}, N={n}')
f.suptitle('First eight fixed starts at every registered mesh (display clamp only)');f.tight_layout();export(f,'generation-resolution')
table='| Model | FID N=32 | FID N=64 | FID N=128 |\n|---|---:|---:|---:|\n'
for arm in study.ARMS:table+='| '+names[arm]+' | '+' | '.join(f'{score(arm,n):.6f}' for n in study.GRIDS)+' |\n'
(out/'numeric-table.md').write_text(table)
for label,path in [('sampling-audit.json',root/'audit-sampling/result.json'),('feature-audit.json',root/'audit-features/result.json'),('preflight.json',root/'preflight.json')]:
    (out/label).write_bytes(path.read_bytes())
resources=[json.loads(line) for line in (root/'output/resources.jsonl').read_text().splitlines()]
peak={arm:max(v['rss'] for r in resources for v in r['processes'] if v['family']==arm)/2**30 for arm in study.ARMS}
with (out/'results.tsv').open('w') as file:
    file.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    for row in formal:file.write(f"9ea74f7\t{row['fid']:.12g}\t{peak[row['family']]:.3f}\tkeep\t{row['family']} N={row['N']}: frozen diagnostic, not a nominated solver or architecture; sampled CPU RSS\n")
print(json.dumps(dict(status='CURATED',figures=3,decision=decision)),flush=True)
