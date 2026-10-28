"""Curate all nine endpoints, all native/cold curves and matrix histories."""
from pathlib import Path
import argparse,csv,json,sys,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
sys.path.insert(0,str(root/'source'));import study
study.verify(root);assert study.load(root/'orchestrator/status.json')['state']=='COMPLETE'
training=study.load(root/'audit-training/result.json');features=study.load(root/'audit-features/result.json');assert training['status']==features['status']=='PASS'
assert [training[k] for k in ('checkpoints','time_panels','matrix_records','generated_images','stream_batches')]==[33,60,72,3072,5000]
assert len(features['rows'])==9
out=root/'curation';out.mkdir();fig=out/'figures';fig.mkdir();p=study.paths(root);old=study.imported(root,'image_parent')
rows=study.load(root/'metric-output/result.json')['rows'];assert len(rows)==9
order=('ode_k1','ode_k2','swiglu86','swiglu129','ode_k1_constant','ode_k2_constant',*study.ARMS)
names={'ode_k1':'ODE K1','ode_k2':'ODE K2','swiglu86':'Swi86','swiglu129':'Swi129','ode_k1_constant':'K1 constant','ode_k2_constant':'K2 constant','silu64':'SiLU64','silu131':'SiLU131','silu193':'SiLU193'}
costs={r['family']:r for r in study.load(root/'preflight.json')['arms']}
prior_costs={r['family']:r for r in csv.DictReader((p['image_parent']/'curation/comparison.csv').open())}
for row in rows:
    for key in ('weighted_mse','clean_mse','fid','mean_term','covariance_term','out_of_range_fraction'):row[key]=float(row[key])
    arm=row['family']
    if arm in costs:row.update(parameters=costs[arm]['parameters'],FFN_forward_MACs=costs[arm]['FFN_forward_MACs'],FFN_training_MACs=costs[arm]['FFN_training_MACs'])
    elif arm in prior_costs:
        for key in ('parameters','FFN_forward_MACs','FFN_training_MACs'):row[key]=int(prior_costs[arm][key])
    else:row.update(parameters='',FFN_forward_MACs='',FFN_training_MACs='',cost_note='Frozen intervention, no new optimization; original training cost not a new matched arm')
rows.sort(key=lambda r:order.index(r['family']));by={r['family']:r for r in rows}
def save(name,values):
    keys=list(dict.fromkeys(k for row in values for k in row))
    with (out/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(values)
save('comparison.csv',rows)
contrasts=[]
for arm in study.ARMS:
    for ref in order[:6]:
        x,y=by[arm],by[ref];contrasts.append(dict(family=arm,reference=ref,weighted_ratio=x['weighted_mse']/y['weighted_mse'],clean_ratio=x['clean_mse']/y['clean_mse'],FID_delta=x['fid']-y['fid'],FID_ratio=x['fid']/y['fid'],both_quality_lower=x['weighted_mse']<y['weighted_mse'] and x['fid']<y['fid']))
save('contrasts.csv',contrasts)
curves=[]
for source,provenance in ((p['image_parent']/'audit-training/curves.csv','sealed native trajectory'),(root/'audit-training/curves.csv','new cold ordinary trajectory')):
    for r in csv.DictReader(source.open()):curves.append(dict(family=r['family'],step=int(r['step']),weighted_mse=float(r['weighted_mse']),clean_mse=float(r['clean_mse']),provenance=provenance))
assert len(curves)==77;save('curves.csv',curves)
matrices=[];telemetry=[]
for arm in study.ARMS:
    folder=root/'output/runs'/arm
    for step in study.SNAPSHOTS:
        for r in study.load(folder/f'matrices_{step:07d}.json'):
            v=dict(family=arm,step=step,layer=r['layer'],shape='x'.join(map(str,r['shape'])),frobenius=r['frobenius'],sigma_max=r['sigma_max'],singular_CV=r['singular_CV'])
            for key,prefix in [('singular_values','sigma'),('eigen_real','eigen_real'),('eigen_imag','eigen_imag')]:
                for i,value in enumerate(r.get(key,[])):v[f'{prefix}_{i:02d}']=value
            matrices.append(v)
    for line in (folder/'metrics.jsonl').read_text().splitlines():telemetry.append(dict(family=arm,**json.loads(line)))
assert len(matrices)==72 and len(telemetry)==33;save('matrix-history.csv',matrices);save('training-telemetry.csv',telemetry)
def prediction_path(arm,time=None):
    if arm in study.ARMS:folder=root/'output/runs'/arm;return folder/('validation_0005000.npz' if time is None else f'panel_0005000_{time:.1f}.npz')
    if arm.endswith('_constant'):folder=p['clock_parent']/'output/runs'/arm.removesuffix('_constant');return folder/('validation.npz' if time is None else f'panel-{time:.1f}.npz')
    folder=p['image_parent']/'output/runs'/arm;return folder/('validation_0005000.npz' if time is None else f'panel_0005000_{time:.1f}.npz')
val=old.bank(p['image_parent']/'banks/validation.npz');distributions=[];panels=[]
for arm in order:
    with np.load(prediction_path(arm)) as f:err=f['prediction'].astype(np.float64)-val['target'].astype(np.float64)
    per=(err/val['delta'].astype(np.float64)[:,None,None,None])**2;per=per.mean(axis=(1,2,3))
    distributions.append(dict(family=arm,n=1024,mean=float(per.mean()),SD=float(per.std(ddof=1)),median=float(np.median(per)),q10=float(np.quantile(per,.1)),q90=float(np.quantile(per,.9))))
    for time in (.1,.3,.5,.7,.9):
        b=old.bank(p['image_parent']/'banks'/f'panel_{time:.1f}.npz')
        with np.load(prediction_path(arm,time)) as f:m=old.metrics(f['prediction'],b)
        panels.append(dict(family=arm,time=time,**m))
save('per-image-distribution.csv',distributions);save('time-panels.csv',panels)
study.write(out/'decision.json',dict(new_ODE_architecture_nominated=False,original_cycle17_gates_unchanged=True,ordinary_contrasts=contrasts,scope='Cold ordinary baseline coverage; cost and FFN initialization differ by the registered roles'))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=dict(zip(order,('#377eb8','#ff7f00','#4daf4a','#984ea3','#8bbad9','#ffbd78','#1b7837','#a65628','#e41a1c')))
def export(f,name):
    for extension in ('png','pdf'):f.savefig(fig/f'{name}.{extension}',dpi=180,bbox_inches='tight')
    plt.close(f)
f,axes=plt.subplots(1,2,figsize=(15,5))
for ax,key,label in zip(axes,('weighted_mse','fid'),('Paired weighted MSE','FID (1,024 generated / 2,048 reference), Heun128')):
    for i,arm in enumerate(order):
        value=by[arm][key];ax.scatter(i,value,c=colors[arm],s=55);ax.annotate(f'{value:.4f}' if key=='weighted_mse' else f'{value:.2f}',(i,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7)
    ax.set_xticks(range(9),[names[k].replace(' ','\n') for k in order],fontsize=8);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.margins(y=.18)
axes[0].set_yscale('log');f.suptitle('All cold ordinary controls and explicitly reused endpoints');f.tight_layout();export(f,'fixed-comparison')
f,axes=plt.subplots(1,2,figsize=(12,4))
for arm in (*order[:4],*study.ARMS):
    values=[r for r in curves if r['family']==arm]
    for ax,key in zip(axes,('weighted_mse','clean_mse')):ax.plot([r['step'] for r in values],[r[key] for r in values],'-' if arm in study.ARMS else '--',color=colors[arm],label=names[arm],linewidth=1.8 if arm in study.ARMS else 1.)
for ax,label in zip(axes,('Paired weighted MSE','Clean pixel MSE')):ax.set_xlabel('Training update');ax.set_ylabel(label);ax.set_yscale('log');ax.grid(alpha=.2);ax.legend(fontsize=8)
f.suptitle('All recorded checkpoints; constant-clock endpoints were not retrained');f.tight_layout();export(f,'learning-curves')
f,axes=plt.subplots(1,3,figsize=(13,4))
for ax,arm in zip(axes,study.ARMS):
    for j,color in enumerate(('#377eb8','#ff7f00','#4daf4a')):
        for name,style in (('fc1','-o'),('fc2','--s')):
            values=[r for r in matrices if r['family']==arm and r['layer']==f'blocks.{j}.{name}'];ax.plot([r['step'] for r in values],[r['sigma_max'] for r in values],style,color=color,label=f'Block{j+1} {name}')
    ax.set_title(names[arm]);ax.set_xlabel('Update');ax.set_ylabel('Largest singular value of W');ax.grid(alpha=.2);ax.legend(fontsize=7)
f.suptitle('All ordinary matrix histories; these are W maps, not ODE generators');f.tight_layout();export(f,'matrix-histories')
def raw(arm):
    if arm in study.ARMS:return np.load(root/'output/runs'/arm/'generated.npy')
    if arm.endswith('_constant'):return np.load(p['clock_parent']/'output/runs'/arm.removesuffix('_constant')/'generated.npy')
    return np.load(p['resolution_parent']/'output/runs'/arm/'generated-128.npy')
f,axes=plt.subplots(3,3,figsize=(14,9))
for ax,arm in zip(axes.flat,order):
    x=raw(arm)[:8];images=np.rint((np.clip(x,-1,1)+1)*np.float32(127.5)).astype(np.uint8).transpose(0,2,3,1)
    tile=np.concatenate([np.concatenate(images[k:k+4],axis=1) for k in (0,4)],axis=0);ax.imshow(tile);ax.axis('off');ax.set_title(names[arm])
f.suptitle('Same first eight starts for every registered/reused endpoint');f.tight_layout();export(f,'generation-cases')
table='| Model | Parameters | FFN forward MACs | Weighted MSE | Clean MSE | FID128 |\n|---|---:|---:|---:|---:|---:|\n'
for arm in order:
    r=by[arm];table+=f"| {names[arm]} | {r['parameters'] or 'frozen'} | {r['FFN_forward_MACs'] or 'not a new cost arm'} | {r['weighted_mse']:.6f} | {r['clean_mse']:.6f} | {r['fid']:.4f} |\n"
(out/'numeric-table.md').write_text(table)
for label,source in [('training-audit.json',root/'audit-training/result.json'),('feature-audit.json',root/'audit-features/result.json'),('preflight.json',root/'preflight.json'),('agent-doc-pass.json',root/'agent-doc-pass.json')]:shutil.copyfile(source,out/label)
resources=[json.loads(line) for line in (root/'output/resources.jsonl').read_text().splitlines()]
peaks={arm:max(v['rss'] for r in resources for v in r['processes'] if v['family']==arm)/2**30 for arm in study.ARMS}
with (out/'results.tsv').open('w') as f:
    f.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    for r in rows:f.write(f"3146dd0\t{r['fid']:.12g}\t{peaks[r['family']]:.3f}\tkeep\t{r['family']}: new ordinary baseline only; no new ODE nomination\n" if r['family'] in study.ARMS else f"prior\t{r['fid']:.12g}\tNA\tkeep\t{r['family']}: explicitly reused sealed endpoint\n")
print(json.dumps(dict(status='CURATED',figures=4,models=9,new_ODE_architecture_nominated=False)),flush=True)
