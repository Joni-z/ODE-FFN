"""All endpoints, arithmetic controls, operator histories and fixed cases."""
from pathlib import Path
import argparse,csv,json,sys,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);a=parser.parse_args();root=a.root
sys.path.insert(0,str(root/'source'));import study
study.verify(root,True);assert study.load(root/'orchestrator/status.json')['state']=='COMPLETE'
audits={k:study.load(root/f'audit-{k}/result.json') for k in ('sampling','features')}
assert all(v['status']=='PASS' for v in audits.values())
assert audits['sampling']['operator_records']==48 and audits['sampling']['new_generated_images']==4096
assert len(audits['features']['rows'])==13
out=root/'curation';out.mkdir();fig=out/'figures';fig.mkdir();p=study.paths(root);old=study.imported(root,'image_parent')
rows=study.load(root/'metric-output/result.json')['rows'];assert len(rows)==13
order=('ode_k1','ode_k2','swiglu86','swiglu129','ode_k1_constant','ode_k2_constant','silu64','silu131','silu193',*study.ARMS)
names=dict(ode_k1='ODE K1',ode_k2='ODE K2',swiglu86='Swi86',swiglu129='Swi129',ode_k1_constant='K1 constant',ode_k2_constant='K2 constant',silu64='SiLU64',silu131='SiLU131',silu193='SiLU193',ode_k1_folded='K1 folded',ode_k1_exp='K1 exp',ode_k2_folded='K2 folded',ode_k2_exp='K2 exp')
for r in rows:
    for key in ('weighted_mse','clean_mse','fid','mean_term','covariance_term','out_of_range_fraction'):r[key]=float(r[key])
rows.sort(key=lambda r:order.index(r['family']));by={r['family']:r for r in rows}
def save(name,values):
    keys=list(dict.fromkeys(k for r in values for k in r))
    with (out/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(values)
save('comparison.csv',rows)
def rawpath(arm):
    if arm in study.ARMS:return root/'output/runs'/arm/'generated.npy'
    if arm.startswith('silu'):return p['ordinary_parent']/'output/runs'/arm/'generated.npy'
    if arm.endswith('_constant'):return p['clock_parent']/'output/runs'/arm.removesuffix('_constant')/'generated.npy'
    return p['resolution_parent']/'output/runs'/arm/'generated-128.npy'
def predpath(arm,t=None):
    if arm in study.ARMS:return root/'output/runs'/arm/('validation.npz' if t is None else f'panel-{t:.1f}.npz')
    if arm.startswith('silu'):return p['ordinary_parent']/'output/runs'/arm/('validation_0005000.npz' if t is None else f'panel_0005000_{t:.1f}.npz')
    if arm.endswith('_constant'):return p['clock_parent']/'output/runs'/arm.removesuffix('_constant')/('validation.npz' if t is None else f'panel-{t:.1f}.npz')
    return p['image_parent']/'output/runs'/arm/('validation_0005000.npz' if t is None else f'panel_0005000_{t:.1f}.npz')
def prediction(arm,t=None):
    with np.load(predpath(arm,t)) as f:return f['prediction']
contrasts=[]
for base in ('ode_k1','ode_k2'):
    for arm,ref in ((base+'_exp',base+'_folded'),(base+'_folded',base+'_constant'),(base+'_exp',base+'_constant')):
        x,y=by[arm],by[ref]
        contrasts.append(dict(family=arm,reference=ref,weighted_ratio=x['weighted_mse']/y['weighted_mse'],clean_ratio=x['clean_mse']/y['clean_mse'],FID_delta=x['fid']-y['fid'],prediction_RMS=float(np.sqrt(np.mean((prediction(arm).astype(np.float64)-prediction(ref).astype(np.float64))**2))),sample_RMS=float(np.sqrt(np.mean((np.load(rawpath(arm)).astype(np.float64)-np.load(rawpath(ref)).astype(np.float64))**2)))))
save('contrasts.csv',contrasts)
panels=[]
for arm in order:
    for t in (.1,.3,.5,.7,.9):
        bank=old.bank(p['image_parent']/'banks'/f'panel_{t:.1f}.npz');panels.append(dict(family=arm,time=t,**old.metrics(prediction(arm,t),bank)))
save('time-panels.csv',panels)
history=study.load(root/'operator-history.json');operator_rows=[]
for native in history:
    r={k:v for k,v in native.items() if not isinstance(v,list)}
    for key,prefix in [('P_singular_values','sigma'),('P_eigen_real','eigen_real'),('P_eigen_imag','eigen_imag')]:
        for i,value in enumerate(native[key]):r[f'{prefix}_{i:02d}']=value
    operator_rows.append(r)
save('operator-history.csv',operator_rows)
study.write(out/'decision.json',dict(new_ODE_architecture_nominated=False,new_training=False,old_stops_unchanged=True,contrasts=contrasts,scope='Four frozen image interventions with two matched-arithmetic controls; no exact-flow training selected'))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=plt.cm.tab20(np.linspace(0,1,len(order)))
def export(f,name):
    for ext in ('png','pdf'):f.savefig(fig/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(f)
f,axes=plt.subplots(1,2,figsize=(18,5))
for ax,key,label in zip(axes,('weighted_mse','fid'),('Paired weighted MSE','FID, Heun128 (1,024 generated / 2,048 reference)')):
    for i,arm in enumerate(order):
        value=by[arm][key];ax.scatter(i,value,c=[colors[i]],s=45);ax.annotate(f'{value:.4f}' if key=='weighted_mse' else f'{value:.2f}',(i,value),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7)
    ax.set_xticks(range(len(order)),[names[k].replace(' ','\n') for k in order],fontsize=7);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.margins(y=.18)
f.suptitle('All nine reused endpoints and all four frozen-map interventions');f.tight_layout();export(f,'fixed-comparison')
f,axes=plt.subplots(2,2,figsize=(12,8))
for i,arm in enumerate(('ode_k1','ode_k2')):
    for j in range(3):
        for k,part in enumerate(('ode1','ode2')):
            values=[r for r in history if r['family']==arm and r['layer']==f'blocks.{j}.{part}'];x=[r['step'] for r in values];label=f'Block{j+1} {part}'
            axes[i,0].plot(x,[r['relative_Frobenius_error'] for r in values],'-o' if k==0 else '--s',label=label)
            axes[i,1].plot(x,[r['P_sigma_max'] for r in values],'-o' if k==0 else '--s',label=label)
    axes[i,0].set_ylabel(f"{names[arm]}: ||E-T||F / ||E||F");axes[i,1].set_ylabel(f"{names[arm]}: largest singular value of P")
for ax in axes.flat:ax.set_xlabel('Training update');ax.legend(fontsize=7);ax.grid(alpha=.2)
f.suptitle('Fixed final durations applied retrospectively; no clock recalibration');f.tight_layout();export(f,'operator-histories')
f,axes=plt.subplots(5,3,figsize=(14,13))
for ax in axes.flat:ax.axis('off')
for ax,arm in zip(axes.flat,order):
    x=np.load(rawpath(arm))[:8];images=np.rint((np.clip(x,-1,1)+1)*np.float32(127.5)).astype(np.uint8).transpose(0,2,3,1)
    tile=np.concatenate([np.concatenate(images[k:k+4],axis=1) for k in (0,4)],axis=0);ax.imshow(tile);ax.set_title(names[arm])
f.suptitle('Same first eight starts for every endpoint; no case selection');f.tight_layout();export(f,'generation-cases')
table='| Model | Weighted MSE | Clean MSE | FID128 | Raw out-of-range % |\n|---|---:|---:|---:|---:|\n'
for arm in order:
    r=by[arm];table+=f"| {names[arm]} | {r['weighted_mse']:.6f} | {r['clean_mse']:.6f} | {r['fid']:.4f} | {100*r['out_of_range_fraction']:.4f} |\n"
(out/'numeric-table.md').write_text(table)
for name,source in [('sampling-audit.json',root/'audit-sampling/result.json'),('feature-audit.json',root/'audit-features/result.json'),('preflight.json',root/'preflight.json'),('agent-doc-pass.json',root/'agent-doc-pass.json')]:shutil.copyfile(source,out/name)
resources=[json.loads(line) for line in (root/'output/resources.jsonl').read_text().splitlines()]
peaks={arm:max(v['rss'] for r in resources for v in r['processes'] if v['family']==arm)/2**30 for arm in study.ARMS}
with (out/'results.tsv').open('w') as f:
    f.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    for r in rows:
        new=r['family'] in study.ARMS;f.write(f"{study.load(root/'settings.json')['source_commit'][:7] if new else 'prior'}\t{r['fid']:.12g}\t{str(round(peaks[r['family']],3)) if new else 'NA'}\tkeep\t{r['family']}: {'new frozen intervention' if new else 'explicitly reused endpoint'}; no new ODE nomination\n")
print(json.dumps(dict(status='CURATED',models=13,figures=3,operator_records=48)),flush=True)
