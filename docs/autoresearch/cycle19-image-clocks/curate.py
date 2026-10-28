"""All six fixed image endpoints and all predeclared example/time panels."""
from pathlib import Path
import argparse,csv,json,sys,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root
sys.path.insert(0,str(root/'source'));import study
study.verify(root,True);assert study.load(root/'orchestrator/status.json')['state']=='COMPLETE'
assert study.load(root/'audit-sampling/result.json')['status']==study.load(root/'audit-features/result.json')['status']=='PASS'
out=root/'curation';out.mkdir();fig=out/'figures';fig.mkdir();p=study.paths(root);old=study.imported(root,'image_parent')
rows=study.load(root/'metric-output/result.json')['rows'];assert len(rows)==6
for row in rows:
    for name in ('weighted_mse','clean_mse','fid','out_of_range_fraction','mean_term','covariance_term'):row[name]=float(row[name])
order=('ode_k1','ode_k1_constant','ode_k2','ode_k2_constant','swiglu86','swiglu129');rows=sorted(rows,key=lambda r:order.index(r['family']))
names={'ode_k1':'K1 native','ode_k1_constant':'K1 constant','ode_k2':'K2 native','ode_k2_constant':'K2 constant','swiglu86':'Swi86','swiglu129':'Swi129'}
by={r['family']:r for r in rows}
def save(name,values):
    keys=list(dict.fromkeys(k for row in values for k in row))
    with (out/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(values)
save('comparison.csv',rows);best=min(by[x]['fid'] for x in ('swiglu86','swiglu129'));contrasts=[]
for arm in study.ARMS:
    native=by[arm];fixed=by[arm+'_constant'];lead=best-native['fid'];remaining=best-fixed['fid']
    contrasts.append(dict(family=arm,delta_weighted_mse=fixed['weighted_mse']-native['weighted_mse'],weighted_mse_ratio=fixed['weighted_mse']/native['weighted_mse'],delta_clean_mse=fixed['clean_mse']-native['clean_mse'],delta_FID=fixed['fid']-native['fid'],native_lead_over_best_ordinary=lead,constant_lead_over_best_ordinary=remaining,lead_fraction_retained=remaining/lead,constant_beats_both_ordinary=all(fixed['fid']<by[x]['fid'] for x in ('swiglu86','swiglu129'))))
save('contrasts.csv',contrasts)
panels=[]
for arm in order:
    for t in (.1,.3,.5,.7,.9):
        b=old.bank(p['image_parent']/'banks'/f'panel_{t:.1f}.npz')
        if arm.endswith('_constant'):path=root/'output/runs'/arm.removesuffix('_constant')/f'panel-{t:.1f}.npz'
        else:path=p['image_parent']/'output/runs'/arm/f'panel_0005000_{t:.1f}.npz'
        with np.load(path) as f:pred=f['prediction']
        panels.append(dict(family=arm,time=t,**old.metrics(pred,b)))
save('time-panels.csv',panels)
clockrows=[dict(family=arm,layer=name,**value) for arm,values in study.load(root/'calibration.json').items() for name,value in values.items()];save('calibration.csv',clockrows)
study.write(out/'decision.json',dict(new_architecture_nominated=False,cycle17_fitting_gate=False,cycle17_order_gate=False,contrasts=contrasts,scope='One simultaneous frozen-clock intervention, exposed calibration; no retraining or unique causal attribution'))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=['#377eb8','#85b7de','#ff7f00','#ffba75','#4daf4a','#984ea3']
def export(f,name):
    for extension in ('png','pdf'):f.savefig(fig/f'{name}.{extension}',dpi=180,bbox_inches='tight')
    plt.close(f)
f,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,metric,label in zip(axes,('weighted_mse','fid'),('Paired weighted validation MSE','FID (1,024 / 2,048), Heun128')):
    for i,row in enumerate(rows):
        ax.scatter(i,row[metric],c=colors[i],s=65);ax.annotate(f"{row[metric]:.4f}" if metric=='weighted_mse' else f"{row[metric]:.2f}",(i,row[metric]),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
    ax.set_xticks(range(6),[names[k].replace(' ','\n') for k in order]);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.margins(y=.2)
axes[0].set_yscale('log');axes[1].axhline(best,color='gray',linestyle='--',alpha=.6,label='Better ordinary FID');axes[1].legend(fontsize=8)
f.suptitle('Frozen input-clock collapse; all learned tensors unchanged');f.tight_layout();export(f,'fixed-comparison')
f,axes=plt.subplots(1,2,figsize=(11,4))
for ax,arm in zip(axes,study.ARMS):
    for family,color,style in ((arm,'#377eb8','-o'),(arm+'_constant','#ff7f00','-s'),('swiglu86','#4daf4a','--'),('swiglu129','#984ea3','--')):
        r=[q for q in panels if q['family']==family];ax.plot([q['time'] for q in r],[q['weighted_mse'] for q in r],style,label=names[family],color=color)
    ax.set_xlabel('Fixed interpolation time');ax.set_ylabel('Paired weighted MSE, log scale');ax.set_yscale('log');ax.set_title(names[arm]);ax.grid(alpha=.2);ax.legend(fontsize=8)
f.suptitle('All five exposed validation panels, 64 images each');f.tight_layout();export(f,'time-panels')
def raw(arm):return np.load(root/'output/runs'/arm.removesuffix('_constant')/'generated.npy') if arm.endswith('_constant') else np.load(p['resolution_parent']/'output/runs'/arm/'generated-128.npy')
f,axes=plt.subplots(6,1,figsize=(8,15))
for ax,arm in zip(axes,order):
    x=raw(arm)[:8];images=np.rint((np.clip(x,-1,1)+1)*np.float32(127.5)).astype(np.uint8).transpose(0,2,3,1)
    tile=np.concatenate([np.concatenate(images[k:k+4],axis=1) for k in (0,4)],axis=0);ax.imshow(tile);ax.axis('off');ax.set_title(names[arm])
f.suptitle('Same first eight Gaussian starts; display clamp only');f.tight_layout();export(f,'generation-cases')
table='| Model | Weighted MSE | Clean MSE | FID Heun128 |\n|---|---:|---:|---:|\n'
for row in rows:table+=f"| {names[row['family']]} | {row['weighted_mse']:.6f} | {row['clean_mse']:.6f} | {row['fid']:.4f} |\n"
(out/'numeric-table.md').write_text(table)
for name,source in [('sampling-audit.json',root/'audit-sampling/result.json'),('feature-audit.json',root/'audit-features/result.json'),('preflight.json',root/'preflight.json'),('calibration.json',root/'calibration.json')]:shutil.copyfile(source,out/name)
resources=[json.loads(line) for line in (root/'output/resources.jsonl').read_text().splitlines()]
peak={arm:max(v['rss'] for r in resources for v in r['processes'] if v['family']==arm)/2**30 for arm in study.ARMS}
with (out/'results.tsv').open('w') as f:
    f.write('commit\tmetric\tmemory_gb\tstatus\tdescription\n')
    for row in rows:
        memory=f"{peak[row['family'].removesuffix('_constant')]:.3f}" if row['family'].endswith('_constant') else 'NA'
        f.write(f"baeaace\t{row['fid']:.12g}\t{memory}\tkeep\t{row['family']}: {row['provenance']}; no new architecture nominated\n")
print(json.dumps(dict(status='CURATED',contrasts=contrasts,figures=3)),flush=True)
