"""Retain all eleven endpoints, the declared decision and fixed examples."""
import argparse,csv,json,shutil,sys
from pathlib import Path
import numpy as np

ORDER=('ode_k1','ode_k1_time','ode_k1_pooled','ode_k1_constant','ode_k2','ode_k2_constant','swiglu86','swiglu129','silu64','silu131','silu193')
LABELS=dict(zip(ORDER,('K1 native','K1 time','K1 pooled','K1 old constant','K2 native','K2 old constant','Swi86','Swi129','SiLU64','SiLU131','SiLU193')))
def table(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main(root):
    sys.path.insert(0,str(root/'source'));import study
    study.verify(root,True);assert study.load(root/'orchestrator/status.json')['state']=='COMPLETE'
    phases=study.load(root/'orchestrator/launches.json');assert len(phases)==4 and all(r['exit_code']==0 for r in phases)
    for name in ('sampling','features'):assert study.load(root/f'audit-{name}/result.json')['status']=='PASS'
    formal=study.load(root/'metric-output/result.json')['rows'];assert {r['family'] for r in formal}==set(ORDER)
    decision=study.load(root/'metric-output/decision.json');assert decision==study.decision(formal)
    audited=study.load(root/'audit-features/result.json')['rows'];rows=[]
    for name in ORDER:
        r=dict(next(r for r in formal if r['family']==name));assert abs(r['fid']-next(v['fid'] for v in audited if v['family']==name))<1e-8
        if name.endswith('_constant'):r.update(active_parameters=185744,clock_constants=6)
        elif name not in study.ARMS:r.update(active_parameters=r['parameters'],clock_constants=0)
        rows.append(r)
    out=root/'curation';out.mkdir();figures=out/'figures';figures.mkdir()
    fields=('family','step','weighted_mse','clean_mse','fid','mean_term','covariance_term','out_of_range_fraction','parameters',
        'active_parameters','clock_constants','FFN_forward_MACs','generated','reference','Heun_steps')
    table(out/'comparison.csv',[{k:r[k] for k in fields} for r in rows]);study.write(out/'decision.json',decision)
    with np.load(root/'native-calibration.npz') as f:raw={k:f[k] for k in f.files}
    means,pooled=study.calibration(root);profile=[];poolrows=[]
    prior=study.load(study.PARENTS['clocks20k']/'calibration.json')['ode_k1']
    for name in means:
        values=raw[name].astype(np.float64)
        poolrows.append(dict(layer=name,pooled=pooled[name],old_constant=prior[name]['value'],native_SD=float(values.std()),minimum=float(values.min()),maximum=float(values.max()),count=values.size))
        for j in range(129):profile.append(dict(layer=name,time_index=j,time=j/128,value=float(means[name][j]),native_mean=float(values[j].mean()),
            native_SD=float(values[j].std()),native_min=float(values[j].min()),native_max=float(values[j].max()),count=4096))
    table(out/'calibration-profile.csv',profile);table(out/'calibration-pooled.csv',poolrows)
    panels=[]
    for t in (.1,.3,.5,.7,.9):
        with np.load(study.PARENTS['native']/'banks'/f'panel_{t:.1f}.npz') as f:target=f['target'].astype(np.float64);delta=f['delta'].astype(np.float64)[:,None,None,None]
        for name in ORDER:
            if name in study.ARMS:path=root/'output/runs'/name/f'panel-{t:.1f}.npz'
            elif name.endswith('_constant'):path=study.PARENTS['clocks20k']/'output/runs'/name.removesuffix('_constant')/f'panel-{t:.1f}.npz'
            else:path=study.PARENTS['budget']/'output/runs'/name/f'panel_0020000_{t:.1f}.npz'
            with np.load(path) as f:error=f['prediction'].astype(np.float64)-target
            panels.append(dict(family=name,time=t,weighted_mse=float(np.mean((error/delta)**2)),clean_mse=float(np.mean(error**2)),examples=64))
    table(out/'time-panels.csv',panels)
    resources=[json.loads(line) for line in (root/'output/resources.jsonl').read_text().splitlines()];costs=[]
    for arm in study.ARMS:
        maximum=max(r['RSS'] for line in resources for r in line['resources'] if r['arm']==arm)
        r=study.load(root/'output/runs'/arm/'result.json');costs.append(dict(family=arm,sampling_seconds=r['sampling_seconds'],peak_CPU_RSS_bytes=maximum,co_running_workers=2))
    table(out/'resource-summary.csv',costs)
    with (root/'results.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(('commit','metric','memory_gb','status','description'))
        for r in rows:
            new=r['family'] in study.ARMS;memory=next((v['peak_CPU_RSS_bytes']/2**30 for v in costs if v['family']==r['family']),None)
            status='discard' if r['family']=='ode_k1_time' and not decision['time_specific_explanation_supported'] else 'keep'
            w.writerow((study.load(root/'settings.json')['source_commit'][:7] if new else 'prior',format(r['fid'],'.12f'),format(memory,'.3f') if memory is not None else 'NA',status,
                r['family']+': '+('frozen time-specific screen; not a trained architecture' if r['family']=='ode_k1_time' else 'reference/control evidence only; not nominated')))
    shutil.copyfile(root/'results.tsv',out/'results.tsv')
    for src,target in [('preflight.json','preflight.json'),('metric-preflight.json','metric-preflight.json'),('documented-pass.json','agent-doc-pass.json'),
        ('audit-sampling/result.json','sampling-audit.json'),('audit-features/result.json','feature-audit.json')]:shutil.copyfile(root/src,out/target)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def export(fig,name):
        for ext in ('png','pdf'):fig.savefig(figures/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained');labels=[LABELS[r['family']] for r in rows]
    color=['#2378a8','#3a923a','#e1812c','#b38b6d',*['#8b98a3']*7]
    for ax,key,title in zip(axes,('fid','weighted_mse'),('FID: lower is better','Weighted validation MSE: lower is better')):
        ax.bar(range(11),[r[key] for r in rows],color=color);ax.set_xticks(range(11),labels,rotation=55,ha='right');ax.set_title(title)
    export(fig,'all-endpoints')
    fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    for ax,poolrow in zip(axes.flat,poolrows):
        selected=[r for r in profile if r['layer']==poolrow['layer']];x=np.array([r['time'] for r in selected]);y=np.array([r['value'] for r in selected]);sd=np.array([r['native_SD'] for r in selected])
        ax.plot(x,y,color='#2378a8',label='per-time mean');ax.fill_between(x,y-sd,y+sd,color='#2378a8',alpha=.15,label='native token/image SD')
        ax.axhline(poolrow['pooled'],color='#e1812c',ls='--',label='matched pooled');ax.axhline(poolrow['old_constant'],color='#9a78a4',ls=':',label='old constant')
        ax.set(title=poolrow['layer'],xlabel='External time',ylabel='Local duration',ylim=(.2,1.0))
    axes[0,0].legend(fontsize=7);export(fig,'clock-calendar')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');selected_names=('ode_k1','ode_k1_time','ode_k1_pooled','ode_k1_constant','swiglu86','silu64','silu131')
    for name in selected_names:
        subset=[r for r in panels if r['family']==name]
        for ax,key in zip(axes,('weighted_mse','clean_mse')):ax.plot([r['time'] for r in subset],[r[key] for r in subset],'-o',label=LABELS[name],markersize=3)
    for ax,key in zip(axes,('weighted_mse','clean_mse')):ax.set(xlabel='Fixed panel time',ylabel=key)
    axes[0].legend(fontsize=7);export(fig,'time-panels')
    fig,axes=plt.subplots(11,8,figsize=(10,13),layout='constrained')
    for row_index,name in enumerate(ORDER):
        images=np.load(study.sample_bank(root,name))[:8];images=np.rint((np.clip(images,-1,1)+1)*127.5).clip(0,255).astype(np.uint8)
        for column in range(8):
            ax=axes[row_index,column];ax.imshow(images[column].transpose(1,2,0),interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            if column==0:ax.set_ylabel(LABELS[name],rotation=0,ha='right',va='center',fontsize=8)
            if row_index==0:ax.set_title(f'Start {column}',fontsize=8)
    export(fig,'fixed-examples')
    study.write(out/'summary.json',dict(status='PASS',endpoints=11,new_interventions=2,calendar_values=774,pooled_values=6,time_panels=55,
        fixed_examples=88,figures=4,new_training=False,new_trained_method=False))
    study.verify(root,True);print(json.dumps(study.load(out/'summary.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
