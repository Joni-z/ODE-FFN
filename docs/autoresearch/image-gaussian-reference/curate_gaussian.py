"""Complete tables and fixed cases for the two classical image references."""
import argparse,csv,json,shutil,sys
from pathlib import Path
import numpy as np

ORDER=('ode_k1','fixed06','matched_silu64','ppca59','isotropic','native_posthoc06','ode_k1_time','ode_k1_pooled','ode_k1_constant',
       'ode_k2','ode_k2_constant','swiglu86','swiglu129','silu64','silu131','silu193')
LABELS=dict(zip(ORDER,('K1 native','Fixed 0.6 P','Matched-init SiLU64','PPCA59','Isotropic Gaussian','Native posthoc 0.6',
    'Old K1 time','Old K1 pooled','Old K1 constant','K2 native','Old K2 constant','Swi86','Swi129','Default SiLU64','SiLU131','SiLU193')))
COLORS={'ode_k1':'#0072b2','fixed06':'#d55e00','matched_silu64':'#009e73','ppca59':'#cc79a7','isotropic':'#e69f00','swiglu86':'#332288','silu131':'#88ccee'}
def table(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def main(root):
    sys.path.insert(0,str(root/'source'));import gaussian_reference as g
    assert g.load(root/'orchestrator/status.json')['state']=='COMPLETE'
    assert g.load(root/'controller-tool-exit.json')['exit_code']==0;g.verify(root)
    out=root/'curation';out.mkdir();figures=out/'figures';figures.mkdir()
    formal=g.load(root/'metric-output/result.json')['rows'];assert len(formal)==34
    old={(r['arm'],int(r['step'])):r for r in csv.DictReader((g.COLD/'curation/comparison.csv').open())}
    rows=[]
    for r in formal:
        statistical=r['arm'] in g.ARMS;prior=old.get((r['arm'],r['step']))
        raw=np.load(r['sample_path']);assert raw.shape==(1024,3,32,32) and raw.dtype==np.float32 and np.isfinite(raw).all()
        rows.append(dict(arm=r['arm'],fit_kind=r['fit_kind'],budget_label='train-only statistical fit' if statistical else str(r['step'])+' updates',
            training_updates=0 if statistical else r['step'],weighted_mse=r['weighted_mse'],clean_mse=r['clean_mse'],fid=r['fid'],
            FID_mean_term=r['FID_mean_term'],FID_covariance_term=r['FID_covariance_term'],raw_tail_fraction=float(np.mean(np.abs(raw)>1)),
            stored_fitted_scalars=r['parameters'],inference_active_values=r['parameters'] if statistical else int(prior['active_learned_parameters']),
            value_precision='float64 statistics; float32 output' if statistical else 'float32 neural',
            counted_matrix_MACs_per_image=r['matrix_MACs_per_image'] if statistical else r['FFN_forward_MACs']*64*3,
            matrix_cost_scope='complete Gaussian predictor; float64; pointwise excluded' if statistical else 'FFN only; float32; backbone/pointwise excluded',
            generated=1024,reference=2048,Heun_steps=128))
    table(out/'comparison.csv',rows)
    primary=sorted([r for r in rows if r['training_updates']==20000 or r['arm'] in g.ARMS],key=lambda r:ORDER.index(r['arm']))
    assert len(primary)==16;table(out/'primary-comparison.csv',primary)
    ratios=[]
    for arm in g.ARMS:
        a=next(r for r in primary if r['arm']==arm)
        for b in primary:
            if b['arm'] in g.ARMS:continue
            ratios.append(dict(arm=arm,reference=b['arm'],**{k+'_ratio':a[k]/b[k] for k in ('fid','weighted_mse','clean_mse')},FID_difference=a['fid']-b['fid']))
    assert len(ratios)==28;table(out/'primary-ratios.csv',ratios)
    panels=[];budget=g.cold().PARENTS['budget']
    for arm in ('ode_k1','fixed06','matched_silu64','swiglu86','silu131',*g.ARMS):
        for t in (.1,.3,.5,.7,.9):
            if arm in g.ARMS:path=root/'sampling'/f'{arm}-panel-{t:.1f}.npz'
            else:path=(g.COLD if arm in ('fixed06','matched_silu64') else budget)/'output/runs'/arm/f'panel_0020000_{t:.1f}.npz'
            with np.load(path) as f:prediction=f['prediction']
            b=g.bank(t);error=prediction.astype(np.float64)-b['target'].astype(np.float64)
            panels.append(dict(arm=arm,time=t,examples=64,weighted_mse=float(np.mean((error/b['delta'].astype(np.float64)[:,None,None,None])**2)),clean_mse=float(np.mean(error**2))))
    assert len(panels)==35;table(out/'time-panels.csv',panels)
    p=g.gm.load_parameters(root/'fit/ppca59.npz');trace=g.load(root/'fit/result.json')['rows'][0]['trace']
    spectrum=[dict(index=i+1,eigenvalue=float(value),kind='retained empirical eigenvalue') for i,value in enumerate(p['eigenvalues'])]
    table(out/'retained-spectrum.csv',spectrum)
    summary=dict(status='PASS',rank=59,stored_scalars=g.gm.stored_scalars(p),stored_float64_bytes=g.gm.stored_scalars(p)*8,
        isotropic_tail_variance=p['tail_variance'],empirical_covariance_trace=trace,
        retained_variance_fraction=float(p['eigenvalues'].sum()/trace),tail_variance_fraction=float((3072-59)*p['tail_variance']/trace),
        total_endpoints=34,primary_rows=16,new_sample_banks=2,primary_ratios=28,time_panel_rows=35,figures=4,fixed_examples=128,
        new_ODE_method=False,neural_mechanism_identified=False)
    g.write(out/'summary.json',summary);g.write(out/'decision.json',g.decisions(formal))
    for src,name in [('preflight.json','full-preflight.json'),('documented-pass.json','full-documented-pass.json'),
        ('audit-fit/result.json','fit-audit.json'),('audit-sampling/result.json','sampling-audit.json'),('audit-features/result.json','feature-audit.json')]:shutil.copyfile(root/src,out/name)
    launches=g.load(root/'orchestrator/launches.json');resources=[json.loads(v) for v in (root/'orchestrator/resources.jsonl').read_text().splitlines()]
    table(out/'resource-summary.csv',[dict(phase=r['phase'],seconds=r['seconds'],peak_CPU_RSS_bytes=max((v['RSS'] for q in resources if q['phase']==r['phase'] for v in q['resources']),default=0)) for r in launches])
    with (root/'results.tsv').open('w') as f:
        writer=csv.writer(f,delimiter='\t');writer.writerow(('commit','metric','memory_gb','status','description'))
        for r in rows:writer.writerow((g.load(root/'settings.json')['source_commit'][:7] if r['arm'] in g.ARMS else 'prior',format(r['fid'],'.12f'),'NA','keep','reference evidence only: '+r['arm']+'; '+r['budget_label']+'; no ODE method nominated'))
    shutil.copyfile(root/'results.tsv',out/'results.tsv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def export(fig,name):
        for ext in ('png','pdf'):fig.savefig(figures/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    axes[0].bar(range(16),[r['fid'] for r in primary],color=[COLORS.get(r['arm'],'#aab0b5') for r in primary])
    axes[0].set_xticks(range(16),[LABELS[r['arm']] for r in primary],rotation=65,ha='right',fontsize=7)
    axes[0].set(ylabel='FID',title='Classical fits and all primary 20k neural references')
    selected=[next(r for r in primary if r['arm']==a) for a in ('ode_k1','fixed06','matched_silu64','swiglu86',*g.ARMS)]
    axes[1].bar(range(6),[r['FID_mean_term'] for r in selected],label='Feature-mean term',color='#6baed6')
    axes[1].bar(range(6),[r['FID_covariance_term'] for r in selected],bottom=[r['FID_mean_term'] for r in selected],label='Covariance term',color='#fdae6b')
    axes[1].set_xticks(range(6),[LABELS[r['arm']] for r in selected],rotation=35,ha='right',fontsize=8);axes[1].set(ylabel='FID accounting terms',title='Accounting, not a causal decomposition');axes[1].legend(fontsize=8)
    export(fig,'reference-comparison')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for arm in ('ode_k1','fixed06','matched_silu64','swiglu86','silu131',*g.ARMS):
        selected=[r for r in panels if r['arm']==arm]
        for ax,k in zip(axes,('weighted_mse','clean_mse')):ax.plot([r['time'] for r in selected],[r[k] for r in selected],'-o',label=LABELS[arm],color=COLORS[arm],markersize=3)
    for ax,title in zip(axes,('Weighted paired MSE','Clean paired MSE')):ax.set(xlabel='Original fixed panel time',ylabel=title)
    axes[0].legend(fontsize=7,ncol=2);export(fig,'time-panels')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axes[0].plot(np.arange(1,60),p['eigenvalues'],color=COLORS['ppca59']);axes[0].axhline(p['tail_variance'],color='#555555',linestyle='--',label='Average discarded variance')
    axes[0].set(xlabel='Retained mode index',ylabel='Training covariance eigenvalue',yscale='log',title='PPCA rank fixed by scalar budget');axes[0].legend(fontsize=7)
    axes[1].bar([0],[summary['retained_variance_fraction']],label='59 retained modes',color=COLORS['ppca59'])
    axes[1].bar([0],[summary['tail_variance_fraction']],bottom=[summary['retained_variance_fraction']],label='3013 residual modes',color='#aaaaaa')
    axes[1].set(xticks=[0],xticklabels=['Empirical covariance trace'],ylim=(0,1),ylabel='Fraction of training variance',title='Variance accounting does not establish generation quality');axes[1].legend(fontsize=7)
    export(fig,'fitted-spectrum')
    fig,axes=plt.subplots(16,8,figsize=(10,18),layout='constrained')
    by={(r['arm'],r['step']):r for r in formal}
    for i,r in enumerate(primary):
        arm=r['arm'];source=by[(arm,0 if arm in g.ARMS else 20000)];images=np.load(source['sample_path'])[:8]
        images=np.rint((np.clip(images,-1,1)+1)*127.5).clip(0,255).astype(np.uint8)
        for j in range(8):
            ax=axes[i,j];ax.imshow(images[j].transpose(1,2,0),interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            if j==0:ax.set_ylabel(LABELS[arm]+(' fit' if arm in g.ARMS else ' 20k'),rotation=0,ha='right',va='center',fontsize=7)
            if i==0:ax.set_title(f'Start {j}',fontsize=8)
    export(fig,'fixed-examples');g.verify(root);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);main(parser.parse_args().root.resolve())
