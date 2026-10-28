"""Complete cold-control summaries, fixed-budget comparisons and matrix histories."""
import argparse,csv,json,shutil,sys
from pathlib import Path
import numpy as np

BASE_ORDER=('ode_k1','fixed06','matched_silu64','native_posthoc06','ode_k1_time','ode_k1_pooled','ode_k1_constant',
    'ode_k2','ode_k2_constant','swiglu86','swiglu129','silu64','silu131','silu193')
LABELS=dict(zip(BASE_ORDER,('K1 native','Fixed 0.6 P','Matched-init SiLU64','Native posthoc 0.6','Old K1 time','Old K1 pooled','Old K1 constant',
    'K2 native','Old K2 constant','Swi86','Swi129','Default SiLU64','SiLU131','SiLU193')))
COLORS={'ode_k1':'#2378a8','fixed06':'#d65b36','matched_silu64':'#3a923a','swiglu86':'#8172b2','silu64':'#937860','silu131':'#da8bc3'}
def table(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def jsonlines(path):return [json.loads(v) for v in path.read_text().splitlines()]
def verify_external_data(study,root):
    """Hash the actual external pixels/split as well as the parent's manifest."""
    native=study.PARENTS['native'];data=Path(study.load(native/'settings.json')['data'])
    bound=study.load(native/'bindings.json');started=study.load(root/'orchestrator/identity.json')['create_time'];rows=[]
    for name in ('images.uint8.npy','split.npz','manifest.json'):
        p=data/name;key=str(p.resolve());digest=study.sha(p)
        assert bound[key]==digest,(name,'external input changed')
        assert p.stat().st_ctime<started,(name,'external input metadata changed after launch')
        rows.append(dict(path=key,sha256=digest,bytes=p.stat().st_size,ctime_precedes_controller=True))
    study.old().verify(native,True)
    return dict(status='PASS',files=rows,native_external_and_metric_bindings_verified=True)
def sample_path(study,root,arm,step):
    if arm in study.ARMS or arm=='native_posthoc06':return root/'sampling'/f'{arm}-{step:07d}.npy'
    if arm in ('ode_k1_time','ode_k1_pooled','ode_k1_constant','ode_k2_constant'):return study.imported('timeclocks').sample_bank(study.PARENTS['timeclocks'],arm)
    if step==5000:
        return study.PARENTS['ordinary']/'output/runs'/arm/'generated.npy' if arm.startswith('silu') else study.PARENTS['resolution']/'output/runs'/arm/'generated-128.npy'
    return study.PARENTS['budget']/'output/runs'/arm/f'generated-{step:07d}.npy'
def main(root):
    sys.path.insert(0,str(root/'source'));import training_controls as study
    study.verify(root);data_check=verify_external_data(study,root)
    assert study.load(root/'orchestrator/status.json')['state']=='COMPLETE'
    assert study.load(root/'controller-tool-exit.json')['exit_code']==0
    phases=study.load(root/'orchestrator/launches.json');assert len(phases)==6 and all(v['exit_code']==0 for v in phases)
    for phase in ('training','sampling','features'):assert study.load(root/f'audit-{phase}/result.json')['status']=='PASS'
    formal=study.load(root/'metric-output/result.json')['rows'];assert len(formal)==32
    decision=study.load(root/'metric-output/decision.json');assert decision==study.decisions(formal)
    out=root/'curation';out.mkdir();figures=out/'figures';figures.mkdir()
    rows=[]
    for r in sorted(formal,key=lambda v:(v['step'],BASE_ORDER.index(v['arm']))):
        arm=r['arm'];frozen=arm in ('native_posthoc06','ode_k1_time','ode_k1_pooled','ode_k1_constant','ode_k2_constant')
        active=185744 if frozen or arm in study.ARMS else r['parameters']
        training_active=211094 if frozen else active
        raw=np.load(sample_path(study,root,arm,r['step']));assert raw.shape==(1024,3,32,32) and np.isfinite(raw).all()
        rows.append(dict(arm=arm,step=r['step'],weighted_mse=r['weighted_mse'],clean_mse=r['clean_mse'],fid=r['fid'],
            FID_mean_term=r['FID_mean_term'],FID_covariance_term=r['FID_covariance_term'],raw_tail_fraction=float(np.mean(np.abs(raw)>1)),
            stored_learned_parameters=r['parameters'],active_learned_parameters=active,training_active_learned_parameters=training_active,
            FFN_forward_MACs=r['FFN_forward_MACs'],generated=1024,reference=2048,Heun_steps=128,provenance=r['provenance']))
    table(out/'comparison.csv',rows);study.write(out/'decision.json',decision);primary=[r for r in rows if r['step']==20000];assert len(primary)==14
    table(out/'primary-20k.csv',primary)
    ratios=[]
    for arm in study.ARMS:
        candidate=next(r for r in primary if r['arm']==arm)
        for reference in primary:
            if reference['arm']==arm:continue
            ratios.append(dict(arm=arm,reference=reference['arm'],step=20000,
                **{key+'_ratio':candidate[key]/reference[key] for key in ('fid','weighted_mse','clean_mse')},
                FID_difference=candidate['fid']-reference['fid']))
    table(out/'primary-ratios.csv',ratios)
    panels=[]
    for arm in (*study.ARMS,'native_posthoc06'):
        for step in study.STEPS if arm in study.ARMS else (20000,):
            for t in (.1,.3,.5,.7,.9):
                path=root/'output/runs'/arm/f'panel_{step:07d}_{t:.1f}.npz' if arm in study.ARMS else root/'sampling'/f'posthoc-panel-{t:.1f}.npz'
                with np.load(path) as f:prediction=f['prediction']
                b=study.bank(t);error=prediction.astype(np.float64)-b['target'].astype(np.float64)
                panels.append(dict(arm=arm,step=step,time=t,weighted_mse=float(np.mean((error/b['delta'].astype(np.float64)[:,None,None,None])**2)),clean_mse=float(np.mean(error**2)),examples=64))
    assert len(panels)==75;table(out/'time-panels.csv',panels)
    curves=[]
    for arm in (*study.ARMS,'ode_k1','swiglu86','silu64','silu131'):
        if arm in study.ARMS:source=jsonlines(root/'output/runs'/arm/'metrics.jsonl')
        else:
            parent=study.PARENTS['ordinary'] if arm.startswith('silu') else study.PARENTS['native']
            source=jsonlines(parent/'output/runs'/arm/'metrics.jsonl')+jsonlines(study.PARENTS['budget']/'output/runs'/arm/'metrics.jsonl')
        by={r['step']:r for r in source}
        for step,r in sorted(by.items()):curves.append(dict(arm=arm,step=step,weighted_mse=r['weighted_mse'],clean_mse=r['clean_mse'],
            train_loss=r.get('train_loss'),grad_norm=r.get('grad_norm'),clip_fraction=r.get('clip_fraction')))
    table(out/'training-curves.csv',curves)
    matrices=[]
    for arm in study.ARMS:
        for step in study.STEPS:
            for r in study.load(root/'output/runs'/arm/f'matrix-history-{step:07d}.json'):
                matrices.append(dict(arm=arm,step=step,layer=r['layer'],coordinate=r['coordinate'],sigma_max=r['sigma_max'],sigma_min=r['sigma_min'],
                    singular_CV=r['singular_CV'],frobenius=r['frobenius'],eigen_max_real=max(r['eigen_real']),eigen_max_abs=float(np.max(np.hypot(r['eigen_real'],r['eigen_imag'])))))
    for step in study.STEPS:
        state=study.native_state(step)
        for j in range(3):
            for k in (1,2):
                a=state['model'][f'blocks.{j}.mlp.ode{k}.proj.weight'].numpy().astype(np.float64);singular=np.linalg.svd(a,compute_uv=False);eigen=np.linalg.eigvals(a)
                matrices.append(dict(arm='ode_k1',step=step,layer=f'block{j}.P{k}',coordinate='trained_P',sigma_max=float(singular[0]),sigma_min=float(singular[-1]),
                    singular_CV=float(singular.std()/singular.mean()),frobenius=float(np.linalg.norm(a)),eigen_max_real=float(eigen.real.max()),eigen_max_abs=float(np.abs(eigen).max())))
    assert len(matrices)==210;table(out/'matrix-history.csv',matrices)
    cost=study.load(root/'preflight.json')['costs'];study.write(out/'cost-witness.json',dict(rows=cost))
    resources=jsonlines(root/'output/resources.jsonl');peaks={arm:max(v['RSS'] for r in resources for v in r['resources'] if v['arm']==arm) for arm in study.ARMS}
    table(out/'resource-summary.csv',[dict(arm=arm,training_seconds=study.load(root/'output/runs'/arm/'result.json')['seconds'],peak_CPU_RSS_bytes=peaks[arm]) for arm in study.ARMS])
    for src,target in [('preflight.json','preflight.json'),('metric-preflight.json','metric-preflight.json'),('documented-pass.json','agent-doc-pass.json'),
        ('audit-training/result.json','training-boundary-audit.json'),('audit-sampling/result.json','sampling-audit.json'),('audit-features/result.json','feature-audit.json')]:shutil.copyfile(root/src,out/target)
    study.write(out/'external-data-verification.json',data_check)
    eligibility={r['arm']:r['allocation_margin'] and r['SiLU64_strict_ordering'] for r in decision['rows']}
    with (root/'results.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(('commit','metric','memory_gb','status','description'))
        for r in rows:
            arm=r['arm'];new=arm in study.ARMS
            w.writerow((study.load(root/'settings.json')['source_commit'][:7] if new or arm=='native_posthoc06' else 'prior',format(r['fid'],'.12f'),
                format(peaks[arm]/2**30,'.3f') if new else 'NA','keep' if not new or eligibility[arm] else 'discard',
                f"{arm} step{r['step']}: "+('cold ordinary-control family; primary20k decision, no ODE-specific method' if new else 'reference evidence only; not nominated')))
    shutil.copyfile(root/'results.tsv',out/'results.tsv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def export(fig,name):
        for ext in ('png','pdf'):fig.savefig(figures/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    for arm in ('ode_k1','fixed06','matched_silu64','ode_k2','swiglu86','swiglu129','silu64','silu131','silu193'):
        selected=[r for r in rows if r['arm']==arm];axes[0].plot([r['step'] for r in selected],[r['fid'] for r in selected],'-o',label=LABELS[arm],color=COLORS.get(arm),markersize=4)
    axes[0].set(xlabel='Fixed training updates',ylabel='FID',title='All cold native/control budgets');axes[0].legend(fontsize=7,ncol=2)
    axes[1].bar(range(14),[r['fid'] for r in primary],color=[COLORS.get(r['arm'],'#a0a7ad') for r in primary])
    axes[1].set_xticks(range(14),[LABELS[r['arm']] for r in primary],rotation=60,ha='right',fontsize=7);axes[1].set(ylabel='FID',title='Primary20k: every control/context')
    export(fig,'FID-budgets')
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for arm in (*study.ARMS,'ode_k1','swiglu86','silu64','silu131'):
        selected=[r for r in curves if r['arm']==arm]
        for ax,key in zip(axes[:2],('weighted_mse','clean_mse')):ax.plot([r['step'] for r in selected],[r[key] for r in selected],label=LABELS[arm],color=COLORS[arm])
        if arm in study.ARMS:
            selected=[r for r in selected if r['clip_fraction'] is not None];axes[2].plot([r['step'] for r in selected],[r['clip_fraction'] for r in selected],'-o',label=LABELS[arm],color=COLORS[arm])
    for ax,key in zip(axes,('Weighted validation MSE','Clean validation MSE','Clip fraction over preceding interval')):ax.set(xlabel='Training updates',ylabel=key)
    axes[0].legend(fontsize=7);axes[2].legend(fontsize=7);export(fig,'training-curves')
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for arm in ('ode_k1',*study.ARMS):
        for ax,key,coordinate in zip(axes,('sigma_max','singular_CV','sigma_max'),('P','P','effective_W')):
            if arm=='ode_k1' and coordinate=='effective_W':continue
            selected=[r for r in matrices if r['arm']==arm and ((r['coordinate']=='effective_W') if coordinate=='effective_W' else r['coordinate']!='effective_W')]
            x=list(study.STEPS);y=[np.mean([r[key] for r in selected if r['step']==step]) for step in x]
            label=LABELS[arm]+(' (inferred P)' if arm=='matched_silu64' and coordinate=='P' else '')
            ax.plot(x,y,'-o',label=label,color=COLORS[arm],markersize=3)
    for ax,title in zip(axes,('Mean largest singular value: P coordinates','Mean singular-value CV: P coordinates','Effective W: mean largest singular value')):ax.set(xlabel='Training updates',title=title);ax.legend(fontsize=7)
    fig.suptitle('Six-layer summaries; coordinate histories are not complete Jacobian bounds');export(fig,'matrix-histories')
    def cases(records,name,height):
        fig,axes=plt.subplots(len(records),8,figsize=(10,height),layout='constrained',squeeze=False)
        for i,(arm,step) in enumerate(records):
            images=np.load(sample_path(study,root,arm,step))[:8];images=np.rint((np.clip(images,-1,1)+1)*127.5).clip(0,255).astype(np.uint8)
            for j in range(8):
                ax=axes[i,j];ax.imshow(images[j].transpose(1,2,0),interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
                if j==0:ax.set_ylabel(f'{LABELS[arm]} {step//1000}k',rotation=0,ha='right',va='center',fontsize=7)
                if i==0:ax.set_title(f'Start {j}',fontsize=8)
        export(fig,name)
    cases([(r['arm'],20000) for r in primary],'primary-examples',16)
    cases([(arm,step) for step in (5000,10000) for arm in study.ARMS],'earlier-control-examples',5)
    study.write(out/'summary.json',dict(status='PASS',endpoints=32,primary20k=14,new_training_trajectories=2,new_sample_banks=7,
        matrix_summary_rows=210,time_panel_scores=75,primary_ratios=26,figures=5,unique_fixed_examples=144,full_optimization_trajectory_replayed=False,new_ODE_method=False))
    study.verify(root);assert verify_external_data(study,root)==data_check
    print(json.dumps(study.load(out/'summary.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
