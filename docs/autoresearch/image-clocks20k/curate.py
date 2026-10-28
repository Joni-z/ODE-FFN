"""Curate all nine endpoints only after the four monitored phases pass."""
from pathlib import Path
import argparse,csv,json,shutil,sys
import numpy as np

ORDER=('ode_k1','ode_k1_constant','ode_k2','ode_k2_constant','swiglu86','swiglu129','silu64','silu131','silu193')
LABELS=dict(zip(ORDER,('K1 native','K1 constant','K2 native','K2 constant','Swi86','Swi129','SiLU64','SiLU131','SiLU193')))
def table(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def main(root):
    sys.path.insert(0,str(root/'source'));import study
    study.verify(root,True);assert study.load(root/'orchestrator/status.json')['state']=='COMPLETE'
    phases=study.load(root/'orchestrator/launches.json');assert len(phases)==4 and all(r['exit_code']==0 for r in phases)
    sampling=study.load(root/'audit-sampling/result.json');features=study.load(root/'audit-features/result.json')
    assert sampling['status']==features['status']=='PASS'
    assert sampling['new_generated_images_bitwise']==features['new_generated_features_bitwise']==2048
    assert features['reference_features_bitwise']==2048 and len(features['rows'])==9
    formal=study.load(root/'metric-output/result.json')['rows'];assert {r['family'] for r in formal}==set(ORDER)
    rows=[next(r for r in formal if r['family']==name) for name in ORDER]
    for row in rows:
        checked=next(r for r in features['rows'] if r['family']==row['family'])
        assert abs(row['fid']-checked['fid'])<1e-8
    decision=study.load(root/'metric-output/decision.json')
    assert decision['criteria']==study.decisions(rows,study.load(study.PARENTS['budget']/'curation/decision.json'))
    out=root/'curation';out.mkdir(exist_ok=False);figures=out/'figures';figures.mkdir()
    keys=('family','step','weighted_mse','clean_mse','fid','mean_term','covariance_term','out_of_range_fraction',
          'parameters','FFN_forward_MACs','generated','reference','Heun_steps','provenance')
    table(out/'comparison.csv',[{k:r[k] for k in keys} for r in rows])
    table(out/'calibration.csv',[dict(family=arm,layer=name,**value) for arm,clock in study.load(root/'calibration.json').items() for name,value in clock.items()])
    contrasts=[]
    for r in decision['criteria']:
        contrasts.append(dict(family=r['arm'],strict_FID_order_retained=r['strict_FID_order_retained'],
            allocation_margin_retained=r['allocation_margin_retained'],ordinary_gap_retained_fraction=r['ordinary_gap_retained_fraction'],
            native_allocation_margin_met=r['native_allocation_margin_met'],
            **{'delta_'+key:value for key,value in r['native_to_constant'].items()}))
    table(out/'contrasts.csv',contrasts);study.write(out/'decision.json',decision)
    panels=[]
    for t in (.1,.3,.5,.7,.9):
        with np.load(study.PARENTS['native']/'banks'/f'panel_{t:.1f}.npz') as bank:
            target=bank['target'].astype(np.float64);delta=bank['delta'].astype(np.float64)[:,None,None,None]
        for family in ORDER:
            path=root/'output/runs'/family.removesuffix('_constant')/f'panel-{t:.1f}.npz' if family.endswith('_constant') else study.PARENTS['budget']/'output/runs'/family/f'panel_0020000_{t:.1f}.npz'
            with np.load(path) as b:prediction=b['prediction'].astype(np.float64)
            error=prediction-target;panels.append(dict(family=family,time=t,examples=len(prediction),weighted_mse=float(np.mean((error/delta)**2)),clean_mse=float(np.mean(error**2))))
    assert len(panels)==45;table(out/'time-panels.csv',panels)
    resources=[json.loads(line) for line in (root/'output/resources.jsonl').read_text().splitlines()]
    peak={arm:max(r['RSS'] for row in resources for r in row['resources'] if r['arm']==arm)/2**30 for arm in study.ARMS}
    commit=study.load(root/'settings.json')['source_commit'][:7]
    with (root/'results.tsv').open('w') as f:
        w=csv.writer(f,delimiter='\t');w.writerow(['commit','metric','memory_gb','status','description'])
        for r in rows:
            arm=r['family'].removesuffix('_constant');new=r['family'].endswith('_constant')
            w.writerow([commit if new else 'prior',format(r['fid'],'.12f'),format(peak[arm],'.3f') if new else 'NA','keep',r['family']+': retained frozen evidence; no new method acceptance'])
    shutil.copyfile(root/'results.tsv',out/'results.tsv')
    for source,target in [('preflight.json','preflight.json'),('documented-pass.json','agent-doc-pass.json'),('audit-sampling/result.json','sampling-audit.json'),('audit-features/result.json','feature-audit.json')]:shutil.copyfile(root/source,out/target)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors=('#2378a8','#85b7de','#d65b36','#ffba75','#4f6d7a','#70a9a1','#b1a7a6','#9a8c98','#6b705c')
    def export(fig,name):
        for ext in ('png','pdf'):fig.savefig(figures/f'{name}.{ext}',dpi=180)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,key,title in zip(axes,('weighted_mse','fid'),('Paired weighted validation MSE','Screening FID, Heun128 (1,024 / 2,048)')):
        for j,r in enumerate(rows):
            ax.scatter(j,r[key],color=colors[j]);ax.annotate(format(r[key],'.4f' if key=='weighted_mse' else '.2f'),(j,r[key]),xytext=(0,6),textcoords='offset points',ha='center',fontsize=7)
        ax.set(xticks=range(9),xticklabels=[LABELS[x].replace(' ','\n') for x in ORDER],title=title);ax.margins(y=.2)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Fixed clocks at native 20k weights; all learned tensors unchanged');export(fig,'fixed-comparison')
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=True,layout='constrained')
    for ax,arm in zip(axes,study.ARMS):
        for family in (arm,arm+'_constant',*study.CONTROLS[arm]):
            values=[r for r in panels if r['family']==family]
            ax.plot([r['time'] for r in values],[r['weighted_mse'] for r in values],marker='.',label=LABELS[family])
        ax.set(title=LABELS[arm],xlabel='External FM time',ylabel='Paired weighted MSE',yscale='log');ax.legend(frameon=False,fontsize=8)
    export(fig,'time-panels')
    fig,axes=plt.subplots(9,8,figsize=(8,9),layout='constrained')
    for j,family in enumerate(ORDER):
        raw=np.load(study.sample_bank(root,family),mmap_mode='r')[:8]
        images=np.rint((np.clip(raw,-1,1)+1)*np.float32(127.5)).astype(np.uint8).transpose(0,2,3,1)
        for k,ax in enumerate(axes[j]):
            ax.imshow(images[k],interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            if k==0:ax.set_ylabel(LABELS[family],fontsize=8)
    fig.suptitle('Same first eight Gaussian starts; 20k checkpoints, Heun128');export(fig,'generation-cases')
    study.write(out/'curation.json',dict(status='PASS',comparison_rows=9,calibration_scalars=12,time_panels=45,figures=3,
        case_indices=list(range(8)),visual_review_pending=True,analysis_report_pending=True,new_method_established=False))
    print(json.dumps(study.load(out/'curation.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
