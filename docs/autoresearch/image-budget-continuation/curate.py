"""Curate the complete audited comparison; never select an earlier endpoint."""
from __future__ import annotations
import argparse,copy,csv,json
from pathlib import Path
import numpy as np

ARMS=('ode_k1','ode_k2','swiglu86','swiglu129','silu64','silu131','silu193')
CONTROLS={'ode_k1':('swiglu86','silu131'),'ode_k2':('swiglu129','silu193')}
LABELS={'ode_k1':'ODE K1','ode_k2':'ODE K2','swiglu86':'Swi86','swiglu129':'Swi129',
        'silu64':'SiLU64','silu131':'SiLU131','silu193':'SiLU193'}
PRIOR=Path('/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle20/curation/comparison.csv')
IMAGE_PARENT=PRIOR.parent.parent.parent/'oct28-cycle17-v2'
load=lambda p:json.loads(Path(p).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def table(path,rows):
    with Path(path).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def priority(rows):
    final=[r for r in rows if r['step']==20000]
    assert len(final)==7 and {r['arm'] for r in final}==set(ARMS)
    byarm={r['arm']:r for r in final};result=[]
    for arm,controls in CONTROLS.items():
        model=byarm[arm];comparison=[]
        for reference in controls:
            other=byarm[reference]
            ratios={k:model[k]/other[k] for k in ('fid','weighted_mse','clean_mse')}
            comparison.append(dict(reference=reference,**{k+'_ratio':v for k,v in ratios.items()},
                FID_margin_pass=ratios['fid']<=.95))
        fitting={key:model[key]/min(byarm[a][key] for a in controls) for key in ('weighted_mse','clean_mse')}
        allowed=all(r['FID_margin_pass'] for r in comparison) and all(v<=1.05 for v in fitting.values())
        result.append(dict(arm=arm,primary_step=20000,comparisons=comparison,
            ratios_to_best_fitting_control=fitting,follow_up_priority_margin_met=allowed,
            new_method_established=False,old_failed_gate_relabelled=False))
    return result


def check_rule():
    rows=[dict(arm=a,step=s,fid=100.,weighted_mse=1.,clean_mse=1.) for s in (10000,20000) for a in ARMS]
    def set_value(rows,arm,step,**values):next(r for r in rows if r['arm']==arm and r['step']==step).update(values)
    boundary=copy.deepcopy(rows);set_value(boundary,'ode_k1',20000,fid=95.,weighted_mse=1.05,clean_mse=1.05)
    assert priority(boundary)[0]['follow_up_priority_margin_met']
    weak=copy.deepcopy(rows);set_value(weak,'ode_k1',20000,fid=90.);set_value(weak,'silu131',20000,fid=90.)
    assert not priority(weak)[0]['follow_up_priority_margin_met']
    fit=copy.deepcopy(rows);set_value(fit,'ode_k2',20000,fid=90.,weighted_mse=1.051)
    assert not priority(fit)[1]['follow_up_priority_margin_met']
    early=copy.deepcopy(rows);set_value(early,'ode_k1',10000,fid=1.)
    assert not priority(early)[0]['follow_up_priority_margin_met']
    return dict(status='PASS',cases=['exact inclusive boundary','both ordinary comparators required',
                'fitting guard required','10k cannot replace20k'],new_experimental_result=False)


def main(root,out):
    check_rule();training=load(root/'audit-training/result.json');features=load(root/'audit-features/result.json')
    assert training['status']==features['status']=='PASS'
    assert len(training['rows'])==112 and len(training['sampling'])==14 and training['independent_stream_batches']==15000
    assert len(features['rows'])==14 and features['generated_features_bitwise']==14336 and features['reference_features_bitwise']==2048
    assert {(r['arm'],r['step']) for r in training['rows']}=={(a,s) for a in ARMS for s in range(5000,20001,1000)}
    assert {(r['arm'],r['step']) for r in features['rows']}=={(a,s) for a in ARMS for s in (10000,20000)}
    produced=load(root/'metric-output/result.json');assert produced['status']=='PASS' and len(produced['rows'])==14
    for row in produced['rows']:
        verified=next(r for r in features['rows'] if r['arm']==row['arm'] and r['step']==row['step'])
        assert abs(row['fid']-verified['fid'])<1e-8
        fit=next(r for r in training['rows'] if r['arm']==row['arm'] and r['step']==row['step'])
        for key in ('weighted_mse','clean_mse','prediction_rms'):np.testing.assert_allclose(row[key],fit[key],rtol=1e-8,atol=1e-10)
    prior=list(csv.DictReader(PRIOR.open()))
    prior={r['family']:r for r in prior if r['family'] in ARMS};assert len(prior)==7
    rows=[]
    reference_features=np.load(IMAGE_PARENT/'metric-inputs/reference-features.npy').astype(np.float64)
    for arm in ARMS:
        old=prior[arm];cost=dict(parameters=int(old['parameters']),FFN_forward_MACs=int(old['FFN_forward_MACs']))
        rows.append(dict(arm=arm,step=5000,weighted_mse=float(old['weighted_mse']),clean_mse=float(old['clean_mse']),fid=float(old['fid']),
            FID_mean_term=float(old['mean_term']),FID_covariance_term=float(old['covariance_term']),**cost,provenance='sealed5k endpoint'))
        for step in (10000,20000):
            row=next(r for r in produced['rows'] if r['arm']==arm and r['step']==step)
            feature=np.load(root/'metric-output'/f'{arm}-{step:07d}-features.npy').astype(np.float64)
            mean_term=float(np.sum((feature.mean(0)-reference_features.mean(0))**2))
            rows.append(dict(arm=arm,step=step,**{k:row[k] for k in ('weighted_mse','clean_mse','fid')},
                FID_mean_term=mean_term,FID_covariance_term=row['fid']-mean_term,**cost,provenance='new audited continuation'))
    out.mkdir(exist_ok=False);table(out/'comparison.csv',rows)
    curves=[]
    for arm in ARMS:
        for line in (root/'output/runs'/arm/'metrics.jsonl').read_text().splitlines():curves.append(dict(arm=arm,**json.loads(line)))
    table(out/'curves.csv',[{k:r.get(k,'') for k in ('arm','step','weighted_mse','clean_mse','prediction_rms','train_loss','grad_norm','clip_fraction')} for r in curves])
    panels=[]
    for step in (10000,20000):
        for time in (.1,.3,.5,.7,.9):
            with np.load(IMAGE_PARENT/'banks'/f'panel_{time:.1f}.npz') as bank:
                target=bank['target'].astype(np.float64);delta=bank['delta'].astype(np.float64)[:,None,None,None]
            for arm in ARMS:
                with np.load(root/'output/runs'/arm/f'panel_{step:07d}_{time:.1f}.npz') as bank:prediction=bank['prediction'].astype(np.float64)
                error=prediction-target
                panels.append(dict(arm=arm,step=step,time=time,examples=len(prediction),
                    weighted_mse=float(np.mean((error/delta)**2)),clean_mse=float(np.mean(error**2))))
    assert len(panels)==70;table(out/'time-panels.csv',panels)
    with np.load(IMAGE_PARENT/'banks/validation.npz') as bank:times=bank['t'].reshape(-1)
    write(out/'time-coverage.json',dict(examples=len(times),minimum=float(times.min()),maximum=float(times.max()),
        t_at_least_07=int((times>=.7).sum()),t_at_least_08=int((times>=.8).sum()),t_at_least_09=int((times>=.9).sum()),
        interpretation='Coverage of the fixed random-time validation bank only. Fixed panels are complementary; this does not causally explain FID ordering.'))
    Ps=[]
    for arm in ('ode_k1','ode_k2'):
        for step in (5000,10000,15000,20000):
            for p in load(root/'output/runs'/arm/f'P_{step:07d}.json'):
                sv=np.asarray(p['singular_values']);Ps.append(dict(arm=arm,step=step,layer=p['layer'],sigma_max=float(sv[0]),
                    singular_CV=float(sv.std()/sv.mean()),frobenius=p['frobenius'],max_real_eigenvalue=max(p['eigenvalues_real'])))
    assert len(Ps)==48;table(out/'P-history.csv',Ps)
    decision=dict(primary_step=20000,context_step=10000,criteria=priority(produced['rows']),
        new_method_established=False,confirmation_required_for_any_lead=True,
        interpretation='Predeclared practical allocation margins only; no significance or integration-order claim.')
    write(out/'decision.json',decision);write(out/'rule-validation.json',check_rule())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(ARMS,('#2378a8','#d65b36','#4f6d7a','#70a9a1','#b1a7a6','#9a8c98','#6b705c')))
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for ax,key,title in zip(axes,('weighted_mse','fid'),('Weighted validation MSE','Screening FID, Heun128')):
        vals=[next(r[key] for r in rows if r['arm']==a and r['step']==20000) for a in ARMS]
        ax.bar(range(7),vals,color=[colors[a] for a in ARMS]);ax.set(xticks=range(7),xticklabels=[LABELS[a] for a in ARMS],title=title)
        ax.tick_params(axis='x',rotation=35);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Fixed 20,000-update endpoint; all seven models');fig.savefig(out/'final-comparison.png',dpi=180);fig.savefig(out/'final-comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for arm in ARMS:
        selected=[r for r in curves if r['arm']==arm]
        for ax,key in zip(axes,('weighted_mse','clean_mse')):ax.plot([r['step']/1000 for r in selected],[r[key] for r in selected],color=colors[arm],label=LABELS[arm])
    for ax,title in zip(axes,('Weighted validation MSE','Clean-image MSE')):
        ax.set(xlabel='Total updates (thousands)',title=title);ax.spines[['top','right']].set_visible(False)
    axes[1].legend(frameon=False,fontsize=8);fig.savefig(out/'continuation-curves.png',dpi=180);fig.savefig(out/'continuation-curves.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
    for ax,step in zip(axes,(10000,20000)):
        for arm in ARMS:
            selected=[r for r in panels if r['step']==step and r['arm']==arm]
            ax.plot([r['time'] for r in selected],[r['weighted_mse'] for r in selected],color=colors[arm],marker='.',label=LABELS[arm])
        ax.set(xlabel='External FM time',ylabel='Paired weighted MSE',title=f'Fixed 64-image panels, {step:,} updates',yscale='log')
        ax.spines[['top','right']].set_visible(False)
    axes[1].legend(frameon=False,fontsize=8);fig.savefig(out/'time-panels.png',dpi=180);fig.savefig(out/'time-panels.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),sharey=True,layout='constrained')
    for ax,arm in zip(axes,('ode_k1','ode_k2')):
        for layer in sorted({r['layer'] for r in Ps if r['arm']==arm}):
            selected=[r for r in Ps if r['arm']==arm and r['layer']==layer]
            ax.plot([r['step']/1000 for r in selected],[r['sigma_max'] for r in selected],marker='.',label=layer.replace('blocks.',''))
        ax.set(title=LABELS[arm],xlabel='Total updates (thousands)',ylabel='Largest singular value of hidden P');ax.legend(frameon=False,fontsize=7)
    fig.savefig(out/'P-trajectories.png',dpi=180);fig.savefig(out/'P-trajectories.pdf');plt.close(fig)
    for step in (10000,20000):
        fig,axes=plt.subplots(7,8,figsize=(8,7),layout='constrained')
        for j,arm in enumerate(ARMS):
            samples=np.load(root/'output/runs'/arm/f'generated-{step:07d}.npy',mmap_mode='r')
            for i,ax in enumerate(axes[j]):
                ax.imshow(np.clip((samples[i].transpose(1,2,0)+1)/2,0,1));ax.set_xticks([]);ax.set_yticks([])
                if i==0:ax.set_ylabel(LABELS[arm],fontsize=8)
        fig.suptitle(f'First eight shared starts; Heun128 at {step:,} updates')
        fig.savefig(out/f'generation-{step}.png',dpi=180);plt.close(fig)
    write(out/'curation.json',dict(status='PASS',comparison_rows=len(rows),continuation_points=len(curves),P_records=len(Ps),
        figures=6,time_panel_rows=70,case_indices=list(range(8)),primary_step=20000,all_figures_require_visual_review=True,analysis_report_pending=True))
    print(json.dumps(load(out/'curation.json')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--output',type=Path);p.add_argument('--check-rule',action='store_true');a=p.parse_args()
    if a.check_rule:print(json.dumps(check_rule()))
    else:main(a.root,a.output)
