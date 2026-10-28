"""Descriptive fixed-protocol analysis; requires a consumed successful tool exit."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LAWS=('rotated_product_bimodal_16d','sign_code','half_mixture')
ARMS=('rotation','euler','normalized_euler','silu_identity','fixed06_identity','swiglu129','native_k1')
HISTORY=('ode_k2','free_isru','swiglu64','swiglu128','swiglu160')
METRICS=('oracle_mse','jacobian_error','sw2')
STEPS=(0,1000,2500,5000)
load=lambda p:json.loads(Path(p).read_text())
def write(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def table(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def curate(root,out):
    tool=load(root/'run-tool.json');assert tool['status']=='TERMINAL' and tool['exit_code']==0 and tool['actual_terminal_exit_consumed']
    assert load(root/'orchestrator/status.json')['state']=='COMPLETE'
    assert all(v['exit_code']==0 for v in load(root/'orchestrator/launches.json'))
    audit=load(root/'audit/result.json');assert audit['status']=='PASS'
    out.mkdir(exist_ok=False);fig=out/'figures';fig.mkdir()
    costs={v['arm']:v for v in load(root/'preflight.json')['cells']}
    rows=[];dynamics=[];clocks=[];training=[];accept=[]
    for law in LAWS:
        for arm in (*ARMS,*HISTORY):
            reused=arm in HISTORY
            row=next(v for v in audit['historical_endpoints' if reused else 'new_endpoints'] if v['law']==law and v['arm']==arm)
            cost=row if reused else costs[arm]
            rows.append(dict(law=law,arm=arm,reused_endpoint=reused,**{m:row[m] for m in METRICS},**{k:cost[k] for k in ('parameters','dense_forward_macs','dense_training_forward_backward_macs')}))
        group=[v for v in rows if v['law']==law];candidate=next(v for v in group if v['arm']=='rotation');normalized=next(v for v in group if v['arm']=='normalized_euler')
        references=[v for v in group if v['arm'] not in ('rotation','normalized_euler')]
        accept.append(dict(law=law,**{m+'_ratio_to_best_practical':candidate[m]/min(v[m] for v in references) for m in METRICS},
            **{m+'_best_practical_arm':min(references,key=lambda v:v[m])['arm'] for m in METRICS},
            **{m+'_ratio_to_normalized_euler':candidate[m]/normalized[m] for m in METRICS}))
        for arm in ARMS:
            folder=root/'output/runs'/law/arm
            for line in (folder/'metrics.jsonl').read_text().splitlines():
                v=json.loads(line)
                if 'oracle_mse' in v:training.append(dict(law=law,arm=arm,step=v['step'],oracle_mse=v['oracle_mse']))
            for step in STEPS:
                v=load(folder/f'generator_{step:07d}.json')
                dynamics.extend(dict(law=law,arm=arm,step=step,**p) for p in v['matrices'])
                clocks.extend(dict(law=law,arm=arm,step=step,**{k:p[k] for k in ('layer','clock_mean','clock_std','clock_across_example_std','rate_max_absolute','action_max_absolute_change')},
                    complete_gain_mean=float(np.mean(p['promoted_complete_map_gains'])),complete_gain_max=max(p['promoted_complete_map_gains'])) for p in v['flow_inputs'])
    candidate=costs['rotation'];swi=costs['swiglu129'];budget={k:candidate[k]/swi[k] for k in ('parameters','dense_forward_macs','dense_training_forward_backward_macs')}
    criteria=dict(at_least_one_10_percent_MSE_gain=any(v['oracle_mse_ratio_to_best_practical']<=.9 for v in accept),
        all_laws_MSE_within_5_percent=all(v['oracle_mse_ratio_to_best_practical']<=1.05 for v in accept),
        all_laws_Jacobian_and_SW2_within_10_percent=all(v[m+'_ratio_to_best_practical']<=1.1 for v in accept for m in METRICS[1:]),
        normalized_control_limits=all(v[m+'_ratio_to_normalized_euler']<=(1.05 if m=='oracle_mse' else 1.1) for v in accept for m in METRICS),
        Swi129_budget_within_5_percent=all(v<=1.05 for v in budget.values()))
    passed=all(criteria.values());write(out/'summary.json',dict(status='ANALYZED',registered_screen_pass=passed,criteria=criteria,per_law=accept,budget_ratios_to_Swi129=budget,
        new_trajectories=21,reused_endpoints=15,source_commit=load(root/'settings.json')['source_commit'],actual_tool=tool))
    for name,data in (('endpoints',rows),('registered-criteria',accept),('P-trajectories',dynamics),('visited-clocks',clocks),('field-curves',training)):table(out/(name+'.csv'),data)
    colors={arm:plt.get_cmap('tab10')(i) for i,arm in enumerate(ARMS)}
    figure,axes=plt.subplots(1,3,figsize=(16,4.2))
    for ax,m in zip(axes,METRICS):
        for i,arm in enumerate((*ARMS,*HISTORY)):
            values=[]
            for law in LAWS:
                group=[v for v in rows if v['law']==law];den=min(v[m] for v in group if v['arm'] not in ('rotation','normalized_euler'))
                values.append(next(v[m] for v in group if v['arm']==arm)/den)
            ax.plot(range(3),values,'o-',label=arm,color=colors.get(arm),alpha=1 if arm=='rotation' else .65,linewidth=2 if arm=='rotation' else 1)
        ax.axhline(1,color='black',linestyle=':',linewidth=.7);ax.set_xticks(range(3),('Product16','Sign-code','Half-mixture'));ax.set_title(m+' / best practical');ax.set_yscale('log');ax.grid(alpha=.15)
    axes[-1].legend(fontsize=7,bbox_to_anchor=(1.03,1),loc='upper left');figure.tight_layout();figure.savefig(fig/'01-comparison.png',dpi=180);figure.savefig(fig/'01-comparison.pdf');plt.close(figure)
    figure,axes=plt.subplots(1,3,figsize=(13,3.6))
    for ax,law in zip(axes,LAWS):
        for arm in ARMS:
            v=[p for p in training if p['law']==law and p['arm']==arm];ax.plot([p['step'] for p in v],[p['oracle_mse'] for p in v],'.-',label=arm,color=colors[arm])
        ax.set_title(law);ax.set_yscale('log');ax.set_xlabel('Update');ax.set_ylabel('Oracle field MSE');ax.grid(alpha=.15)
    axes[-1].legend(fontsize=7,bbox_to_anchor=(1.03,1),loc='upper left');figure.tight_layout();figure.savefig(fig/'02-field-curves.png',dpi=180);figure.savefig(fig/'02-field-curves.pdf');plt.close(figure)
    figure,axes=plt.subplots(2,3,figsize=(13,6.5))
    for col,law in enumerate(LAWS):
        for arm in ARMS[:3]:
            sigma=[];gain=[];lo=[];hi=[]
            for step in STEPS:
                values=[v['sigma_max'] for v in dynamics if v['law']==law and v['arm']==arm and v['step']==step];sigma.append(np.mean(values));lo.append(min(values));hi.append(max(values))
                gain.append(np.mean([v['complete_gain_mean'] for v in clocks if v['law']==law and v['arm']==arm and v['step']==step]))
            axes[0,col].plot(STEPS,sigma,'.-',label=arm,color=colors[arm]);axes[0,col].fill_between(STEPS,lo,hi,color=colors[arm],alpha=.1)
            axes[1,col].plot(STEPS,gain,'.-',label=arm,color=colors[arm])
        axes[0,col].set_title(law);axes[0,col].set_ylabel('P largest singular value');axes[1,col].set_ylabel('Mean complete local gain');axes[1,col].set_xlabel('Update')
        for ax in axes[:,col]:ax.grid(alpha=.15)
    axes[0,-1].legend(fontsize=7);figure.tight_layout();figure.savefig(fig/'03-P-and-feedback.png',dpi=180);figure.savefig(fig/'03-P-and-feedback.pdf');plt.close(figure)
    lines=['# Invariant-clock fixed screen','','The registered conjunction '+('passes.' if passed else 'fails.')+' All comparisons below use the fixed 5,000-update endpoint.','',
        '| Law | Rotation MSE | Swi129 MSE | Best practical MSE | Rotation / best | Rotation SW2 | Swi129 SW2 |','|---|---:|---:|---:|---:|---:|---:|']
    for law,a in zip(LAWS,accept):
        group=[v for v in rows if v['law']==law];r=next(v for v in group if v['arm']=='rotation');w=next(v for v in group if v['arm']=='swiglu129');best=min(v['oracle_mse'] for v in group if v['arm'] not in ('rotation','normalized_euler'))
        lines.append(f'| {law} | {r["oracle_mse"]:.7g} | {w["oracle_mse"]:.7g} | {best:.7g} ({a["oracle_mse_best_practical_arm"]}) | {r["oracle_mse"]/best:.4f} | {r["sw2"]:.7g} | {w["sw2"]:.7g} |')
    lines+=['','All 36 endpoints and all criteria are in the CSV/JSON files; the table does not select checkpoints or omit old strong controls. The operator is an exact amplitude-preserving hidden flow, but that property does not imply a contractive complete derivative or improved FM quality.','',
        '## Claim candidates','','- Claim: the tested invariant-clock constraint '+('meets' if passed else 'does not meet')+' the registered old-law quality/cost conjunction.',
        '  - Source evidence: summary.json, all 36 audited endpoints, fixed protocol.md.',
        '  - Allowed wording: this particular parameterization and fixed budget '+('justify separately registered task/image confirmation.' if passed else 'do not justify task/image promotion.'),
        '  - Forbidden stronger wording: all ODE architectures succeed/fail, exact flows inherently improve generation, or feature conservation guarantees full-network stability.',
        '  - Uncertainty: these designed synthetic laws are distinct tasks, not independent repeated trials from a task population.',
        '  - Next check: '+('separately register confirmation without changing this screen.' if passed else 'stop this setting; use existing failure evidence before choosing another design.'),
        '  - Decision: '+('keep' if passed else 'discard'),'',
        '## Verification','','Actual controller and both phase exits are zero. The automatic independent audit replays all checkpoint fields, native Jacobian panels and sampled outputs, checks alternative explicit-chain derivatives and raw matrices, and reconstructs the training-stream boundaries. It does not rerun complete optimizer trajectories. No held-out test bank was opened.']
    (out/'analysis-report.md').write_text('\n'.join(lines)+'\n')
    (out/'stats-appendix.md').write_text('# Statistical scope\n\nUnits are one trained model on one specified law. Report endpoint values and ratios; do not infer population-level significance from three deliberately constructed laws, checkpoints, model layers or evaluation points. No p-values, confidence intervals, or task averages are warranted by this fixed comparison. Shaded P ranges describe the six layers, not sampling uncertainty. Lower is better for every quality metric.\n')
    (out/'figure-catalog.md').write_text('# Figure interpretation\n\n1. Comparison: all 12 methods on all three laws, separately normalized by the best registered practical control for each metric. Inspect the fixed candidate across every axis; a favorable point alone cannot satisfy the conjunction. Historical rows are reused, with no claim of new training.\n2. Field curves: all seven fresh arms at the four fixed checkpoints. Inspect convergence at the fixed endpoint; no favorable-checkpoint selection.\n3. P and feedback: six-layer mean/layer range of the largest singular value, and mean complete local derivative gain on fixed actual-input anchors. The gain includes state-dependent clock feedback. A small P does not certify a small full derivative. All exact layer/clock summaries are in the CSVs; no confidence bands are shown.\n')
    write(out/'provenance.json',dict(runtime=str(root),audit_sha256=hashlib.file_digest((root/'audit/result.json').open('rb'),'sha256').hexdigest(),source_bindings_sha256=hashlib.file_digest((root/'bindings.json').open('rb'),'sha256').hexdigest(),tool=tool))
    print(json.dumps(dict(status='CURATED',registered_screen_pass=passed,per_law=accept)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();curate(a.root,a.output)
