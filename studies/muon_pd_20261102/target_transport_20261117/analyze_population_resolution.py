"""All registered meshes and paired bank effects; no best-mesh selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_transport import compositions, interval, sha


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser()
    for name in ('input','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();terminal=read(a.input/'TERMINAL.json')
    assert terminal['passed'] and len(terminal['cases'])==8
    assert all(r['actual_terminal_exit']==0 for r in terminal['cases'])
    counts,weights=compositions();rows=[];bindings={}
    for entry in terminal['cases']:
        if entry['stage']!='evaluate':continue
        assert Path(entry['output']).name==entry['case']
        root=a.input/entry['case'];done=read(root/'COMPLETE.json');check=read(root/'independent.json')
        assert done['passed'] and not done['witness'] and done['rows']==10
        assert check['passed'] and check['complete_sha256']==sha(root/'COMPLETE.json')
        assert check['rows_sha256']==sha(root/'rows.json')
        results=read(root/'rows.json');bindings[str(root/'rows.json')]=sha(root/'rows.json')
        for r in results:
            assert r['points_per_bank']==4096
            assert sha(root/r['file'])==r['sha256']
            native=np.array(r['native']['bank_sw2']);crossed=np.array(r['crossed']['bank_sw2'])
            delta=crossed**2-native**2
            row=dict(**r,bank_loss_change=delta.tolist(),bank_change_mean=float(delta.mean()),
                bank_change_sd=float(delta.std(ddof=1)),banks_improving=int((delta<0).sum()),
                ci95=interval(counts@delta/8,weights,.025),
                ci_adjusted=None if r['steps']==128 else interval(counts@delta/8,weights,.05/64),
                pooled_loss_change=r['crossed']['pooled_sw2']**2-r['native']['pooled_sw2']**2)
            rows.append(row)
    assert len(rows)==40 and len({(r['case'],r['split'],r['steps']) for r in rows})==40
    primary=[r for r in rows if r['case']=='product_rk4_trained' and r['steps']==2048]
    assert len(primary)==2
    survives=all(r['pooled_loss_change']<0 and r['ci_adjusted'][1]<0 for r in primary)
    a.output.mkdir(parents=True,exist_ok=False);figures=a.output/'figures';figures.mkdir()
    summary=dict(primary_finer_mesh_benefit_survives=survives,primary_case='product_rk4_trained',
        primary_steps=2048,rows=rows,source_bindings=bindings,bootstrap_multisets=6435,
        bootstrap_ordered_samples=8**8,contrast_family=32,adjusted_two_sided_tail=.05/64,
        training_seed=0,paired_banks=8,points_per_bank=4096,source_sha256=sha(__file__))
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    table=['| Weights | Split | Heun steps | Native SW2 | Crossed SW2 | Banks improving |',
           '|---|---|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda v:(v['case'],v['split'],v['steps'])):
        table.append(f"| {r['case']} | {r['split']} | {r['steps']} | {r['native']['pooled_sw2']:.3f} | {r['crossed']['pooled_sw2']:.3f} | {r['banks_improving']}/8 |")
    (a.output/'table.md').write_text('\n'.join(table)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    cases=['ring_midpoint_trained','ring_rk4_trained','product_midpoint_trained','product_rk4_trained']
    names=['Ring8 / midpoint weights','Ring8 / RK4 weights','Product16 / midpoint weights','Product16 / RK4 weights']
    splits=['selection','confirmation'];colors=['#0072B2','#D55E00'];steps=[127,128,129,512,2048]
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for axis,case,name in zip(axes.flat,cases,names,strict=True):
        for split,color in zip(splits,colors,strict=True):
            data=[next(r for r in rows if (r['case'],r['split'],r['steps'])==(case,split,n)) for n in steps]
            for kind,style in [('native','-o'),('crossed','--s')]:
                axis.plot(range(5),[r[kind]['pooled_sw2'] for r in data],style,color=color,label=split+' '+kind,markersize=4)
        axis.set_title(name);axis.set_xticks(range(5),[str(n) for n in steps]);axis.set_xlabel('Outer Heun steps (categorical)')
        axis.set_ylabel('Pooled SW2');axis.legend(fontsize=7)
    for ext in ('png','pdf'):fig.savefig(figures/('pooled-resolution.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for axis,case,name in zip(axes.flat,cases,names,strict=True):
        for s,(split,color) in enumerate(zip(splits,colors,strict=True)):
            for i,n in enumerate([127,129,512,2048]):
                r=next(r for r in rows if (r['case'],r['split'],r['steps'])==(case,split,n))
                x=i+(-.12 if s==0 else .12);low,high=r['ci_adjusted']
                axis.scatter(x+np.linspace(-.035,.035,8),r['bank_loss_change'],color=color,s=12,alpha=.5)
                axis.plot([x,x],[low,high],color=color,linewidth=2)
                axis.scatter([x],[r['bank_change_mean']],color=color,marker='D',s=25,label=split if i==0 else None)
        axis.axhline(0,color='.5',linewidth=.8);axis.set_title(name);axis.set_xticks(range(4),['127','129','512','2048'])
        axis.set_xlabel('Outer Heun steps (categorical)');axis.set_ylabel('Crossed - native bank squared SW2')
        axis.ticklabel_format(axis='y',style='sci',scilimits=(0,0));axis.legend(fontsize=8)
    for ext in ('png','pdf'):fig.savefig(figures/('paired-effects.'+ext),dpi=180)
    plt.close(fig)
    decision=('The registered ProductRK4benefit survives the2048-step test on both streams.' if survives else
              'The registered ProductRK4benefit does not meet the2048-step criterion on both streams.')
    (a.output/'analysis-report.md').write_text('# Full-bank outer-resolution intervention\n\n'+decision+
        '\n\nAll four frozen20kweights, both original streams, eight4096-source banks and five prescribed meshes are retained. '
        'All formal producers/independent verifiers terminate0. Full-bank128endpoint/metric replay gates pass before new meshes. '
        'No model is retrained or best mesh selected.\n\n![Pooled](figures/pooled-resolution.png)\n\n'
        'Compare the same-weight native/crossed gap across neighboring and finer time meshes.2048is a finer observation, not a continuous reference.\n\n'+
        '\n'.join(table)+'\n\n![Paired](figures/paired-effects.png)\n\n'
        'Points retain every paired bank; diamonds show means and lines the32-contrast-adjusted conditional bootstrap intervals. '
        'A negative value favors the crossed hidden solver. These are fixed-weight evaluation draws.\n\n'
        '## Claim candidates\n\n- Claim: '+decision+
        '\n  - Evidence: complete same-source sampling, full128identity gates, independently reconstructed metrics and exact bootstrap enumeration.\n'
        '  - Allowed wording: the observed same-weight solver substitution has the recorded outer-mesh dependence.\n'
        '  - Uncertainty: finite meshes, fixed learned models, empirical target banks.\n'
        '  - Next check: '+('apply the analytic two-solver defect to the complete field and target functional.' if survives else 'diagnose outer-time integration before promoting a hidden-ODE generation mechanism.')+
        '\n  - Decision: '+('keep for the analytic mechanism test.' if survives else 'revise the mesh-independent mechanism hypothesis.')+'\n')
    (a.output/'stats-appendix.md').write_text('# Statistical scope\n\n'
        'Eight paired4096-source/target banks per fixed weight/stream. Summary records every bank effect, mean, sampleSD,95%CIand adjustedCI. '
        'Loss changes use squared SW2; pooled SW2and bank-mean effects have different units. Enumerate all6435multinomial vectors, multiplicity8^8. '
        'The32new-mesh contrasts use tail.05/64. Cached-reference128is outside this family. Percentile intervals are conditional on the trained weights; '
        'coverage is not guaranteed and the banks are not independent training replicates. Primary2048requires both pooled effects and adjusted upper bounds negative. '
        'No ordinary architecture ranking or continuous-convergence claim.\n')
    (a.output/'figure-catalog.md').write_text('# Figure purposes\n\n'
        '- pooled-resolution: all40pooled native/crossed comparisons. No error bars: empirical pooled scores. Tests whether the useful gap depends on the sampling mesh.\n'
        '- paired-effects: all32new-mesh contrasts, every bank and adjusted conditional intervals. Tests whether pooled directions have paired-bank support.\n')
    print('POPULATION_RESOLUTION_ANALYSIS',survives,len(rows),flush=True)


if __name__=='__main__':main()
