"""Complete descriptive analysis of all fixed rare/median/probe controls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(x):
    value=float(np.hypot.reduce(np.abs(x)))
    return value if np.isfinite(value) else None
def display(x):
    if x is None:return 'unresolved'
    return f'{x:.3e}' if 0<abs(x)<.0005 else f'{x:.3f}'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();terminal=read(args.input/'TERMINAL.json')
    assert terminal['passed'] and not terminal['witness'] and len(terminal['cases'])==8
    assert not args.output.exists()
    rows=[];bindings={}
    for entry in terminal['cases']:
        assert [v['actual_terminal_exit'] for v in entry['commands']]==[0,0]
        root=Path(entry['folder']);summary=read(root/'summary.json');verified=read(root/'independent.json')
        assert verified['passed'] and verified['summary_sha256']==sha(root/'summary.json')
        assert not summary['witness'] and len(summary['roles'])==3
        bindings[str(root/'summary.json')]=sha(root/'summary.json')
        assert sha(root/'arrays.npz')==summary['binding']['arrays_sha256']
        with np.load(root/'arrays.npz') as z:a={k:z[k] for k in z.files}
        assert a['steps'].tolist()==[127,128,129,256,512]
        for role in summary['roles']:
            j=role['unique_row'];resolutions=[];differences=[]
            for i,steps in enumerate(a['steps']):
                resolutions.append(dict(steps=int(steps),tangent_norm=norm(a['tangent'][i,j]),
                    actual_displacement_norm=norm(a['crossed'][i,j]-a['native'][i,j]),
                    native_difference_from_finest_observed=norm(a['native'][i,j]-a['native'][-1,j]),
                    augmented_primal_exact=summary['steps'][i]['augmented_primal_exact']))
            tangent=a['tangent'][1,j];denominator=norm(tangent)
            for i,epsilon in enumerate(a['epsilons']):
                fd=(a['fd_plus'][i,j]-a['fd_minus'][i,j])/(2*epsilon)
                error=norm(fd-tangent)
                stored=summary['finite_differences'][i]['unique_source_comparisons'][j]
                differences.append(dict(epsilon=float(epsilon),error_norm=error,
                    relative_norm_error=error/denominator if denominator and error is not None else None,
                    component_gate_qualified=stored['qualified'],failed_components=stored['failed_components']))
            near=[v['tangent_norm'] for v in resolutions[:3]]
            rows.append(dict(case=summary['case'],split=summary['split'],role=role,
                replay_qualified=summary['replay_qualified'],resolutions=resolutions,finite_differences=differences,
                neighboring_norm_span_ratio=max(near)/min(near) if all(v is not None and v>0 for v in near) else None))
    assert len(rows)==24
    args.output.mkdir(parents=True);figures=args.output/'figures';figures.mkdir()
    output=dict(scope='all8cases/24prebound roles; selected-source diagnosis, not generation quality',
        rows=rows,source_bindings=bindings,source_sha256=sha(__file__),new_training=0,
        primary_target_gate_changed=False,finest_observed_steps=512,
        finest_resolution_is_not_a_continuous_reference=True)
    (args.output/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    table=['| Case | Split | Max-source tangent127 | 128 | 129 | 256 | 512 | FD1e-5 | FD1e-6 |',
           '|---|---|---:|---:|---:|---:|---:|---|---|']
    for row in rows:
        if row['role']['role']!='maximum':continue
        values=' | '.join(display(v['tangent_norm']) for v in row['resolutions'])
        flags=['pass' if row['finite_differences'][i]['component_gate_qualified'] else 'fail' for i in (1,2)]
        table.append(f"| {row['case']} | {row['split']} | {values} | {' | '.join(flags)} |")
    (args.output/'table.md').write_text('\n'.join(table)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    order=['ring_midpoint_trained','ring_rk4_trained','product_midpoint_trained','product_rk4_trained']
    names=['Ring8 / midpoint weights','Ring8 / RK4 weights','Product16 / midpoint weights','Product16 / RK4 weights']
    styles=[('maximum','-','o'),('lower_median','--','s'),('probe_maximum',':','^')]
    colors=['#0072B2','#D55E00']
    for kind in ('resolution-sensitivity','finite-difference-convergence'):
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for axis,case,name in zip(axes.flat,order,names,strict=True):
            for split,color in zip(['selection','confirmation'],colors,strict=True):
                for role,style,marker in styles:
                    data=next(v for v in rows if v['case']==case and v['split']==split and v['role']['role']==role)
                    if kind=='resolution-sensitivity':
                        xs=np.arange(5);ys=[v['tangent_norm'] for v in data['resolutions']]
                    else:
                        xs=[v['epsilon'] for v in data['finite_differences']]
                        ys=[v['relative_norm_error'] for v in data['finite_differences']]
                    plotted=[v if v is not None and v>0 else np.nan for v in ys]
                    axis.plot(xs,plotted,linestyle=style,marker=marker,color=color,markersize=4)
            axis.set_title(name);axis.set_yscale('log')
            if kind=='resolution-sensitivity':
                axis.set_xticks(range(5),['127','128','129','256','512'])
                axis.set_xlabel('Native Heun steps (categorical spacing)');axis.set_ylabel('Endpoint tangent norm')
            else:
                axis.set_xscale('log');axis.set_xlabel('Central perturbation epsilon');axis.set_ylabel('Relative tangent norm error')
        legend=[Line2D([0],[0],color=c,label=s) for s,c in zip(['selection','confirmation'],colors,strict=True)]
        legend += [Line2D([0],[0],color='.2',linestyle=style,marker=marker,label=role.replace('_',' ')) for role,style,marker in styles]
        fig.legend(handles=legend,loc='lower center',ncol=3,fontsize=9)
        fig.tight_layout(rect=(0,.10,1,1))
        for suffix in ('png','pdf'):fig.savefig(figures/(kind+'.'+suffix),dpi=180)
        plt.close(fig)
    (args.output/'analysis-report.md').write_text('# Rare-source derivative and sampler-resolution diagnosis\n\n'
        'All four frozen weights, both splits and all24prebound roles are retained. '
        'The unchanged sampler uses all five registered step counts. '+
        'There is no generation-quality estimate from these selected sources.\n\n'+ '\n'.join(table)+
        '\n\n![Resolution](figures/resolution-sensitivity.png)\n\n'
        'All maximum, lower-median and original-probe controls are shown. '
        'The512-step run is the finest observed value, not a certified continuous solution. '
        'Nearby non-dyadic counts and finer counts answer different resolution questions.\n\n'
        '![Finite differences](figures/finite-difference-convergence.png)\n\n'
        'Every registered perturbation scale is kept. The graph shows relative vector-norm error; '
        'the table/JSON separately retain the component gate, which has an absolute term. '
        'Do not turn convergence at one scale into agreement at every scale.\n')
    (args.output/'figure-catalog.md').write_text('# Figure purposes\n\n'
        '- resolution-sensitivity: all24roles/five counts, categorical spacing. Compare source-local sensitivity across neighboring and finer time meshes. '
        'Selected-source diagnosis; no population error bars or sampler ranking.\n'
        '- finite-difference-convergence: all24roles/four perturbation scales. Compare the native128tangent with independent endpoint perturbations. '
        'Relative L2error and the component qualification are different quantities. Raw zeros/nonfinite values remain in JSON; log display omits them.\n')
    print('EXTREME_COMPLETE_ANALYSIS',len(rows),flush=True)


if __name__=='__main__':main()
