"""Complete learned-field prediction; retain all points and numerical flags."""
import argparse
from pathlib import Path
import numpy as np
from population_resolution import read,dump,sha


def main():
    p=argparse.ArgumentParser()
    for name in ('input','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();done=read(a.input/'COMPLETE.json');verified=read(a.input/'independent.json')
    assert done['passed'] and not done['witness'] and done['refined_reference'] and done['rows']==8
    assert verified['passed'] and verified['complete_sha256']==sha(a.input/'COMPLETE.json')
    assert verified['rows_sha256']==sha(a.input/'rows.json')
    rows=read(a.input/'rows.json');assert len(rows)==8 and all(r['points']==4096 for r in rows)
    points={};results=[]
    for row in rows:
        file=a.input/row['file'];assert sha(file)==row['sha256']
        with np.load(file) as z:values={k:z[k].copy() for k in z.files}
        delta=values['crossed']-values['native'];pred=values['tangent256']
        signal=np.sum(delta**2,axis=1);error=np.sum((pred-delta)**2,axis=1)
        amplitude=np.sum(pred**2,axis=1);dot=float(np.sum(pred*delta))
        denominator=float(np.sqrt(signal.sum()*amplitude.sum()))
        bins=[]
        for i in range(5):
            t=values['time'].ravel();mask=(t>=i/5)&((t<(i+1)/5) if i<4 else (t<=1))
            energy=float(signal[mask].sum())
            bins.append(dict(lower=i/5,upper=(i+1)/5,points=int(mask.sum()),
                score=1-float(error[mask].sum())/energy if energy else None))
        result=dict(**row,relative_rms_error=float(np.sqrt(error.sum()/signal.sum())) if signal.sum() else None,
            global_alignment=dot/denominator if denominator else None,
            predicted_to_actual_rms=float(np.sqrt(amplitude.sum()/signal.sum())) if signal.sum() else None,
            zero_signal_points=int((signal==0).sum()),time_bins=bins)
        results.append(result);points[(row['case'],row['split'])]=(np.sqrt(signal),np.sqrt(amplitude))
    primary=[r for r in results if r['case']=='product_rk4_trained'];assert len(primary)==2
    qualified=all(r['numerically_qualified'] for r in primary)
    passes=qualified and all(r['score'] is not None and r['score']>0 for r in primary)
    decision='passes' if passes else ('fails' if qualified else 'is numerically unresolved')
    a.output.mkdir(parents=True,exist_ok=False);figures=a.output/'figures';figures.mkdir()
    dump(a.output/'summary.json',dict(primary_field_criterion_passes=passes,
        primary_numerically_qualified=qualified,rows=results,reference_steps=256,fitted_coefficients=0,
        source_rows_sha256=sha(a.input/'rows.json'),source_complete_sha256=sha(a.input/'COMPLETE.json'),source_sha256=sha(__file__)))
    table=['| Weights | Stream | Score vs zero | Relative RMS error | Alignment | Numerical |',
           '|---|---|---:|---:|---:|---|']
    for r in results:
        fmt=lambda x:'undefined' if x is None else f'{x:.3f}'
        table.append(f"| {r['case']} | {r['split']} | {fmt(r['score'])} | {fmt(r['relative_rms_error'])} | {fmt(r['global_alignment'])} | {'qualified' if r['numerically_qualified'] else 'unresolved'} |")
    (a.output/'table.md').write_text('\n'.join(table)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=['ring_midpoint_trained','ring_rk4_trained','product_midpoint_trained','product_rk4_trained']
    names=['Ring8 / midpoint weights','Ring8 / RK4 weights','Product16 / midpoint weights','Product16 / RK4 weights']
    colors=['#0072B2','#D55E00'];splits=['selection','confirmation']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for kind in ('field-change-norms','prediction-by-time'):
        fig,axes=plt.subplots(2,2,figsize=(10,7))
        for axis,case,name in zip(axes.flat,cases,names,strict=True):
            largest=0.
            for split,color in zip(splits,colors,strict=True):
                if kind=='field-change-norms':
                    x,y=points[(case,split)];axis.scatter(x,y,color=color,s=4,alpha=.2,label=split,rasterized=True)
                    largest=max(largest,float(x.max()),float(y.max()))
                else:
                    row=next(r for r in results if (r['case'],r['split'])==(case,split))
                    axis.plot(np.arange(.1,1,.2),[np.nan if b['score'] is None else b['score'] for b in row['time_bins']],
                        '-o',color=color,label=split,markersize=4)
            axis.set_title(name)
            if kind=='field-change-norms':
                axis.plot([0,largest],[0,largest],'--',color='.4',linewidth=.8)
                axis.set_xscale('symlog',linthresh=1e-7);axis.set_yscale('symlog',linthresh=1e-7)
                axis.set_xlabel('Actual complete-field change norm');axis.set_ylabel('Unfitted analytic prediction norm')
            else:
                axis.axhline(0,color='.5',linewidth=.8);axis.set_xlabel('Noise-time bin center')
                axis.set_ylabel('Prediction score vs zero change')
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='lower center',ncol=2,fontsize=8)
        fig.tight_layout(rect=(0,.06,1,1))
        for ext in ('png','pdf'):fig.savefig(figures/(kind+'.'+ext),dpi=180)
        plt.close(fig)
    (a.output/'analysis-report.md').write_text('# ODE-derived prediction of the complete learned field\n\n'
        'The registered refined field-prediction criterion '+decision+' on the two ProductRK4 streams. '
        'All four frozen models, both original streams and every4096-input field set are retained. '
        'The correction is derived from BOTH solvers and passed through all six local flows and the original network, without fitting its amplitude.\n\n'+
        '\n'.join(table)+'\n\n![Complete field change](figures/field-change-norms.png)\n\n'
        'Every input is shown. The diagonal compares vector norms; alignment and relative error in the table assess direction as well. '
        'Both axes use a symmetric log scale with a1e-7linear region, retaining zero responses. Norm agreement alone is insufficient.\n\n'
        '![Prediction by time](figures/prediction-by-time.png)\n\n'
        'The five prespecified equal time bins reveal whether aggregate prediction conceals a weak time range. '
        'Each score compares squared vector prediction error with predicting zero change; it is not fitted explained variance. '
        'No time bin or source is selected after evaluation.\n\n'
        '## Next causal link\n\nThe earlier full-bank study establishes a same-weight sampling benefit across the prescribed outer meshes. '
        'This field experiment tests its analytic precursor. Next propagate the analytic correction through the actual learned sampler and target functional; '
        'a field prediction alone does not quantify final distribution quality.\n\n'
        '## Claim candidates\n\n- Claim: the unfitted analytic field account '+decision+'.\n'
        '  - Evidence: all4096inputs per case, independent saved-array arithmetic and original/refined numerical flags.\n'
        '  - Allowed wording: the recorded ODE-derived approximation to complete-field solver differences.\n'
        '  - Uncertainty: fixed learned weights, finite solver budget and reference resolution.\n'
        '  - Next check: analytic forcing through the actual sampler and target metric.\n'
        '  - Decision: '+('keep for target-response testing.' if passes else 'diagnose numerical qualification or the coarse-step approximation before target claims.')+'\n')
    (a.output/'stats-appendix.md').write_text('# Empirical field-prediction quantities\n\n'
        'Each row contains all4096original field inputs for one frozen model/stream. There is no fitted coefficient or sample selection. '
        'Score=1-sum||prediction-change||^2/sum||change||^2; relative RMS error is the square root of its normalized residual. '
        'Global alignment is their vectorized cosine. Five fixed time bins retain counts and undefined zero-signal values. '
        'These are deterministic full-cohort descriptive quantities; no training-replicate significance is inferred. '
        'All original32->64and refined128->256flags remain available. Formal prediction uses the prebound256reference.\n')
    (a.output/'figure-catalog.md').write_text('# Figure purposes\n\n'
        '- field-change-norms: every actual versus unfitted predicted response norm, with identity line. Assess scale and tails, alongside vector error/alignment.\n'
        '- prediction-by-time: all five fixed noise-time bins. Assess whether the full-field account holds across the original time range.\n')
    print('ANALYTIC_DEFECT_FIELD_ANALYSIS',decision,len(results),flush=True)


if __name__=='__main__':main()
