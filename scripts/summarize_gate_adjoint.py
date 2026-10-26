"""Report the audited gate-path mechanism test without further model calls."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(results,context):
    assert json.loads((results/'status.json').read_text())['state']=='COMPLETE'
    decision=json.loads((results/'decision.json').read_text())
    assert not decision['practical_candidate'] and not decision['extend_exact_factor_to_Gaussian_Ring']
    rows=list(csv.DictReader((results/'comparison.csv').open()))
    p=list(csv.DictReader((results/'p-trajectories.csv').open()))
    curves=json.loads((results/'training-curves.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for readout in ('gradient','gate_omitted'):
        arm='ode_k2_'+readout
        steps=sorted({int(r['step']) for r in p if r['arm']==arm})
        values=[np.array([float(r['sigma_max']) for r in p if r['arm']==arm and int(r['step'])==step]) for step in steps]
        assert all(len(v)==6 for v in values)
        axes[0].plot(steps,[v.mean() for v in values],label=readout.replace('_',' '))
        axes[0].fill_between(steps,[v.min() for v in values],[v.max() for v in values],alpha=.15)
        cc=[r for r in curves if r['arm']==arm and 'clip_fraction' in r]
        axes[1].plot([r['step'] for r in cc],[r['clip_fraction'] for r in cc],label=readout.replace('_',' '))
    axes[0].set(xlabel='Optimizer update',ylabel='Largest singular value of P')
    axes[1].set(xlabel='Optimizer update',ylabel='Fraction of updates clipped',ylim=(-.02,1.02))
    for ax in axes:ax.grid(alpha=.18);ax.legend(fontsize=8)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(results/f'p-and-clipping.{suffix}',dpi=180)
    plt.close(fig)
    context_rows=[r for r in csv.DictReader(context.open()) if r['dataset']=='rotated_product_bimodal_16d']
    with (results/'context-controls.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(context_rows[0]));writer.writeheader();writer.writerows(context_rows)
    text='''# Cold training compensates for part of the omitted gate sensitivity

The six-cell Product16 experiment does not support an ODE-specific gate-adjoint explanation. Removing the gate-sensitivity pathway worsens all three backbones, with a larger relative penalty for both SwiGLU controls. The registered two-task extension is not triggered; the setting stops without coefficient, clipping or training-length changes. No practical architecture is nominated.

Each backbone was cold-trained with the same full explicit adjoint or with its gate branch omitted. ODE retains both original scalar K2 solves and SiLU. SwiGLU omits the analogous channel-gate sensitivity while retaining its value pathway. Actual gate values and every ordinary derivative of the resulting vector function remain live. This is a changed forward function, not a surrogate training gradient. Full-explicit controls avoid treating mathematical identity as bitwise float32 training equivalence.

| Backbone | Full field MSE | Gate-path omitted field MSE | Omitted/full |
|---|---:|---:|---:|
'''
    for family in ('ode_k2','swiglu128','swiglu192'):
        full=next(float(r['oracle_mse']) for r in rows if r['arm']==family+'_gradient')
        omitted=next(float(r['oracle_mse']) for r in rows if r['arm']==family+'_gate_omitted')
        text+=f'| {family} | {full:.6f} | {omitted:.6f} | {omitted/full:.3f} |\n'
    text+='''
All are final full uniform-time validation oracle MSEs. Seed0,5000updates,batch512,AdamW3e-4/wd0/clip1 and all actual input streams are paired. The new full ODE matches the old A field error within0.000060% relative. Thus the numerical implementation reproduces the old operating point closely; the omitted model's.06438 error is not explained by a failed full control.

![Cold endpoints and omission ratios](gate-adjoint-comparison.png)

## What the case study changes

In the preceding frozen-weight experiment, deleting clock sensitivities from the old A ODE raised MSE from.03732 to.88249. Cold training the omitted function reaches.06438. This is a different learned weight state, not a finite switch of the same checkpoint. It shows that other parameters can compensate for much of the frozen deletion penalty under this training setup. The residual1.73x gap still supports a useful contribution, but does not establish a unique representational necessity or an ODE-specific advantage. Swi128/Swi192 suffer4.11x/2.61x penalties under their analogous gate-path removal. Their gate families differ, so those ratios are an interaction observation, not identical interventions on equivalent functions.

The ODE without clock adjoints remains much better in field MSE than the old direct ODE (.32553), while costing158144 versus106944 dense inference MACs. It still contains a complete reverse computation through transposed P polynomials, SiLU, normalization and residuals. The improvement cannot be assigned to clock derivatives alone, or claimed at equal compute. The old strong generic controls remain in context-controls.csv: Swi192 radial reaches.03254, below full ODE's.03732.

![Complete fitting curves](learning-curves.png)

## Derivatives, samples and optimization

ODE omission worsens Jacobian error8.89→15.77 and SW2 .09317→.10697. For Swi128, Jacobian error worsens14.01→39.60, while SW2 improves.09465→.09043. Swi192 similarly has worse field/Jacobian fit but lower SW2 (.09993→.09258). These metrics measure different objects; the sampling score does not erase the field-fitting failures. Full six-arm values and actual MACs remain in comparison.csv.

Both ODE arms have almost the same mean final largest P singular value:2.3267 full versus2.3180 omitted, with individual ranges2.054–2.776 and1.764–2.744. These internal spectra do not explain the field gap or certify network stability. Optimization differs: over the final500 updates, clipping fractions are.084 full and.828 omitted. We therefore do not interpret the outcome as a pure expressivity test or change clipping to rescue it.

![P spectra and the actual clipping histories](p-and-clipping.png)

## Validation and scope

The independent audit replayed all six full fields, all30 complete Jacobian panels and all six generated-sample arrays bitwise in native float32. It regenerated the actual training stream and checked24 checkpoints and48 P matrices. A separate native-network partial-Jacobian construction reproduces complete float64 output fields within9.77e-15. Thirty independent input finite differences and six parameter finite differences pass, with maximum relative errors3.45e-10 and2.86e-11. The reference fixes gate values only for the inner value computation and recomputes them at every finite perturbation; it is not a training rule. Sources, inputs and analyzers are hash-bound.

Two one-thread Mac mini workers completed the six cold runs, with ten-second process telemetry. Lenovo remains offline, confirmed by current Tailscale state and a direct SSH timeout. No idle GPU allocation or previous run modification occurred. The original submitted Report1 stays unchanged.

The protocol required an ODE omission/full ratio above2 and more than twice both generic ratios, with the generic ratios at most2, before expanding the same factor to Gaussian/Ring. That ODE-specific interaction is absent. The cold results weaken an explanation based on a uniquely useful ODE clock pathway; the broader search for a useful original-ODE architecture remains open.
'''
    (results/'analysis-report.md').write_text(text)
    print(json.dumps(dict(status='PASS',report=str(results/'analysis-report.md'))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--context',type=Path,required=True)
    a=p.parse_args();main(a.results.resolve(),a.context.resolve())
