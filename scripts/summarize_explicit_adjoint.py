"""Render the completed explicit-adjoint computational experiment."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(raw,output):
    audit=json.loads((raw/'audit.json').read_text())
    assert audit['status']=='PASS' and json.loads((raw/'status.json').read_text())['state']=='COMPLETE'
    output.mkdir(parents=True,exist_ok=False)
    for name in ('audit.json','comparison.csv','clock-omission.csv','panels.csv','budgets.csv','timings.json','input-bindings.json'):
        shutil.copy2(raw/name,output/name)
    omissions=list(csv.DictReader((raw/'clock-omission.csv').open()))
    times=json.loads((raw/'timings.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.0))
    names=['Gaussian2D','Ring8','Product16']
    positions=np.arange(3)
    axes[0].bar(positions-.18,[float(r['exact_mse']) for r in omissions],width=.36,label='Exact A')
    axes[0].bar(positions+.18,[float(r['omitted_clock_mse']) for r in omissions],width=.36,label='Clock adjoint omitted')
    axes[0].set(xticks=positions,xticklabels=names,yscale='log',ylabel='Full validation oracle MSE',title='Frozen ODE weights')
    axes[0].legend(fontsize=8)
    for j,mode in enumerate(('inference','loss_and_backward')):
        values=[]
        for family in ('ode_k2','swiglu128','swiglu192'):
            rows={r['method']:r for r in times if r['dataset']=='rotated_product_bimodal_16d' and r['family']==family and r['mode']==mode}
            values.append(rows['autograd']['median_ms']/rows['explicit']['median_ms'])
        axes[1].bar(positions+(-.18 if j==0 else .18),values,width=.36,label=mode.replace('_',' '))
    axes[1].axhline(1.,color='black',linewidth=.8)
    axes[1].set(xticks=positions,xticklabels=['ODE','SwiGLU128','SwiGLU192'],ylabel='Autograd / explicit median latency',title='Product16, one CPU thread, batch512')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for suffix in ('png','pdf'):fig.savefig(output/f'adjoint-study.{suffix}',dpi=180)
    plt.close(fig)
    text='''# The clock sensitivity is part of the learned ODE predictor

An explicit primal/adjoint sweep reproduces the complete potential-gradient field, including all six original ODE solves, their scalar gates, middle SiLU, normalization, FiLM, residuals and projections. It adds no parameter and calls no input-autograd routine internally. Ordinary differentiation of its tensor operations reproduces the original field's input, time and parameter derivatives.

All18 learned A/B network states across the three families/tasks passed full-field and five-panel comparisons. Maximum float64 value/input-Jacobian discrepancies are8.89e-15/7.11e-14. All-parameter/time loss gradients differ by at most2.99e-13; actual cloned AdamW updates differ by at most1.36e-14. Float32 field values are numerically compatible, with maximum observed discrepancy4.30e-6; bitwise cold-training equivalence is not claimed. Seven focused unit cases passed before the frozen evaluation. The source and input hashes are retained in the runtime and input-bindings.json.

## Removing a specific adjoint contribution

For the unchanged original solve,

```text
L(h)=h+aPh+a²/2 P²h
J_L^Tq=q+aP^Tq+a²/2(P^T)²q
        +grad(a)*dot(Ph+aP²h,q).
```

The last term sends sensitivity through the learned clock. Removing it from every reverse solve defines a different vector function while retaining the actual input-dependent gate values. No parameter is retrained in this comparison.

| Task | Exact A MSE | Clock-adjoint omitted MSE |
|---|---:|---:|
'''
    for name,r in zip(names,omissions):text+=f"| {name} | {float(r['exact_mse']):.6f} | {float(r['omitted_clock_mse']):.6f} |\n"
    text+='''
On Product16, omission raises field MSE from.03732 to.88249. The finite switch's squared change contributes.84922, while its signed residual alignment is-.00405; their sum exactly explains the risk difference. The native predictor therefore depends strongly on this clock sensitivity after learning. It is not merely applying transposed P powers. These frozen-weight cases do not prove the term caused the cold-training gain: the other weights can co-adapt, and a newly trained omission model might compensate.

![Clock omission and actual latency](adjoint-study.png)

## Implementation cost

Product16 explicit ODE inference takes a median2.882ms versus3.064ms for the original autograd construction: about1.06x speedup in this isolated CPU measurement. Its loss/backward takes9.617ms versus9.379ms, so this implementation does not improve training throughput. SwiGLU explicit versions are slower than their autograd counterparts in these tests. Repeated raw timings, quartiles and all tasks remain in timings.json; no hardware-general speed claim is made.

Dense inference MACs are182720 versus183104 for explicit/autograd ODE, and training MACs544000 versus545152. Counts exclude pointwise work. This is essentially the same matrix-work budget, not a cheaper architecture with the same quality. The explicit version also works under inference_mode, but those calls are not substituted into the matched no_grad timing comparison.

## Next bounded question

The useful next test is whether cold training can compensate for removing the clock-adjoint pathway, with the same operation applied to the analogous gate path in SwiGLU. That would test necessity under retraining, rather than turning this frozen penalty into a design claim. Paired full-explicit controls must be trained too: mathematical equivalence of implementations does not imply bitwise float32 training histories. The potential/readout success claim remains unestablished, and this experiment nominates no practical architecture.
'''
    (output/'analysis-report.md').write_text(text)
    print(json.dumps(dict(status='PASS',figure=str(output/'adjoint-study.png'))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.raw.resolve(),a.output.resolve())
