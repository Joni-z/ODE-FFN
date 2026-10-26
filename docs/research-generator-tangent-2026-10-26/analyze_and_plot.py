"""Readable tables and one figure from the independently audited diagnosis."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/Users/mr.z/ode-workspace-prep/architecture-search/generator_tangent_20261026')
DOC=Path(__file__).parent
def load(p):return json.loads(p.read_text())
def table(name,rows):
    with (DOC/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
def short(layer):
    a=layer.split('.');return f'Block {int(a[1])+1}, solve {a[-1][-1]}'

def main():
    out=ROOT/'output';assert load(out/'status.json')['state']=='COMPLETE'
    assert load(ROOT/'independent-audit/audit.json')['status']=='PASS'
    pooled=load(out/'aggregates.json');panels=load(out/'panels.json');spectral=load(out/'spectral-context.json');clocks=load(out/'clocks.json')
    table('aggregate-routes.csv',pooled);table('all-time-routes.csv',panels);table('spectral-context.csv',spectral);table('clock-statistics.csv',clocks)
    names=list(dict.fromkeys(r['layer'] for r in pooled));bounds=[]
    with np.load(out/'matrices.npz') as data:
        for name in names:
            p=data['P__'+name];mu=float(np.linalg.eigvalsh((p+p.T)*.5)[0]);pnorm=float(np.linalg.svd(p,compute_uv=False)[0])
            lower=min(1+.2*mu,1+mu);upper=1+pnorm
            assert lower>0
            bounds.append(dict(layer=name,symmetric_P_minimum=mu,P_spectral_norm=pnorm,
                allowed_duration_minimum=.2,allowed_duration_maximum=1.,normalized_singular_lower_bound=lower,
                normalized_singular_upper_bound=upper,condition_upper_bound=upper/lower))
    table('uniform-differential-bounds.csv',bounds)
    figdir=DOC/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12.3,4.2),layout='constrained');x=np.arange(6)
    labels=[f'B{i+1}\nsolve {j}' for i in range(3) for j in [1,2]]
    for ax,objective,title in zip(axes,['nonlinear','total'],['Nonlinear odd-error gradient (primary)','Total paired-error gradient (control)']):
        rows=[next(r for r in pooled if r['objective']==objective and r['layer']==name) for name in names]
        ax.bar(x-.19,[r['local_ratio'] for r in rows],width=.38,color='#0072B2',label='Per-input energy ratio')
        ax.bar(x+.19,[r['coherent_ratio'] for r in rows],width=.38,color='#E69F00',label='Summed-gradient energy ratio')
        ax.axhline(.1,color='#D55E00',linestyle='--',linewidth=1.5,label='Severe cancellation: 0.10')
        ax.axhline(1,color='#777777',linestyle=':',linewidth=1)
        ax.set(xticks=x,xticklabels=labels,ylim=(0,2),ylabel='Combined energy / separate-route energy',title=title)
        for j,r in enumerate(rows):
            ax.text(j-.19,r['local_ratio']+.03,f"{r['local_ratio']:.2f}",ha='center',fontsize=8)
            ax.text(j+.19,r['coherent_ratio']+.03,f"{r['coherent_ratio']:.2f}",ha='center',fontsize=8)
    axes[0].legend(fontsize=8,loc='upper left')
    for ext in ['png','pdf']:fig.savefig(figdir/f'P-gradient-routes.{ext}',dpi=180)
    plt.close(fig)
    primary=[r for r in pooled if r['objective']=='nonlinear'];secondary=[r for r in pooled if r['objective']=='total'];time_primary=[r for r in panels if r['objective']=='nonlinear']
    local=[r['local_ratio'] for r in primary];coherent=[r['coherent_ratio'] for r in primary]
    native=load(out/'audit.json');auditor=load(ROOT/'independent-audit/audit.json')
    primary_table='\n'.join(f"| {short(r['layer'])} | {r['local_ratio']:.4f} | {r['coherent_ratio']:.4f} |" for r in primary)
    bound_table='\n'.join(f"| {short(r['layer'])} | {r['normalized_singular_lower_bound']:.4f} | {r['condition_upper_bound']:.2f} |" for r in bounds)
    report=f'''# The trained K2 polynomial does not show severe gradient-route cancellation

The proposed obstruction is absent in this frozen predictor. Its linear and quadratic P routes do not nearly cancel in either the nonlinear-error or total-error gradient. All six matrices also admit a positive lower bound on the local matrix-polynomial differential throughout the original allowed duration interval. These findings do not identify a better architecture or establish good conditioning of the complete network.

The original layer uses `Q_a(P)=I+aP+a²P²/2`. Its matrix derivative in direction H is `aH+a²(PH+HP)/2`. At `P=-I/a` the derivative vanishes although the forward map is `I/2`. The earlier project theory already contained this stationary-point example. This study asks whether that mechanism is visible in the actual trained scalar K2 model.

We use all five population panels and all 4096 antithetic row pairs from the preceding [parity study](../research-parity-2026-10-26/analysis-report.md). The measured adjoints include the complete downstream network and all later gates. The primary objective is the nonlinear odd-error component, whose raw-linear projection and envelope derivative are kept explicit. No predictor is changed or optimized.

For each P, we sum the squared norms of its two derivative routes and compare them with their combined squared norm. A ratio of one means zero net cross term; above one means constructive interaction. The registered severe-cancellation threshold is 0.10. Per-input energies and coherent sums across all rows/times are reported separately.

| Original solve | Per-input energy ratio | Coherent-gradient ratio |
|---|---:|---:|
{primary_table}

All six primary ratios are far above the threshold. Local ratios range from {min(local):.3f} to {max(local):.3f}; coherent ratios range from {min(coherent):.3f} to {max(coherent):.3f}. The second solve in block 2 has some destructive interaction, but its coherent ratio is {primary[3]['coherent_ratio']:.3f}, not a near-zero response. Across all 30 individual primary time/solve panels, the minimum coherent ratio is {min(r['coherent_ratio'] for r in time_primary):.3f}. Total-error controls reach the same qualitative conclusion; every row is retained in [the time table](all-time-routes.csv).

![Both objectives and all six original solves](figures/P-gradient-routes.png)

There is also an algebraic conditioning certificate for these particular frozen matrices. Let mu be the smallest eigenvalue of `(P+P^T)/2`. In the Frobenius inner product, the normalized differential `T(H)=H+a(PH+HP)/2` satisfies

```text
<H,T(H)> >= (1+a*mu)*||H||_F²
||T(H)||_F <= (1+a*||P||2)*||H||_F.
```

Whenever the lower factor is positive, Cauchy–Schwarz bounds the minimum singular value by that factor. This argument does not require P to be normal. Each measured mu is greater than -1, so the certificate extends to every a in the declared (.2,1) interval. The following uniform bounds are an algebraic corollary calculated after inspecting the final matrices, not a replacement for the registered gradient criterion.

| Original solve | Uniform normalized singular lower bound | Uniform condition upper bound |
|---|---:|---:|
{bound_table}

The smallest uniform lower bound is {min(r['normalized_singular_lower_bound'] for r in bounds):.4f}; the largest condition upper bound is {max(r['condition_upper_bound'] for r in bounds):.2f}. These concern P-to-Q matrix sensitivity at a fixed duration. Applying Q to a particular input, differentiating clocks, composing normalization and other blocks, and optimizing the loss introduce different conditioning questions. The table does not certify the complete field Jacobian or the history between saved training checkpoints.

The result therefore provides no support for selecting a new P parameterization solely to remove K2 route cancellation. It also does not prove that the function class is adequate or that training has reached its best solution. The preceding evidence still places the large direct-model deficit in useful nonlinear fitting; a future design needs a different, specific mechanism and practical comparison with strong SwiGLU controls.

The main analytic sums match complete-network autograd to {native['maximum_native_P_gradient_discrepancy']:.2e}. Independent explicit per-row outer products reproduce every panel, aggregate and decision, with maximum matrix discrepancy {auditor['maximum_matrix_discrepancy']:.2e}. The main took 1.79 seconds and the independent replay 16.72 seconds on one Mac mini CPU thread. Lenovo SSH timed out, so no remote job or GPU was reserved. No training, optimizer update, commit or push occurred; earlier results and submitted Report1 remain unchanged.

The [protocol](protocol.md), [mathematical details](mathematics.md), [aggregate table](aggregate-routes.csv), [uniform bounds](uniform-differential-bounds.csv), and [source/execution notes](notes.md) preserve the scope and evidence. No architecture is nominated, and the useful-ODELayer goal remains active.
'''
    (DOC/'analysis-report.md').write_text(report)
    receipt=dict(status='PASS',runtime=str(ROOT),panel_rows=len(panels),aggregate_rows=len(pooled),spectral_rows=len(spectral),clock_rows=len(clocks),
        uniform_bound_rows=len(bounds),figures=1,primary_local_range=[min(local),max(local)],primary_coherent_range=[min(coherent),max(coherent)],
        minimum_uniform_lower_bound=min(r['normalized_singular_lower_bound'] for r in bounds),maximum_uniform_condition_bound=max(r['condition_upper_bound'] for r in bounds),architecture_selected=False)
    (DOC/'analysis-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))

if __name__=='__main__':main()
