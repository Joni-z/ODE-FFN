# Decision after the complete error-transport test

**Stop using the full-scale oracle-path first-order approximation as a general
explanation of the learned models. Return the next diagnosis to the actual
hidden ODE computation.** This decision follows completed numerical checks;
it is not a response to an unresolved solver failure.

## Completed result

All twelve models and their independent verifiers exit0. Every original
256source is retained and every endpoint/oracle/tangent precision gate passes:
eight cases at4096steps, four at8192. Independent metric error is2.14e-14.

| Model | Ring8 endpoint fraction | Product16 endpoint fraction |
|---|---:|---:|
| Shared P |0.372|-3.128|
| Released stages |0.343|-11.443|
| Palindrome |0.207|-8.097|
| Alternating |0.273|-1.941|
| Free ISRU |0.137|-2.602|
| SwiGLU reference |0.101|-1.849|

Positive means the first-order prediction improves on zero paired endpoint
displacement. It is not a generation score. Ring8 has six positive point
estimates with modest reduction in residual energy; only released stages
has a positive lower nominal twelve-comparison-adjusted bank interval.
Product16 has six negative point estimates. Their wide input-bank intervals
cross zero, so no training-population rejection is inferred from this cohort.
[Full values, intervals and figures](refined-trajectory-results/analysis-report.md).

The [actual-law control](product-oracle-control-results/analysis-report.md)
is decisive about interpretation: the Product16 marginals stay exactly correct
while its full-scale fraction is-12.579. Thus even perfect generation does
not require this approximation to succeed. The learned models are not assumed
to have that special error. The original fixed-grid SW2scores remain intact.

## What is established about the architecture

| Proposed source of value | Current evidence | Disposition |
|---|---|---|
| Prescribed exponential coefficients | Released coefficients are active and attain similar sampling scores. | No demonstrated necessity of factorial tying on these tasks. |
| Accurate hidden continuous evolution | Fixed midpoint has measured second-order convergence; refining the trained hidden map can worsen Product16sampling. | A valid solver is not itself a task advantage. |
| One shared generator | Sharing retains a within-family Product16SW2lead after the optimizer-coordinate control; ordinary SwiGLU is stronger. | A useful task-dependent reuse constraint, with no established general ODE-specific gain. |
| Symmetric stage ordering | Leading bracket cancellation holds, but sampling helps Ring8 and hurts Product16. | Mathematical cancellation is active; its utility is task dependent. |
| Whole-output derivative agreement | The focal PAL/ALTresponse association confirms on all eight input banks. | Association, with ordinary-model counterexamples to a universal ranking. |
| First-order error transport | Numerically resolved, weak on Ring8 and unsuccessful on the fixed Product16population. | Do not extrapolate it into a penalty or claim it explains generation. |

The matched image comparison also remains negative for current frozen-diagonal
ISRU-D: at160epochs native/K1/K8FIDis24.304/27.789/28.083, with near-equal
parameter counts. [Original completed comparison](../../../celeba128-wide-20261114/continuation-160/matched-results.md).
Those values establish the tested budget, not convergence of every trajectory.

## Next bounded hypothesis

Investigate the difference between the learned four-step hidden computation
and its continuous limit. The ISRU flow has an exact feature-potential balance:
the symmetric part of Pchanges that potential; the skew part redirects features
without changing it instantaneously. A finite solver can violate this balance.
That gives a direct, falsifiable question about the ODElayer, whereas the failed
endpoint approximation applied equally to every architecture.

The [next energy/direction protocol](../hidden-energy-protocol.md) uses existing
checkpoints and fixed entrance states only. It first checks whether numerical
energy distortion explains a material part of the hidden-map change. If it does
not, reject that account and retain the measured directional change. No response
search, new training, spectral penalty or image promotion follows this decision.
