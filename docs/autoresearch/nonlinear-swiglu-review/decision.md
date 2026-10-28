# Do not launch another Taylor-training rescue

This review has not produced a better model. The nonlinear SwiGLU correction
has a valid local Taylor interpretation, but the existing results and new
algebra do not identify a task-specific reason to spend twice the dense work
on it. No training is selected.

For `f(h)=Wo[SiLU(Wa h+ba)*(Wb h+bb)]+bo`, the full second increment is
`f + .5*J_f*f`. Differentiation includes both gate/value branches and the
dependence of the inner f on the same parameters. The [registered fixtures](protocol.md)
agree with independent AD to normalized error1.99e-16; both fixed central
differences agree to1.84e-10. These18artificial inputs are not benchmark evidence.

Actual synthetic native nominal128 means hidden85. At input width64:

| FFN | Parameters | Forward dense MACs/row | Forward+backward dense MACs/row |
|---|---:|---:|---:|
| Native SwiGLU, hidden85 |16,554|16,320|48,960|
| Full Taylor2, hidden85 |16,554|32,640|97,920|
| Recomputed-gate midpoint, hidden85 |16,554|32,640|97,920|
| Ordinary SwiGLU, hidden170 |33,044|32,640|97,920|
| Frozen-gate correction, hidden85 |16,554|27,200|81,600|

These are instrumented matrix calls with all input/parameter gradients active.
Bias/pointwise work is excluded; equal dense work is not equal wall time.
The separate image constructor uses a different width convention. Midpoint
and wide SwiGLU would be required ordinary controls for a justified proposal.

Unrestricted SwiGLU is not necessarily a globally complete vector field.
The legal scalar instance `f(y)=y*SiLU(y)=y^2*sigmoid(y)` has finite escape
time for every positive initial state. At y0=2:

```text
T_escape = integral_2^infinity (1+exp(-y))/y^2 dy
         = 0.5187671309102453 < (1+exp(-2))/2 < 1.
```

Separation of variables gives the integral; its finite value proves escape.
Two quadrature coordinates and an exponential-integral identity agree. This
refutes a universal exact-flow interpretation over duration1, not finite
feed-forward evaluation. It does not show any trained field escaped or
explain historical errors.

The [cycle01 result](../cycle01-complete-field/analysis.md) explicitly stops
Taylor-training rescue without new evidence. The [nonlinear-drift screen](../../research-architecture-search/nonlinear-drift-readout.md)
failed before K2 training; the [coupled-generator review](../../research-architecture-search/coupled-generator-prior-review.md)
already distinguishes gate feedback from ordinary extra recurrent work.
Replacing a weak field with strong SwiGLU changes the function but does not
supply the missing task premise. The escape counterexample is a caveat,
not a causal account of those failures.

The [objective-level derivation](../cycle22-denoiser-dynamics/analysis-report.md)
identifies a conditional mean and its noise-time PDE. It supplies no equation
requiring arbitrary hidden features to follow this autonomous SwiGLU field.
Therefore higher order, shared weights and cross-feature terms alone are
insufficient selection arguments. No normalization, coefficient, duration,
loss or training setting is selected to repair this proposal.

[Shazeer, 2020](https://arxiv.org/html/2002.05202v1), Section2, defines SwiGLU
and its2/3 width convention, without motivating this Taylor correction.
[Geiping et al., 2025](https://arxiv.org/html/2502.05171v2), Section3, studies
input-conditioned recurrent latent computation with normalization. Its LLM
results establish neither this operator's novelty nor an FM advantage. Both
primary methods sections were rechecked; this is not a comprehensive novelty
search or a frontier-paper recommendation.

The next proposal needs a measured regression deficit, a state/update that
addresses it, and a prediction distinguishing it from ordinary extra work.
That premise is still missing. No quality improvement should be reported
for this review. Source bb878a6 preceded execution; the Mac mini process
exited0 in0.73seconds without data, fitted models, GPU or new packages.
Small numerical records are in [results](results/). Existing P trajectories
and the submitted Report1 remain unchanged.
