# Can cold training compensate for removal of the gate adjoint?

Registered after the explicit-adjoint frozen-weight experiment and before all new cold-training outcomes. This tests a mechanistic explanation of the original ODE A gain; no new practical architecture or potential-readout rescue is eligible.

## Factor and mathematical meaning

The full explicit model computes A=.5*(G+J_G^T x), mathematically identical to the completed A readout. In its ODE reverse solves retain or remove only

```text
grad(a)*dot(Ph+aP²h,q).
```

For SwiGLU `y=W3[SiLU(g)*v]+b`, the corresponding gate-sensitivity branch is `Wg^T[(W3^Tq)*v*SiLU'(g)]`. Its value-sensitivity branch `Wv^T[(W3^Tq)*SiLU(g)]` remains. This is a structural analogy, not an assertion of equivalent gate parameterizations: ODE has scalar sigmoid durations and SwiGLU channel SiLU gates.

In all arms the raw primal G, gate values and ordinary training derivatives stay live. The omission is part of the forward vector-function definition, not a detached or surrogate training gradient. Differentiate the explicit omitted function completely, including all input dependence still present in its values and the readout seed x. No changed loss or target.

## Phase A: six paired Product16 cells

Use original ODE K2, SwiGLU128 and SwiGLU192, each with full or omitted gate-adjoint. Both full-explicit controls and omitted variants are cold-trained; reused native-autograd A is contextual, not the paired causal comparator, because exact mathematical equivalence does not imply bitwise float32 training histories.

All paired starting tensors, nonpersistent buffers, data RNG and initialization RNG must match exactly. Width64/depth3, two original scalar K2 solves and SiLU, same surrounding backbone, seed0,5000updates,batch512,AdamW3e-4/wd0/clip1, uniform-time cached-bank pair FM. Actual draws and hash streams match previous studies. No analytic training supervision, distillation, altered normalization or new hyperparameter.

Verify finite differences of both families' complete altered functions, all-parameter/input/time derivatives, real trainer step3 reloads and exact repeated child execution. Capture actual ODE forward gate inputs through the first clock projection, since the explicit implementation does not call the ODELayer container's forward hook. Preserve all six P matrices at steps0/1000/2500/5000, visits, complete fitting/gradient/clipping histories, full uniform-time field MSE, all five512-input Jacobian panels, Heun32 SW2 with2048 common starts, executed dense costs and10-second actual process telemetry.

## Interpretation and stop/extension rule

Primary descriptive quantity: omitted/full final field MSE within each family, plus the ODE ratio relative to both generic ratios. No training-variability significance claim.

First require full-explicit ODE Product16 MSE within10% of the old A value .03732037. If it fails this contextual compatibility check, report the paired new result but do not claim to explain the old A gain; investigate implementation/data evidence without tuning or continuing these runs.

Only if (1) the ODE omission/full ratio exceeds2, (2) it exceeds twice the larger generic ratio, and (3) both generic ratios are at most2, is a two-task Gaussian/Ring replication of this exact six-arm factor warranted. Otherwise stop Phase A without more tasks, coefficients, architectures or extra updates. A ratio at most1.25 weakens necessity of the removed pathway under this tested retraining budget. Intermediate outcomes remain unresolved; they are not rescued. Failure after5000 steps is not an approximation impossibility proof or a unique distinction between representation and optimization.

Removing a derivative branch changes the vector-function family, optimization geometry, compute and conservativity together. A positive ODE-specific interaction supports investigating how the gate-adjoint supplies nonlinear directions, but does not isolate hard conservativity or prove that a local symmetric block will work. Frozen omission penalties are not retraining predictions. No variant is promoted as an architecture from this mechanism study, regardless of its raw MSE.

Two one-thread Mac mini CPU workers; no GPU allocation while Lenovo remains offline. Do not restart any process based on observation timeout. Independently verify actual output arrays, spectra and regenerated streams before interpreting outcomes.
