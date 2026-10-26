# Stage7: change the internal field and isolate returned-state semantics

Registered before this stage's training outcomes. The preceding product16 result is decisive: vectorK1 validation oracle MSE.06459 versus full SwiGLU.01523, with much higher derivative and well-recovery errors. The scalar conditional clock-rank bound explains only7–12% of its current Jacobian residual. More clock outputs or controller reuse does not address this gap. Test the base operator instead of expanding the gate or adding a constraint.

## Relation to the professor block

Retain the same two learned bias-free square matrices P1/P2, the same SiLU, and the first vector controller g(x). Construct the complete previous reference before retaining these tensors so all initial common tensors and subsequent RNG draws are paired. The reference has no extra learned output projection or normalization. The one-controller split-linear map is

```text
F_split(x) = (I+D(x)P2) SiLU((I+D(x)P1)x).
```

The candidate instead defines one nonlinear internal field, with its controller computed once from the initial state:

```text
D0 = diag(s(x))
f_D0(y) = D0 P2 SiLU(P1 y).
```

For this fixed D0, evolve y'(tau)=f_D0(y), y(0)=x. At K1, delta(x)=D0 P2 SiLU(P1x). Compare returning delta versus x+delta. Inside the unchanged outer residual, the map arm injects an additional alpha*z (the normalized/conditioned FFN input), while the increment arm omits it. Both are legitimate network parameterizations; the old arm is not declared mathematically wrong. These two arms differ only by the returned internal state and have identical tensors, gate values, parameter counts and MACs.

The split-linear reference versus nonlinear-map contrast changes the internal operator using exactly the same tensors and budget. It is not a pure reparameterization: the function classes and initial hidden functions differ. The map/increment contrast then isolates the return choice. All complete models initially predict zero because of the common zero output head, but their hidden features and subsequent gradients differ.

## Differential and Taylor structure

The frozen internal field derivative is

```text
J_fD(y) = D0 P2 diag(SiLU'(P1y)) P1.
```

The complete K1 increment differentiated through its initial-state controller is

```text
J_delta(x) = D(x) P2 diag(SiLU'(P1x)) P1
             + diag(P2 SiLU(P1x)) J_s(x).
```

Thus the base field itself is nonlinear before the clock correction; this is the structural distinction from h+D(h)Ph. It does not prove FM-specific benefit. At K1, fixed D=.6I is exactly an ordinary bias-free two-matrix SiLU MLP. Include that rival and credit a K1 gain to the nonlinear field/gating/return choice, never to high-order integration.

The correct second Taylor coefficient for this frozen D0 is available for verification:

```text
d1 = D0 P2 SiLU(P1x)
d2 = .5 D0 P2 [SiLU'(P1x) * (P1 d1)]
y(tau)-x = tau*d1 + tau^2*d2 + O(tau^3).
```

It is generally not P-squared. The controller is held fixed during the internal solve; this is not the autonomous field D(y)P2SiLU(P1y), whose second coefficient would contain an additional gate derivative. Backpropagation through the final model still differentiates D(x). Unit tests compare d2 with an exact autograd field-JVP and check local error scaling against high-resolution RK4. K2 is not trained in this first screen.

SiLU' obeys the conservative bound |SiLU'|<=1+1/e. Therefore the fixed-D0 field is globally Lipschitz with constant at most s_max*||P2||_2*(1+1/e)*||P1||_2. The smooth field has a unique global flow and bounded finite-horizon growth. For a small horizon, Taylor order statements follow on its bounded trajectory. They do not guarantee a small truncation error at horizon1 with learned large matrices. An initial-state-dependent D also does not automatically make the combined map globally invertible. P1/P2 spectra are logged as field factors, not as the spectrum of this nonlinear generator or the complete velocity Jacobian.

## Six fixed arms

1. `linear_reuse`: preceding split-linear K1 map, one reused vector controller.
2. `drift_map`: nonlinear K1 field, return x+delta.
3. `drift_increment`: same nonlinear K1 field, return delta.
4. `drift_fixed`: fixed-rate nonlinear increment; exact ordinary-MLP rival.
5. `swiglu_budget`: nominal hidden96, actual hidden64.
6. `swiglu`: preceding full reference.

At dim2 the three input-gated arms have69282 parameters/67904 MACs. Budget SwiGLU has69570/67904. Fixed-rate MLP has56706/55616, so any loss of generic-gating advantage is visible rather than hidden by comparing only larger rivals. The original controller hidden32, tau10, range(.2,1), width64, depth3 and all training settings remain fixed. No new spectral limit, loss, initialization, activation or time sampler is added.

## Screen and conditional continuation

Use Gaussian2D and Ring8 from the original immutable v1 and product16 from its frozen manifest. Each dataset gets all six arms, seed0,5000updates,batch512 and the existing optimizer/evaluation settings, with two monitored CPU workers. Final5000 random-time validation oracle MSE is primary. Retain common-point field/Jacobian error, sample SW2, trajectories and measured cost. Different data roots are separate registered queue stages. Historical studies remain untouched.

A gated nonlinear increment may advance if it reduces oracle MSE by at least10% versus the same-budget SwiGLU on Ring8 or product16, is within5% on the other nonlinear task and the Gaussian control, and does not worsen either nonlinear task's common Jacobian error or validation SW2 by more than10%. All contrasts are practical screens, not significance claims. Report map/increment and fixed-gate comparisons before attributing a mechanism. A fixed ordinary MLP that matches or beats the gated candidate blocks an ODE/gating-specific explanation; do not call high-order structure established by a K1 result.

Passing leads to an unchanged20000-step continuation, fresh frozen holdout inputs and actual latency/throughput checks. Earlier Ring8/product16 test banks are already exposed. A later independent confirmation bank is generated from the declared population using SeedSequence([0,dataset_identifier,20261024,7]) with disjoint streams for targets/source/directions; this does not add a training seed. No test outcome is read for this screen. Failed candidates are not rescued by adding K2 or widening controllers.
