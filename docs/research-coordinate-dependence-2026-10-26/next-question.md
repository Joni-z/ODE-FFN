# A local primal/adjoint question, not a selected architecture

This post-result note follows the completed coordinate-redraw experiment. The existing whole-field potential, recurrence, gate-omission and P-depth settings remain stopped. No new cold run is selected here. The new evidence narrows the large original deficit to the relevant conditional response; it does not identify missing clock gradients as the cause or prove that local gradients reproduce the complete-field result.

## One concrete alternative to two independent solves

Let L(h) be exactly one original scalar K2 ODELayer, with its input-selected duration evaluated once:

```text
L(h)=h+a(h)Ph+.5*a(h)^2*P^2*h
F(h)=J_L(h)^T * SiLU(L(h)).
```

This replaces the second independent map in the original `L2(SiLU(L1(h)))` by the exact input adjoint of the first. It preserves the original conditional Taylor encoder and central SiLU, and reuses the encoder parameters. If rho'=SiLU, then

```text
F(h)=grad_h sum_j rho(L_j(h)).
```

No primitive needs to be numerically integrated for implementation. The already verified original-layer adjoint gives an ordinary-tensor evaluation:

```text
y=L(h), q=SiLU(y), u=Ph+a*P^2*h
F=q+a*P^T*q+.5*a^2*(P^T)^2*q
    +grad(a)*dot(u,q).
```

The final term is essential to this identity. It is not implemented by simply transposing the second P or reusing a frozen clock. Parameter derivatives must differentiate it as well. This is a different FFN function class, not an exact reduction of the independently trained old ODEFFN or the complete-model potential readout.

## Why this deserves a prototype check

The learned full-potential ODE gains almost entirely in own-coordinate fitting. A local adjoint could expose derivative-response features while reusing the original conditional encoder, avoiding a second independent parameter set. This is a mechanism hypothesis, not a demonstrated transfer of the whole-field gain. Prior frozen composition diagnostics show that complete adjoint paths matter at the trained whole-field weights, so local replacement must be tested cold rather than justified by those old outputs.

For width d and original controller width H, the old two-solve FFN has twice the encoder parameters, namely `2*(d²+dH+2H+1)`. The proposed map has one such set. Its four P matrix applications and two first-controller matrix applications give nominal dense work `4d²+2dH+H`, versus `4d²+2dH+2H` for the old FFN. The removed last-controller projection reappears as pointwise weighting in the clock derivative. Actual operation counting, pointwise cost, full-model training work and latency must be measured before treating this as a cost advantage.

At d=H=64, ordinary tied ridge controls `W^T SiLU(Wh+b)` with128 and192 features bracket the candidate's parameter and dense-work budgets. They are essential: without them, a gain might belong to a standard tied gradient feature map rather than the ODE encoder. Ordinary SwiGLU128/192 and the unchanged original ODE must also remain in any separately registered cold comparison. Counts/initialization/costs still require an actual prototype; no quality threshold is chosen from a new outcome.

## Prior art and claim boundaries

[Gradient Networks](https://arxiv.org/html/2404.07361v2), especially Sections3/5/6, directly parameterizes gradient fields and discusses tied ridge and cascaded constructions. It is prior art for the generic gradient-network idea, not evidence of an FM/ODELayer gain for this map. [Chao et al., ICML2023](https://proceedings.mlr.press/v202/chao23a.html) supplies an important negative control on the reasoning: tying weights is sufficient in simple constructions but can restrict conservative-field approximation unnecessarily. The proposed construction therefore cannot claim novelty or superiority from its gradient identity alone. Targeted searches on26 October2026 verified these primary methods and the negative tied-weight example; this is not an exhaustive novelty search.

F is a gradient in its **FFN input coordinates**. Its Jacobian is symmetric, but need not be PSD: SiLU is not a monotone activation everywhere, and nonlinear L contributes Hessian terms. Surrounding normalization, FiLM, residual composition, input/output maps do not preserve a whole-model gradient guarantee. There is no proved global stability, invertibility, approximation advantage or generation improvement.

The separate [prototype study](../research-local-adjoint-2026-10-26/task_plan.md) has now checked values, full state/parameter derivatives, parameter reuse and actual costs. Its full model has58899 parameters versus83862 original,106752 versus106944 inference MACs, and317120 versus317696 loss/backward MACs. Single-thread CPU timings were4.34%/7.89% slower for forward/loss-backward, so there is no latency advantage. A separate cold protocol fixes all controls and quality conditions before pipeline smoke. This note alone does not launch training or reopen any stopped branch.
