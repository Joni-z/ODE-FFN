# Posterior-response basis: what a plain sigmoid GLU would test

**Conditional GO as a generic basis control, not as a new ODE method.** The question is whether removing the gate preactivation factor from SwiGLU helps fit the affine-plus-signed-offset response. It is not a claim that ambiguous mode boundaries explain the observed gap. A sigmoid GLU is an appropriate comparator for any future claim that an ODE construction supplies this basis. The derivation does not establish that the current wide SwiGLU lacks it, that a hidden basis is the measured bottleneck, or that this control will improve quality. No training is selected by this review.

The new [registered regional readout](posterior-velocity-regions-readout.md) sharpens the motivation: low-entropy coordinates contribute78.26% of the saved Product16 ODE–SwiGLU128 velocity gap, versus1.51% for the highest-entropy bin. At t=.7, confident coordinates contribute93.71% of the same-time gap. In that region the oracle needs an affine slope plus a nearly constant signed mode offset. These observations reject an ambiguous-boundary-only story. They cannot distinguish wrong slopes, offsets, time dependence or cross-coordinate mixing, so they do not by themselves select GLU or an ODE design.

## Exact balanced two-Gaussian velocity

Let Y=S*mu+sigma*E, where S is equally likely +/-1, E and source Z are independent standard Gaussians, a=1-t, and x=aZ+tY. For fixed 0<t<1, write

```text
b=a^2+t^2*sigma^2
P(S=+1|x)=sigmoid(2*t*mu*x/b)
E[S|x]=tanh(t*mu*x/b)
E[Y|x]=(t*sigma^2/b)*x+(a^2*mu/b)*tanh(t*mu*x/b)
v*(x,t)=L*x+C*tanh(k*x)
L=(t*sigma^2-a)/b, C=a*mu/b, k=t*mu/b.
```

The nonlinear term is the posterior responsibility response, not a periodic feature. Its derivative is `C*k*sech(k*x)^2`: the derivative correction is largest around ambiguous mode assignment and small in confidently assigned regions. Its amplitude, transition width and the linear term have different time dependence. A large late-time derivative near a mode boundary need not imply a large average error because little probability mass may remain there.

For the rotated product, let q=R^T*x with orthogonal R. Independence of the isotropic Gaussian source survives rotation. Each q_i has the same scalar calculation with its own mu_i,sigma_i, so

```text
v*(x,t)=R*[L_i*q_i+C_i*tanh(k_i*q_i)]_i.
```

The actual [Product16 definition](../../synthetic/data.py) uses mu_i geometrically spaced from .75 to2.5, all sigma_i^2=.04, and a fixed rotation. Thus it has16 independent nonlinear responsibility responses in the rotated coordinates. The network must learn that coordinate structure; the rotation and coefficients are not supplied to training.

Ring8 is an important counterexample to overgeneralizing this formula. Its equal isotropic component variances give

```text
v*(x,t)=L*x+(a/b)*sum_j responsibility_j(x,t)*mu_j
responsibility_j=softmax_j(t*mu_j dot x/b)
```

because its eight means have equal norm and weight. This is a coupled eight-way normalization, not eight independent affine-logit sigmoids. The simple two-Gaussian GLU construction below does not prove exact Ring8 representation. The single-Gaussian control has an affine velocity and no responsibility-selection term.

## A constructive fixed-time GLU representation

Consider the ordinary biased sigmoid GLU

```text
G(x)=U*[sigmoid(A*x+b_gate)*(V*x+c_value)]+d_out.
```

At a fixed t, a raw-input GLU with2d channels represents the displayed rotated-product population velocity exactly. Let r_i be column i of R:

| Channels | Gate | Value | Output column |
|---|---|---|---|
| i=1,...,d | sigmoid(0)=1/2 | r_i^T*x | 2*L_i*r_i |
| d+i | sigmoid(2*k_i*r_i^T*x) | 1 | 2*C_i*r_i |

Set `d_out=-sum_i C_i*r_i`. The first group supplies the linear term; the second uses `tanh(z)=2*sigmoid(2z)-1`. Product16 therefore needs32 channels in this existence construction. The value bias is essential to these constant-value branches. These are proof weights, **not proposed initialization, teacher weights or oracle supervision**.

This is a fixed-time, direct raw-input construction. It is not an exact full-model capacity theorem for the existing LayerNorm/FiLM/residual backbone. A fixed affine map of concatenated (x,t) also does not automatically generate k_i(t)*q_i and the other nonlinear time coefficients; their representation must be learned by the existing time-conditioning path. The nominal128 gated FFN has85 actual channels, but counting32 proof channels does not establish its learned hidden representation or optimization efficiency.

The training-law distinction also matters. The formula describes the **population evaluation oracle**. Original training draws Y from a cached100000-sample bank. Its exact conditional mean is a normalized sum over those joint atoms, not a factorized product of tanh functions. The earlier bank/population diagnosis found material late-time differences. No label, data law or sampler is changed here; a population-aligned basis would be an inductive-bias hypothesis under the existing bank objective, not an exact match to its conditional teacher.

## What can and cannot be said about SwiGLU

SwiGLU uses `SiLU(a)=a*sigmoid(a)` on its gate branch. The extra a changes the basis. For one nonlinear branch, the distinction is exact: a sigmoid branch with constant value and an affine bypass represents `L*x+C*tanh(k*x)` directly. A single SwiGLU unit with affine value plus any affine bypass cannot represent a bounded nonconstant tanh globally.

To verify the latter, write its nonlinear branch as `(u*x+v)*SiLU(k*x+b)`, absorbing output scaling into u,v. If k>0, the branch vanishes as x->-infinity, so bounded tanh forces the bypass slope to zero. At +infinity the branch approaches the polynomial `(u*x+v)*(k*x+b)`. A finite nonzero difference between the two limiting constants would require its quadratic and linear coefficients to vanish while leaving a nonzero constant; with k!=0 those conditions force u=v=0. For k<0 the tails exchange roles; k=0 gives only an affine function.

This is a **one-unit separation**, not a width or efficiency lower bound. Multiple SwiGLU units can cancel growing terms, use different shifts/scales and approximate nonlinear responses. The existing deep normalized network is far outside that restricted class. Do not conclude that SwiGLU cannot learn responsibility functions, or that its preactivation factor explains the observed ODE gap.

The original vector K1 layer already contains a sigmoid product:

```text
h+D(h)Ph = h+.2Ph+.8*sigmoid(g(h)/10)*Ph.
```

It differs from a free GLU through its coupled receiving projection, fixed hidden width, residual map, deeper clock and two-map/SiLU composition. Moreover, FiLM shifts and upstream features can create effective constant components, so its bias-free P does not prove that the complete network lacks constant-value branches. A plain GLU win would motivate investigating these structural restrictions; it would not diagnose one of them by itself.

## Why this is a new control, but not an ODE rescue

Stage20 tested bounded positive phi1-style corrections multiplying the existing SiLU gate and stopped that family. Replacing `a*sigmoid(a)` by `sigmoid(a)` is not another member of that family: their quotient is1/a, which is singular at zero and changes sign. A plain sigmoid gate is a known generic basis, not a continuation or claimed improvement of the stopped forced-ODE method.

This also limits interpretation of a positive result. It would establish that a simpler generic gate helps under these settings. Any subsequent ODE claim about posterior-shaped responses would have to beat that new control as well as SwiGLU. It would not reopen the stopped phi1, diagonal-flow or state-clock families, and it would not complete the architecture goal.

## Smallest controlled study eligible for separate selection

Use Gaussian2D, Ring8 and Product16 with three arms: established SwiGLU96, established SwiGLU128 and plain GLU128. That is nine cells, all seed0, with the original bank-pair velocity loss, time sampler, data, head, optimizer and budgets. Change only the gate activation in the GLU arm; keep the same two-output input projection, value bias, output projection and actual85-channel convention as SwiGLU128. This gives identical parameter count and dense MACs. Preserve all raw initialization tensors and RNG draws, and retain the shared non-FFN reference initialization. No new slope, learned activation coefficient or width is needed.

The hidden initial functions necessarily differ at native initialization. The common zero final velocity head makes initial complete outputs zero, but the output-head parameter gradients generally differ because the hidden features differ. Do not claim function/NTK pairing, and do not zero or shift gate projections to manufacture it. Use a nonzero-head probe to make the difference visible. Both SwiGLU controls must retain their historical default behavior.

The informative result is the absolute and relative quality/cost of this generic basis, especially whether Product16's middle-time field/Jacobian discrepancy decreases without a nonlinear-task generation regression. Any mechanistic readout should retain all predeclared times and responsibility regions, rather than selecting examples after outcomes. A gain on Gaussian alone would not support the mode-assignment explanation; a Product16-only gain would still face the coupled Ring8 control.

Before selecting training, independent review must accept this narrow question and a fixed practical screen. If selected and negative, stop this activation setting without gate-slope, temperature, extra-bias initialization or wider-feature rescue. If positive, report it as a stronger generic baseline and use its evidence to formulate the next ODE-specific structural question. No oracle labels, pretrained analytic weights or new objective are authorized by this review.
