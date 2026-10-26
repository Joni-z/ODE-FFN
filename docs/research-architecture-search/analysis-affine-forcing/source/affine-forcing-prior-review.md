# Affine forcing in the original ODELayer: a bias-completeness question

Read-only review, 2026-10-25. Stage22 has been launched; its outcomes are not used here. **Conditional GO for one bounded bias-completeness falsification; NO claim of a new ODE mechanism or selection of implementation/training.** Adding a missing value bias is a distinct, inexpensive axis in the original block. The evidence supports asking whether that omission matters, not expecting it to repair the existing gap. Finish and audit the generic GLU comparison before deciding whether to run this control.

## What has actually been omitted

Current `synthetic/models.py:63` constructs `proj=Linear(width,width,bias=False)`. Its additive-generator branch is `h+cPh+(D-cI)h`; it adds no independent vector. The map/increment and identity-location variants change only the identity terms or shift P by `I/c` at paired initialization. Their production recurrence still applies a bias-free P. Thus these experiments did not test `Ph+b`. Controller biases and the outer FiLM/input/output biases already exist; the complete trained model is not bias-free or forced to vanish at the raw data origin.

Keep both original K1 solves and the intervening SiLU. For each solve compare

```text
homogeneous: F_hom(h) = h + D(h)Ph
affine:      F_aff(h) = h + D(h)(Ph+b)
static bias: F_static(h) = h + D(h)Ph + c*b,       c=.6.

F_aff(h)-F_static(h) = (D(h)-cI)b.
```

The controlled difference is a state-dependent bias. At `b=0,D=cI`, the two enlarged models share their functions and all parameter Jacobians, including through a nonzero receiving head: the new b-response is `cI` and all old responses agree. Their first difference around this point is a mixed controller–bias term, not an initially larger kernel. The homogeneous parent has fewer parameter columns, so its full kernel need not equal theirs. This pairing requires constant-clock initialization; zero b alone does not suffice with varying clocks.

Each enlarged FFN adds `2d` parameters and no dense MACs. With the current Product16 width64/depth3 setup, the parent has 83,664 parameters/81,984 dense MACs; either addition gives 84,048 parameters. SwiGLU128 and GLU128 have 83,598 parameters/81,792 MACs. These are close budgets, not exactly identical budgets or elapsed times.

## The exact ODE interpretation and coordinate caveat

For fixed input h, freeze D(h). The affine field is `z'=A*z+q`, where `A=DP,q=Db`. Homogeneous augmentation gives `[[A,q],[0,0]]`. Its K1 map is the proposal; its K2 map would be `h+D(Ph+b)+.5*DPD(Ph+b)`. The latter is mathematical context only, not a proposed next arm.

If `Pr=b` has a solution, as it does for invertible P, every Taylor order obeys

```text
F_aff,K(h) = T_K(D(h)P)*(h+r)-r.
```

Thus a single affine solve is a translated homogeneous solve with controller `D_tilde(z)=D(z-r)`. The existing controller MLP represents this shift exactly by changing its first bias. This is a learned origin, not intrinsically new orbit geometry. If b lies outside the range of a singular P, that translation is unavailable.

This does not establish whole-backbone equivalence. With translations `T_r(h)=h+r`, the two-solve block becomes

```text
T_-r2 H2 T_r2 SiLU T_-r1 H1 T_r1.
```

The internal shifts around SiLU cannot simply be discarded. FiLM can translate the first solve input, but that alone contributes `r+DPr`, rather than the required `Db`; a fixed r does not generally match varying D. The outer residual and LayerNorm also prevent freely moving arbitrary channel translations through the network. Conversely, embeddings, FiLM, controller biases and depth may already synthesize equivalent features. Neither an isolated `F_aff(0)!=0` witness nor failure of this simple reparameterization proves a full-network capacity increase.

The augmented spectrum is `spectrum(DP) union {0}`, independent of b. This gives no forcing-norm bound: P=0 permits arbitrarily large translation. Moreover, the complete input Jacobian gains `diag(b)J_s`, which is absent from frozen-generator spectra. No stability or FM-time guarantee follows.

## Prior art and evidence boundary

This is familiar gated-bias algebra. Shazeer's [GLU Eq.4](https://arxiv.org/html/2002.05202v1#S2) explicitly includes a bias in the value projection. Here `Db=.2b+.8*sigmoid(g(h))*b`: a constant plus gated constant-value features. The two-layer gate, residual structure and composition differ from ordinary GLU, but naming this term affine forcing does not make those features novel.

[FiLM, Section2.1 Eq.2](https://arxiv.org/html/1709.07871v2#S2.SS1) already defines conditional scale plus shift and allows conditioning on the same input. Our outer modulation is time-conditioned, whereas the proposed bias depends on hidden state through D. That distinguishes their placement and conditioning, not the general idea of conditional bias. These are exact relevant precedents; no unrelated recent architecture is needed to justify the algebra.

The population mixture's affine slope plus signed mode offset motivates testing offset features. The registered confident-region excess does **not** identify a missing bias, and its centered population velocity is odd; an uncontrolled constant offset can hurt. Hidden coordinates are learned, so raw-mixture formulas do not identify a hidden forcing vector or justify oracle initialization.

## What one admissible test could establish

If selected later, retain homogeneous vector K1, affine forcing, identically initialized static bias, both established SwiGLU budgets and the known GLU128 control. Preserve objective, paired backbone, clocks, widths and training budget. Adaptive forcing must beat the static-bias control and the practical generic baselines under predefined cross-task/Jacobian/sampling guards. If static bias explains the gain, the result is bias completion; if generic baselines still win, stop. Record actual conditional correction and complete derivatives, without treating a nonzero b as causal proof. Do not follow failure with K2, clock, initialization, width or head rescue. This tests one previously omitted component; it does not reopen the stopped Stage20 gate-response or Stage21 recurrence families.

## Reassessment after Stage22 terminal metrics

Appended 2026-10-25, after the initial review above. The root agent independently checked these terminal summaries; the complete Stage22 audit was still pending at the time of this reassessment. I also read the two terminal `results.json` files directly. The values below preserve their full stored decimal precision for primary validation oracle MSE at step5000; they are not a replacement for checkpoint replay or the completion audit.

| Task | GLU128 | SwiGLU128 |
|---|---:|---:|
| Gaussian2D | 0.0037171965837478637 | 0.0039272540807724 |
| Ring8 | 0.027148163986206054 | 0.015296290397644043 |
| Product16 | 0.15806039810180664 | 0.09232576999664306 |

Sources: [Gaussian2D/Ring8 terminal results](/Users/mr.z/ode-workspace-prep/architecture-search/sigmoid_glu_20261025_m1/output/sigmoid_glu_v1_20261025_m1/results.json) and [Product16 terminal results](/Users/mr.z/ode-workspace-prep/architecture-search/sigmoid_glu_20261025_m1/output/sigmoid_glu_product16_20261025_m1/results.json). The same summaries also show worse GLU Jacobian error and SW2 on both nonlinear tasks. The Gaussian improvement does not satisfy the registered nonlinear-task criterion.

This materially weakens the posterior/sigmoid-basis motivation. It supplies no evidence that missing sigmoid features or a missing signed-offset basis caused the original ODE gap. The remaining **conditional GO is only for one original-bias completeness falsification**: `Ph` omitted an independent value bias while the practical GLU/SwiGLU projections have one, and that omission has never been directly controlled. The full ODE backbone already has other biases; neither its capacity nor the expected sign of this change follows from the omission. Keep the failed GLU visible in the six fixed controls rather than removing it after outcomes.

A static-bias lead could be a useful practical correction to the original block, labeled bias completion. A conditional-forcing interpretation additionally requires improvement over the paired static-bias control and strong generic baselines. This remains distinct from the stopped phi1 and recurrent-clock families, but a follow-up clock/order/width/initialization/forcing-strength rescue after failure would turn it into the unbounded search the review rejects. No implementation or formal training was selected by this reassessment, and no frozen Stage22 document was changed.
