# Review of sharing the value projection with the controller

## Decision

**This is a defensible single parameter-efficiency experiment, provided it is framed and controlled as compute sharing.** It is not established as a remedy for the observed error gap, and the current results do not predict that it will beat SwiGLU. Its strongest justification is an exact allocation-of-computation result at the paired initialization, not a claim that low tangent effective rank proves that this architecture is needed.

The proposal stays close to the professor's K1 solve, retains the two independent solves and intervening SiLU, and changes one internal dependency. It reuses a value feature already paid for to give the controller a wider feature map at approximately the old parameter budget and exactly the old dense matrix MAC count. The proposed untied width-64 arm is essential: it separates sharing from simply supplying wider controller features. This review does not implement the proposal or run training. It considers only full width-64 sharing, not a partial-row, rank or width sweep.

The completed diagnostic at `controller_tangent_20261024_m1/output/summary.json` has 24 cases. Its final product16 vector pointwise duration floors are numerically zero, whereas scalar floors are about 0.70. Controller/full-FFN mean-diagonal scales are 13.882/445.154 for vector multiplication and 68.777/553.709 for vector addition. Their full-FFN effective ranks are 24.93/33.34, compared with 84.70/88.09 for the two SwiGLUs. These are properties of saved local responses, not architecture lower bounds. Low-ridge correction overfits the small fit-anchor collection for both families. Neither those spectra nor the amplitude counterfactual establish the cause of the trained-model gap.

## 1. Define the proposed family precisely

For one solve, with hidden dimension d=64, define

```text
z = P h
a = SiLU(z+b)
g = Wg a + bg
s = .2 + .8 sigmoid(g/10)
F_shared(h) = h + diag(s) z.
```

There is one cached computation of z. Calling the same projection twice would share parameters but would lose the claimed forward compute saving. Do not detach z before the controller: detachment would change the gradient being tested.

The untied width-64 control is

```text
z = P h
a = SiLU(R h+b)
g = Wg a + bg
F_untied(h) = h + diag(s) z,
```

initialized with R=P by copying values into a distinct trainable parameter. P and R must not alias. Wg, b, bg, P, all other solves and all surrounding tensors are paired exactly. This makes the complete initial functions equal for every input, not merely equal because the final velocity head is zero.

The tied family is a subset of the untied family: enforce R=P in the untied formula. It does not increase expressivity relative to that control. Relative to the existing width-32 controller there is no simple nesting: it gains controller features but loses an independent basis. For example, if P has a nontrivial nullspace, the tied increment is identical at h and h+k for every k in that nullspace. An independent R can distinguish those inputs and modulate the nonzero value feature differently. When P is invertible no information is lost at this first projection, but its conditioning and its basis before SiLU still constrain the finite controller.

The map is also exactly an ordinary residual feature-gated map `h+s(Ph)*Ph`. Calling the same formula a generic gated block would not create an independent control. Its ODE connection is the retained conditional K1 form `h+D(h)Ph`; there is no higher-order accuracy, new physical time interpretation, or automatic flow-matching-specific advantage.

## 2. Exact cost comparison

Count one multiply-add as one dense linear MAC. Bias additions, SiLU and sigmoid work are separate. With d=64 and old controller width m=32:

| One solve | Dense matrices | Matrix parameters | Bias parameters | Dense MACs |
|---|---|---:|---:|---:|
| Existing vector32 | P, W1 of shape 32x64, W2 of shape 64x32 | 8192 | 96 | 8192 |
| Shared64 | P, Wg of shape 64x64 | 8192 | 128 | 8192 |
| Untied64 | P, R, Wg, each 64x64 | 12288 | 128 | 12288 |

Thus shared64 has 8320 parameters per solve versus 8288 for vector32, a difference of 32 biases. Across two solves and three blocks it adds 192 parameters. It has the same dense MACs but twice as many hidden SiLU elements inside each controller. Actual throughput must be measured; “same MACs” is not “same wall time.”

The two-solve FFN counts are:

| FFN | Parameters | Dense MACs |
|---|---:|---:|
| Existing vector32 | 16576 | 16384 |
| Shared64 | 16640 | 16384 |
| Untied64 | 24832 | 24576 |
| SwiGLU nominal96, actual inner64 | 12480 | 12288 |
| SwiGLU nominal128, actual inner85 | 16554 | 16320 |
| SwiGLU nominal192, actual inner128 | 24896 | 24576 |

These use the repository's `int(2*nominal_hidden/3)` convention. SwiGLU192 is therefore the appropriate dense-MAC control for untied64, while the existing SwiGLU128 nearly matches shared64. SwiGLU96 remains the established efficient reference. The “1.5x” multiplier applies to the two-solve FFN's dense matrices; the complete model also contains common computation and parameters and does not scale by exactly 1.5.

## 3. The shared matrix receives two gradients

At one native anchor define

```text
D = diag(s)
Z = diag(z)
C = diag(ds/dg)
E = diag(SiLU'(z+b)).
```

Then the shared solve has

```text
J_h F_shared = I + (D+Z C Wg E) P.
```

For the untied solve,

```text
J_h F_untied = I + D P + Z C Wg E R,
```

where E is evaluated at Rh+b. They agree when R=P. For a downstream output Jacobian G, including the other solve, outer residual, later controllers and final projection, write

```text
U = G D
V = G Z C Wg E
A = G Z C.
```

The shared matrix perturbation gives

```text
delta v_shared = (U+V) delta P h.
```

The untied perturbations give

```text
delta v_untied = U delta P h + V delta R h.
```

Other common parameter responses are

```text
delta v_from_b  = V delta b
delta v_from_Wg = A delta Wg a
delta v_from_bg = A delta bg.
```

The shared P gradient is the sum of value-path and controller-feature-path gradients. Autograd obtains this naturally when the cached z participates in both paths. The separate biases must remain explicit; merging the controller feature bias into the bias-free value projection would change the proposed map.

## 4. Tangent-kernel scope: signed cross terms, not extra directions

At a function-matched tied/untied pair, let Jv and Jg denote the stacked output derivatives for the independent P and R matrices. Let Kcommon contain the common controller-output, bias and other parameter terms. Then

```text
K_shared = Kcommon + (Jv+Jg)(Jv+Jg)^T
K_untied = Kcommon + Jv Jv^T + Jg Jg^T
K_shared-K_untied = Jv Jg^T + Jg Jv^T.
```

For two anchors i,j, the corresponding matrix term is multiplied by `h_i^T h_j`; the b, Wg and bg kernels are common. The cross term can have either sign along a residual direction. In the scalar derivative example Jv=Jg=1, sharing changes the matrix contribution from 2 to 4; for Jv=1,Jg=-1, it changes from 2 to 0. Positive clocks do not force these two gradient contributions to agree.

The shared tangent column space is contained in the untied one. Consequently, at the same matched function and anchor set, sharing cannot lower the unrestricted least-squares residual floor by adding directions. Finite-budget updates, implicit regularization and training paths can still differ. A higher *effective* rank after weighting would not contradict the column-space inclusion.

There is also a useful upper bound:

```text
K_shared <= 2 K_untied                    (PSD order).
```

Indeed, `2K_untied-K_shared=Kcommon+(Jv-Jg)(Jv-Jg)^T`. Equal Euclidean parameter norms do not charge a tied matrix twice for its two uses. If the shared-matrix perturbation is analytically scaled by 1/sqrt(2) to match the duplicated untied-coordinate norm, the untied-minus-shared matrix kernel is `0.5*(Jv-Jg)(Jv-Jg)^T`, which is PSD. This is a clarification of the comparison metric, not a suggestion to add a training rescaling arm.

These results concern local Euclidean parameter derivatives. They do not establish an AdamW learning-rate equivalence or an optimization advantage for this complete model.

## 5. A stronger positive rationale at the declared initialization

Use the same constant-clock initialization as the completed matched study: Wg=0 and bg=0, so s=.6. Then V=0 because it contains Wg. At a nonzero output-head probe:

- the P matrix has its ordinary value-path response in both models;
- the untied private first-feature matrix R has zero response;
- the controller feature bias b also has zero response;
- Wg and bg have identical responses because their features and gates are paired.

Therefore **shared64 and untied64 have the same complete initial tangent kernel while shared64 uses two rather than three dense d-by-d matrix products per solve.** This is an exact initialization statement. It is stronger and more precise than inferring a remedy from the saved models' effective ranks.

Compared with vector32, the same matrix budget reallocates computation from an independent, initially inactive controller feature projection toward a wider controller-output feature map. Once the controller output weights move, the shared P can change its gate features through value-path updates as well as gate-path updates; an untied R only receives the latter. This could improve parameter use or could interfere with a useful value basis. No completed experiment yet distinguishes these outcomes.

The real network initializes its velocity head to zero, initially masking all backbone derivatives. The claims above must therefore be verified with the established identical nonzero probe head; they are not statements that backbone parameters move on the first formal training update. Under the actual zero-head training, the proposed advantage would have to develop after that masking disappears and the gate weights start moving.

At nonzero Wg initialization, the functions can still be paired, but V need not vanish and the initial kernels need not agree. Do not silently switch to that initialization while invoking the equal-initial-kernel result.

## 6. Necessary controls and initialization checks

A minimally interpretable comparison includes the following six arms on the same current targets and backend: existing vector32, shared64, untied64 initialized with R=P, SwiGLU96, SwiGLU128 and SwiGLU192. Keep the original two map returns, K1, middle SiLU, positive sigmoid range, tau, time conditioning and training objective. Do not combine the stopped additive receiver, clock rescaling, operator ordering or another residual change.

The paired requirements are concrete:

1. Copy P, b, Wg and bg exactly between shared64 and untied64, with R a distinct copy of P. Cache Ph once in the shared forward; verify actual matrix operations and parameter identity/counts.
2. Use zero final controller affines in both new arms and in the rerun vector32 parent, matching the current constant-clock setting. Verify equality of the new pair's complete hidden functions and input Jacobians with an identical nonzero probe output head.
3. Check P-gradient addition and the full shared/untied parameter kernels against direct complete-network autograd. At zero Wg their kernels should agree; at a nonzero test Wg the signed cross-term identity should hold.
4. Pair all compatible non-FFN tensors across **every** arm, including FiLM, norms, time embeddings and the input/output layers. Architecture-dependent constructor RNG consumption must not silently change these. Preserve the unchanged baseline trajectories against their declared predecessor where applicable.
5. Keep data-generation states, optimizer convention, actual steps, validation panels and solver fixed. Report complete-model counts, matrix MACs and actual wall time separately. Additional untied parameters and the larger SwiGLU budget are visible costs, not a free control.

The existing controller-tangent implementation assumes that P and the controller's first matrix are independent. It must explicitly reject the new shared-feature layer until a sharing-aware formula is implemented and tested. A hidden reuse of P inside one module is not caught merely by checking for duplicate parameters across modules. The prototype therefore needs an explicit compatibility guard; reusing the old diagnostic unchanged would omit the signed cross terms.

SwiGLU is the necessary generic practical control. A separate block with exactly the same shared gated formula under a different name is not informative. If the eventual claim is specifically about flow matching rather than general parameter efficiency, success here would still require a later objective-relevant explanation; it cannot be inferred from Euler notation.

## 7. Interpret outcomes without rescuing the axis

- **Shared64 improves over vector32 and matches or improves untied64 at lower cost, while passing the existing SwiGLU practical screen:** this supports the resource-sharing design as a candidate. It still requires the planned subsequent independent-holdout and budget confirmation; it does not prove that continuous-flow geometry caused the gain.
- **Untied64 improves but shared64 does not:** wider independent controller features may help, while coupling them to P is harmful. That does not support the shared design. Untied64 must also pass its larger SwiGLU192 budget control before being considered useful.
- **Both new arms improve over vector32 but remain behind their SwiGLU controls:** the practical architecture goal is still unmet. Do not extend the setting through another width, temperature or order sweep.
- **Shared64 improves without a uniform kernel or spectrum improvement:** this is not inconsistent with the signed-cross-term analysis. Avoid retrospectively choosing whichever local statistic favors the result.
- **Neither improves:** stop this sharing axis. The exact initialization economy remains true but was not sufficient to produce a useful trained block.

The test is justified as a bounded question about where the original ODELayer spends its projection budget. It should not be sold as a mathematical prediction of success or as an established fix for controller rank, amplitude, spectral stability or the FM objective. The existing failures make those distinctions particularly important.
