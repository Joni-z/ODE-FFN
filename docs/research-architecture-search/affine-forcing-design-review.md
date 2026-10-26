# Affine forcing: one omitted-component falsification

**Conditional GO for at most one bounded bias-completeness comparison, pending the completed Stage22 audit and a separate selection decision.** The question is whether the original bias-free receiving projection omitted a useful component. The existing results do not identify that omission as the cause of the ODE gap. This is familiar conditional-bias algebra, not a newly established ODE mechanism. A useful static-bias result should be retained as bias completion rather than discarded for lacking that mechanism. This review authorizes no implementation or training.

The [prior review](affine-forcing-prior-review.md) establishes that earlier additive-generator, identity-placement and clock experiments did not insert an independent forcing vector. The [posterior-basis review](posterior-basis-design-review.md) explains why population mixture velocities have signed offsets, but neither that formula nor the confident-region error breakdown proves that a hidden bias is missing. Plain GLU already supplies biased value features; its completed result must enter the practical comparison, whichever way that result goes.

## Exact local comparison

Keep vector K1, two original solves with their middle SiLU, width64 and gate-hidden32. Write `s(h)=.2+.8*sigmoid(g(h)/10)`, `D=diag(s)`, and `c=.6`. One solve is

```text
H(h) = h + D(h)Ph                         homogeneous parent
A(h) = h + D(h)(Ph+b)                     affine forcing
B(h) = h + D(h)Ph + c*b                   matched static bias
A(h)-B(h) = (D(h)-cI)b.
```

Both new arms learn one independent64-vector b per solve, initialized to zero. P and the existing controller stay independent. Adding b to the frozen internal field `z'=DPz+Db` gives A as its first Euler/Taylor step. This is a change to the professor's homogeneous field, not a correction of a numerical error. Higher orders, state-clock changes and alternate forcing placements are outside this test.

For the same input and shared parameter values, let `J_s=ds/dh`. Exact input derivatives are

```text
J_H = J_B = I + DP + diag(Ph) J_s
J_A = I + DP + diag(Ph+b) J_s
J_A-J_B = diag(b) J_s.
```

Here `J_B=J_H` is a **same-input local identity**. Two fully composed networks generally feed different inputs to later layers. It is not an equality of their complete velocity Jacobians.

For a receiving loss derivative r at this layer, the parameter differentials are

```text
dA/db = D,                     dB/db = cI
dA/dP[dP] = dB/dP[dP] = D*dP*h
dA/dtheta[dtheta] = diag(Ph+b)*J_s,theta*dtheta
dB/dtheta[dtheta] = diag(Ph)*J_s,theta*dtheta

grad_b L_A = sum_samples D(h)^T*r(h)
grad_b L_B = c*sum_samples r(h).
```

Thus a zero b alone does not pair the new parameter columns if the clocks vary. Require the same constant-clock initialization in all three ODE arms: zero the controller's final affine map, so `D(h)=cI` for every h, and set b=0 without consuming RNG. Then A and B have identical functions, input derivatives and **all** parameter Jacobians. Their complete-network Jacobians and initial Euclidean parameter kernels also agree, including with a shared nonzero output head. Their common old-parameter responses equal H's, but H has no b columns and hence is not a fully kernel-matched comparator. Adding those columns can also affect gradient clipping once their gradients are nonzero.

Near this paired point,

```text
A-B = (D-cI)b
    = diag(J_s,theta * delta_theta)*delta_b + higher-order terms.
```

The first difference is a controller-bias interaction. At b=0 after clocks have moved, the two bias gradients can already differ, while their native outputs and old-parameter responses still agree at matched parameters. The controller's first affine weights initially have zero response because its final weights are zero; the final controller affine parameters need not. A formal zero velocity head hides all backbone responses initially, so it cannot replace the nonzero-head pairing check.

This constant initialization is already used by the established Stage17 `vector_parent`; it is not a proposed new rescue. Stage21 Euler is an algebraic reference with that same initialization. Reuse its evidence only under its recorded state/config/RNG checks, and preserve the reported small finite-arithmetic trajectory differences rather than asserting bit equality from algebra alone. An unrelated native random-clock parent would not be the correct causal control.

## Translation, existing biases and symmetry

If `Pr=b` has a solution, define `z=h+r` and shift the controller to `D_tilde(z)=D(z-r)`. Then

```text
A(h) = H_tilde(h+r)-r,
H_tilde(z) = z+D_tilde(z)Pz.
```

The two-linear controller absorbs the input translation exactly in its first bias. Invertible P always permits this; singular P with b outside its range does not. The same conjugacy holds for every frozen-clock Taylor polynomial or exact affine solve. Affine forcing therefore does not automatically create new internal orbit geometry.

For the two-solve FFN the conjugacy becomes

```text
T_-r2 H2 T_r2 SiLU T_-r1 H1 T_r1,   T_r(h)=h+r.
```

Translations appear on both sides of the middle SiLU. One outer FiLM input shift does not remove them. In the actual backbone, LayerNorm, time-conditioned FiLM shifts, controller biases, embeddings and depth already provide substantial offset flexibility. They may make the intervention redundant in practice. Conversely, a proposed single-layer translation does not establish exact equivalence of the complete backbone: arbitrary channel translations do not commute through LayerNorm, SiLU or the outer residual. The isolated homogeneous FFN maps zero to zero while its biased version need not, but FiLM already shifts its input, so that witness proves no complete-network separation.

Ring8 and Product16 are centered symmetric **populations**, so their optimal population velocity is odd. A nonzero global constant drift is not the missing target feature. The proposed branch decomposes as

```text
D(h)b = c*b + .4*tanh(g(h)/20)*b.
```

The second term can supply signed variation while the first creates an offset that other parameters may cancel. Learned hidden coordinates need not be centered or odd, and the finite cached target bank is not exactly symmetric. Neither population symmetry nor a large learned b identifies success or failure. The noncentered Gaussian control checks ordinary affine fitting; improvement there alone is insufficient.

The augmented generator `[[DP,Db],[0,0]]` has eigenvalues `spectrum(DP) union {0}`, independent of b. That observation bounds neither forcing magnitude nor the full input derivative above. No stability, invertibility or flow-matching-specific guarantee follows.

## Smallest comparison and practical decision

The three mathematical arms are H, B and A. Keep the same six original P/controller pairs and all backbone tensors, clocks, data, uniform-time pair-velocity loss, head, optimizer and seed0. Allocation of zero b must consume no random draws. Preserve the existing Stage15 hash-bound non-FFN initialization and all training/sampling settings. Do not insert an additional output projection, analytic initialization or teacher information.

Parent and generic counts below were checked with the current native constructors and MAC counter; the unimplemented bias-arm rows add the derived384 parameters:

| Model | 2D parameters | 2D dense MACs | Product16 parameters | Product16 dense MACs |
|---|---:|---:|---:|---:|
| Homogeneous vector K1 | 81,858 | 80,192 | 83,664 | 81,984 |
| Either new bias arm | 82,242 | 80,192 | 84,048 | 81,984 |
| SwiGLU96 | 69,570 | 67,904 | 71,376 | 69,696 |
| SwiGLU128 / GLU128 | 81,792 | 80,000 | 83,598 | 81,792 |

Each new arm adds `depth*2*width=384` parameters and no dense matrix MACs. Elementwise work and actual latency still differ. These are close budgets, not exactly matched budgets.

Run the three ODE arms on the existing three tasks: nine ODE cells. Both established SwiGLUs and GLU128 remain mandatory practical comparators. Their already audited trajectories can be hash-bound references if data, objective, backbone initialization, optimizer and sampling contracts are unchanged; a rerun is a compatibility check, not independent evidence. If that reuse cannot be verified, rerun the required controls rather than altering their initialization. The minimal scientific comparison has six model settings; not every immutable reference needs another training execution.

Before any launch, register two separate decisions. Let G be the better of SwiGLU96, SwiGLU128 and GLU128 for the relevant task and metric.

1. **Bias-completion lead:** either A or B must improve primary oracle MSE by at least10% versus H and versus G on the same nonlinear task. On all other tasks its primary MSE must be within5% of G; on both nonlinear tasks its Jacobian error and validation SW2 must each be within10% of the best generic value. A static-bias lead is a useful improvement of the original block, with no conditional-forcing claim.
2. **Conditional-bias evidence:** A must pass that practical screen, beat B by at least5% primary MSE on the same qualifying nonlinear task, and have no primary regression over5% against B on any task. This distinguishes the trained placement comparison from generic extra bias. It still does not establish ODE uniqueness or a posterior-probability mechanism.

Use the entire fixed-time and task readout, not selected examples or entropy bins. A lead would still require independent evaluation and timing before an architecture claim. If neither bias arm passes the practical screen, stop this setting. If only ordinary bias completion passes, retain that result under its actual interpretation; do not tune A until its conditional story wins. No order, clock range, temperature, initialization, width, head or loss rescue follows failure.

## Measurements that identify the added term

At each existing saved checkpoint and standard fixed-time anchors, record each layer's b, `D*b`, `c*b`, and `delta=(D-cI)b`, with absolute RMS and ratios to input and native increment. Also report the mean and centered RMS of delta across anchors. A nonzero mean can be a constant offset; only its centered component demonstrates state variation on that panel. Record `||diag(b)J_s||_F` analytically alongside native complete-field Jacobian errors. It is a hidden local sensitivity, not a full-model stability certificate.

Prespecify one final-checkpoint intervention: replace every `D*b` by `c*b`, keeping P, all controllers, b and other weights fixed. Recompute the whole downstream network and its input derivative on all standard anchors. Report the velocity/output difference and change in oracle field/Jacobian errors. This measures dependence on the added conditional correction at that trained checkpoint; it is not a lower bound on a retrained static model and cannot be inferred by summing local corrections.

The c-replacement also removes delta's mean. If state variation needs to be separated from that mean, a clearly descriptive supplementary probe can replace `D*b` by each layer's native panel mean `mean(D)*b`, with those means frozen before the intervention. This is an evaluation-panel-dependent constant, not the trained static arm or a new train-time center. Recompute downstream gates and label the use of panel activations. Neither intervention supplies oracle labels or changes training, and neither determines the practical screen after outcomes.

These checks can show whether the added term is present, affects the learned function and accompanies a controlled improvement. They cannot establish that missing forcing caused the old gap, or turn a differential-equation rewriting of a gated bias into a novel mechanism.
