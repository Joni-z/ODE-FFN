# Complete-network controller responses after the additive-generator study

This note concerns the completed K1 multiplicative/additive comparison, not a new training run. The frozen study is `additive_generator_20261024_m1`; its [readout](analysis-additive-generator/analysis-report.md) rejects both candidates under the registered practical criterion. The vector additive arm improves product16 relative to its parent but remains behind SwiGLU and worsens Ring8. A scalar local-map witness cannot explain that pattern by itself.

The useful next measurement is a **shared-controller tangent fit on the saved complete networks**, with a separate unrestricted gate-value bound. This distinguishes available instantaneous directions from directions that the finite, shared controller can produce consistently across inputs. It does not establish retrainability or select a new architecture.

All formulas below use column vectors, fixed external generation time t, and a fixed trained parameter state. The current study has independent controllers in the two solves and across the three blocks. There are six controllers in total.

## 1. Direct gate-value responses of one solve

Write c=.6, D=diag(s), and let h be the native input of a solve. The two K1 maps are

```text
F_mul(h) = h + D(h) P h
F_add(h) = h + c P h + (D(h)-cI) h.
```

For a vector controller, at fixed h,

```text
B_mul = partial F_mul / partial s = diag(P h)
B_add = partial F_add / partial s = diag(h).
```

For a scalar controller these are respectively the single columns `P h` and `h`. Define the receiving feature q to be Ph or h. B is a diagonal matrix for vector gates and a column for scalar gates.

Let `S_h=partial s/partial h`. The complete input Jacobians are

```text
J_mul = I + D P + B_mul S_h
J_add = I + cP + D-cI + B_add S_h.
```

The last term is essential. A response computed by freezing every subsequent gate value is generally not the actual response of this trained network to a controller change.

At the paired initialization, the final controller affine is zero, so s=c and S_h=0. Both maps and their input Jacobians agree. Their direct gate responses need not agree: one receives Ph, the other h. The complete network's zero output head initially makes all output-level controller responses zero, so normalized ratios at step zero are undefined and should be reported as such.

## 2. Propagate through both solves, the outer residual and later blocks

For a block write

```text
z = time_modulation(LayerNorm(h_block), t)
y1 = F1(z)
u = SiLU(y1)
y2 = F2(u)
h_next = h_block + alpha*y2,    alpha=1/sqrt(depth).
```

Let `D_phi=diag(SiLU'(y1))`, let J2 be the **native full** input Jacobian of the second solve, and let `T=partial v/partial h_next` include all later blocks, their native controllers, the final normalization and output projection. Direct perturbations of the two controllers produce

```text
R1 = partial v/partial s1 = alpha*T*J2*D_phi*B1
R2 = partial v/partial s2 = alpha*T*B2.
```

Equivalently, for any solve l,

```text
R_l = G_l B_l,
G_l = partial v / partial y_l,
```

where G_l includes native downstream recomputation. This defines the intervention precisely: add an independent small perturbation to one controller's duration output at its node; all downstream controllers remain functions of their perturbed inputs. Simultaneous independent node perturbations add to first order.

Replacing **all** durations by unrelated frozen values instead would produce different downstream Jacobians. Such an open-loop intervention may be studied separately, but it is not the controller parameter tangent used here.

## 3. A crucial scalar/vector distinction

For a scalar gate, the output response is the single vector `G q`. Replacing Ph by h can rotate that vector. With six scalar controllers, arbitrary independent gate-value changes at one input span at most six output directions. This leaves a potential local controller-only restriction in product16, whose output has dimension 16. The same bound is not necessarily restrictive in 2D.

For a vector gate,

```text
R = G diag(q).
```

If every q coordinate is nonzero, multiplication by diag(q) is invertible, hence

```text
column_space(R) = column_space(G).
```

Therefore, **holding the same downstream G fixed, vector Ph and vector h receivers have exactly the same unrestricted pointwise response span whenever both receiving vectors have no zero entries**. They rescale and possibly reverse individual response columns; they do not create new column directions. Zero or numerically tiny coordinates require explicit treatment.

Likewise, if each nonzero output response column is normalized to unit length, its outer product is independent of the sign and magnitude of q. The corresponding normalized pointwise Gram matrices for Ph/h coincide under a common G. This is an exact sanity check, not an empirical hypothesis.

Actual trained mul/add networks have different activations, controller functions and downstream G. Their measured output responses may differ substantially. That difference cannot be attributed solely to a new receiving direction in the additive formula. It may reflect state-dependent amplitudes, finite gate range, shared-controller accessibility or learned downstream representations.

The pointwise span result also does not make the two model families equivalent. Matching their corrections would generally require input-dependent gate ratios such as `(Ph)_j/h_j`, which can be singular, outside the permitted range, or unavailable to the finite controller. Pointwise independent controls are a substantial relaxation of the implemented shared MLP.

## 4. From gate-value controls to the actual finite controller

For one controller define

```text
u = W1 h + b1
a = SiLU(u)
g = W2 a + b2
s = shift + scale*sigmoid(g/tau)
C = diag(ds/dg)
A = R C
T = A W2 diag(SiLU'(u)).
```

For a scalar controller C is scalar. The controller parameter differential, with h fixed at the local node, is

```text
delta v
 = A (delta W2*a + delta b2)
 + T (delta W1*h + delta b1).
```

This contains the full controller parameter response. Dependence of later controller inputs on this controller is already inside R. In the current feedforward network, this controller's own input does not depend on its own independent parameter block, so no extra own-input chain term is missing.

Using augmented vectors `a_bar=[a;1]`, `h_bar=[h;1]`, the parameter Jacobian entries are

```text
J_last[output, gate, feature] = A[output, gate] * a_bar[feature]
J_first[output, hidden, input] = T[output, hidden] * h_bar[input].
```

At two different anchors i,j, the exact controller tangent-kernel block is

```text
K_ij^layer
 = (1+a_i^T a_j) A_i A_j^T
 + (1+h_i^T h_j) T_i T_j^T.
```

Sum this expression over the six independent controllers. This retains cross-input and cross-time couplings without storing the whole parameter Jacobian. The kernel is symmetric positive semidefinite when all anchor blocks are assembled consistently. The full controller kernel includes first and last affine parameters; a last-affine-only kernel is a smaller, explicitly labeled tangent space.

Parameter sharing changes the formula. If two solves share a controller, their parameter Jacobians must first be added before forming J J^T, introducing cross-solve terms. The current completed study uses independent controllers, so simply summing their kernels is correct here.

At finite logits the sigmoid slope is positive but can be tiny. Thus gate-value and logit pointwise spans can agree while their conditioning and accessible response under finite parameter changes differ. Saturation and actual parameter coordinates matter even when a pointwise rank test finds no missing directions.

## 5. Three different minimum residuals

Let `e_i=v*(x_i,t_i)-v_theta(x_i,t_i)` be the current oracle velocity residual. These are field residuals, not generated endpoint errors.

### A. Independent duration controls at each anchor

Concatenate the six R_l at anchor i to obtain M_i. If duration changes are arbitrary, independent across anchors, and unbounded, the least-squares floor is

```text
min_delta ||e_i-M_i delta||^2
  = ||(I-M_i M_i^+) e_i||^2.
```

This is an optimistic **local control bound**. A full row-rank M_i makes it zero regardless of whether a trained shared MLP could supply the needed control. Report numerical rank, singular values and the minimum-norm coefficient size; a tiny singular direction may need an enormous gate change. An unbounded projection alone is especially weak evidence for the vector model.

### B. Shared controller parameter changes across all fit anchors

Stack e_i and the exact controller parameter Jacobians to form e and J. One parameter change must serve every anchor:

```text
min_delta_theta ||e-J delta_theta||^2
  = ||(I-J J^+)e||^2.
```

Every shared-controller perturbation induces a particular collection of duration changes, so this minimum cannot be lower than the sum of the pointwise unrestricted minima. A gap identifies inconsistency or insufficiency of the finite controller tangent space on that anchor collection. It does not prove a global approximation barrier: P, earlier features, the output head and finite nonlinear controller changes are fixed by this diagnostic.

### C. Finite-budget parameter changes

A ridge diagnostic uses

```text
delta_theta_lambda = J^T (J J^T + lambda I)^-1 e
predicted correction at evaluation anchors
  = K_eval,fit (K_fit,fit + lambda I)^-1 e_fit.
```

Positive lambda restricts the use of small singular directions in the declared Euclidean parameter metric. It is not automatically a small nonlinear perturbation. Record the implied correction and, when available, parameter norm or a directional finite-difference check. Native checkpoint error must be reported alongside residual reduction; a fraction explained is not by itself a fair performance ranking across checkpoints with different starting errors.

The true duration range also imposes bounds `shift-s < delta_s < shift+scale-s`. A linear box-constrained fit is more restrictive than the unbounded projection but remains a first-order model of the complete network. A free linear fit that needs inadmissible durations cannot be interpreted as a feasible repair.

## 6. What amplitude controls do and do not mean

Several normalizations answer different questions. None should silently be called a direction-only or coordinate-invariant analysis.

1. **Global kernel trace normalization.** Replacing K by K/c for one positive c removes one overall kernel scale and rescales the effective ridge parameter. It does not remove per-direction conditioning, layer imbalance, sigmoid slopes or input-dependent feature magnitudes.
2. **Per-anchor receiving-feature RMS normalization.** Replacing q_i by `q_i/rms(q_i)` scales that layer's output parameter Jacobian by an input-dependent factor. The cross-anchor kernel becomes `K_ij/(rms(q_i) rms(q_j))`. This changes the shared tangent fitting problem and can change its stacked column space; it is generally not a fixed parameter-coordinate change. It is an amplitude-equalized counterfactual, not proof that amplitude caused the training outcome. It removes only one common feature scale per layer/anchor; vector-channel magnitudes, downstream anisotropy, sigmoid slopes and controller features remain.
3. **Per-column output response normalization.** Normalizing R's nonzero columns removes their pointwise magnitudes and permits the exact common-G comparison in Section 3. Applied independently across inputs, however, it changes which input-dependent controls would be required of a shared controller. It must not be inserted into the shared-MLP analysis and described as an equivalent model.
4. **Parameter-coordinate changes.** If theta is reparameterized, Euclidean kernel, ridge norm and minimum-norm solution generally change. An invertible linear reparameterization preserves the fit-anchor column space, but can change finite-budget fits and evaluation predictions of an underdetermined interpolator. Trace normalization does not eliminate that dependence. This matters in light of the already completed clock-coordinate experiment.

Zero/near-zero receiving RMS and zero kernel trace need explicit declared handling. Do not turn them into artificial large responses through division by a tiny epsilon and then interpret the result as useful capacity.

## 7. Scope of the positive-clock sign theorem

The earlier witness concerns a **single scalar-coordinate map** `x+p*s(x,t)*x` with fixed p and strictly positive s. Its relative increment has sign determined by p. That theorem is valid for its stated family.

It is not a sign theorem for this complete hidden network. A dense P mixes coordinates, Ph changes direction with h, SiLU and the second solve change the local map, learned input/output projections mix directions, and residual blocks compose them. The theorem neither bounds VelocityNet's oracle error nor predicts that scalar or vector additive gates will beat their parents.

Even locally, positive s does not force a controller perturbation to be positive: around any interior gate, delta_s can have either sign. An unbounded tangent fit may exploit decreases larger than the actual feasible interval; this is exactly why its residual floor is only a relaxed local diagnostic.

The additive study's failure to pass the baseline controls is therefore not inconsistent with the local theorem. The theorem established a genuine difference in a restricted family, not the bottleneck of the trained full network.

## 8. One discriminating measurement, and the design decision it can support

Use the already saved common validation-path anchors, with fit/evaluation indices and times fixed before inspecting the response outcomes. For each selected paired checkpoint, report:

- the unrestricted pointwise duration-control residual floor;
- the exact shared-controller kernel and its residual prediction on a separate fixed anchor subset, across a declared ridge grid;
- native residual error, kernel spectrum, and global-trace/per-anchor-RMS normalization clearly labeled with the meanings above.

This is one nested controller-accessibility diagnostic. The separate anchor subset tests whether a local fitted controller perturbation transfers beyond its fit anchors. It is still drawn from existing validation data, not a fresh independent holdout. Do not choose a winning lambda on those evaluation anchors and call it generalization or a trained architecture gain.

Interpretation should be conditional:

- If scalar pointwise residual remains large but vector does not, the scalar controller has a local rank limitation; the already unsuccessful vector model shows that removing this limitation alone is insufficient for the practical objective.
- If vector pointwise residual is tiny but shared-controller residual/transfer is poor, the next question is how finite controller features organize changes across states, not adding more pointwise directions or a sign-changing gate by itself.
- If a shared small tangent perturbation transfers well, the checkpoint has a locally accessible correction; this points toward learning dynamics or supervision allocation, without proving that a new architecture is needed or that Adam will find it.
- If amplitude equalization changes the result, it identifies sensitivity to the counterfactual metric or state-dependent scaling. It does not isolate a causal explanation of the completed training comparison.

No new design is selected from the algebra alone. In particular, adding a learned receiving matrix Q, mixing the failed Ph/h receivers, sweeping gate temperature, or restoring higher K would each introduce another degree of freedom before establishing a measured bottleneck. A proposed ODELayer-derived design should follow a persistent, loss-relevant restriction identified through the complete network and then survive the same SwiGLU controls.

## 9. Relation to the professor's original layer

The current K1 maps remain frozen conditional Euler polynomials: the generators are DP and `cP+D-cI`. This changes the generator's conditioning, not its numerical accuracy. For the original scalar K-layer, the direct duration response is `P T_(K-1)(sP)h`; K1 reduces to Ph, exactly the response studied here.

For a vector controller at higher K, derivatives of powers require all insertion positions. With A=D P, for coordinate j,

```text
partial [T_K(A)h] / partial s_j
 = sum_(k=1..K) 1/k! * sum_(r=0..k-1)
     A^r E_j P A^(k-1-r) h,
```

where E_j has a one at diagonal entry j. The additive generator replaces E_j P by E_j. Noncommutativity prevents importing the scalar formula unchanged. This identity preserves the mathematical connection to ODELayer; it is not a recommendation to run another order experiment.

## 10. Independent implementation review before checkpoint diagnosis

Reviewed `scripts/controller_tangent.py` and `tests/test_controller_tangent.py` on 2026-10-24. The formulas in the initial implementation are correct for the current independent, plain input-sigmoid K1 models and the current SwiGLU references. The review did not modify source or launch training.

### Algebra and indexing

- Replacing one solve's native output with an independent differentiable anchor gives its native downstream G. Subsequent controllers remain in the computation graph, so both the second solve and later blocks contribute the necessary chain terms.
- The multiplicative P response is `G diag(s)`; the additive P response is `cG`. Their parameter-kernel feature Gram is `h_i^T h_j` without a bias term, correctly reflecting the bias-free P projection.
- Controller first/last affine kernels use augmented feature Grams, correctly including both biases. Multiplying a `(sample, output, sample, output)` view by the cross-sample feature Gram matches flattening of `prediction.reshape(-1)`.
- SwiGLU contributes three disjoint parameter blocks: the first affine's gate half, its value half, and the output affine. Their response factors in the code match the product rule and include all three biases. Adding their kernels is correct because their parameter coordinates are disjoint.
- Adding controller and P kernels covers the whole supported ODE FFN. It excludes common input, FiLM, normalization and output parameters deliberately. This is an entire-FFN tangent comparison, not the entire-network parameter tangent.

The direct-autograd tests compare these assembled kernels with complete-network parameter derivatives, including native downstream controllers. That is a strong check of the algebra and flattening. The sum-over-batch output-gradient shortcut additionally assumes no interaction between examples. Current VelocityNet uses example-wise operations and satisfies this; the routine should document this restriction before being reused with batch-coupled architectures.

### Concrete changes requested during review

1. The initial routine permanently calls `model.eval()`. This changes training flags even though it does not change weights. Either restore every module's original flag on exit, or explicitly require an eval-mode model from the caller. Current eval-loaded checkpoints are not numerically affected, but a diagnostic advertised as read-only should make its behavior precise.
2. The initial guards do not reject tied controllers, although the kernel formula assumes independent parameters. A shared controller can pass the per-layer checks yet require cross-layer terms. Reject repeated parameter identities across captured modules/controllers unless parameter-sharing terms are implemented. Current Stage15 controllers are independent.
3. Compute sigmoid probability directly from its raw logit when constructing its derivative. Reconstructing probability from `(s-shift)/scale` introduces avoidable cancellation near saturation. Reusing the already computed first-layer activation also avoids an extra controller forward call. A saturated-logit direct-autograd comparison can check this edge case.

These are scope/state/numerical improvements, not missing algebraic terms for the declared Stage15 setting. Their final resolution should be recorded with the diagnostic source hash and its final tests.

**Resolution:** all three requested changes were incorporated before the large diagnostic. The routine now rejects training-mode modules and repeated parameter identities, and computes the sigmoid slope from the raw logit. An independent rerun using `/Users/mr.z/ode-workspace-prep/.venv-test/bin/python -m unittest tests.test_controller_tangent -v` passes all six tests, including the saturated-logit case and a primal-versus-dual ridge calculation with the correct native update norm. An earlier attempt with the system interpreter could not import PyTorch; that environment mismatch did not execute the tests and is not counted as a source failure. Review status: **PASS for the declared independent current models and frozen eval-mode analysis.**

Reviewed SHA256 values:

```text
controller_tangent.py:
0c853756dda392549a6b4c606a1a28bee5376c85d5e3224c7e268fcb3035f37a
test_controller_tangent.py:
4cddd1f6f8e120bd008944a413cf647de64372e1ccec266ac893f31f1182fd72
diagnose_controller_tangent.py (ridge and execution scope inspected):
ba191f0f8a66d31a486e518ae7100e90ec2e06509b0eff4c9c429fe1a6102714
```

### Performance and fixed reporting rules

With 80 fit and 80 evaluation anchors in product16, a combined kernel has shape 2560 by 2560 and occupies about 52 MB in float64. Several components and temporary matrices coexist; run checkpoints serially and benchmark one product16 case before multiplying work. Only the fit-fit and evaluation-fit blocks are needed for ridge predictions. If the combined matrix is retained for simplicity, that is a documented memory/computation tradeoff. Reuse the fit-kernel eigendecomposition across a fixed regularization grid instead of solving or diagonalizing afresh for each value.

Freeze the following before looking at outcomes:

- exact fit/evaluation indices, times, checkpoint list and the components being compared;
- one declared kernel scale, for example mean fit diagonal `c=trace(K_fit)/n_fit_outputs`, with zero-trace components explicitly skipped;
- the entire dimensionless ridge grid, not a value selected using evaluation labels;
- numerical symmetry/PSD tolerances and the singular-value threshold for any pseudoinverse or rank result;
- native oracle error alongside every predicted residual curve, including unsuccessful or negative changes.

For `K'=K/c`, predictions use the equally scaled evaluation-fit block. Equivalently, solve the native system `(K+c*lambda I) beta=e_fit`. The implied native parameter-change norm is `beta^T K beta`; record it when interpreting finite-budget accessibility. Positive ridge by itself does not guarantee that this is a small step for the nonlinear model. A norm derived from the RMS-equalized kernel belongs to its counterfactual feature model, not to a realizable native parameter update.

Use phrases such as “the frozen network's linearized FFN response predicts a residual reduction on the declared evaluation anchors.” Do not call this a trained gain, a new model, a fresh holdout, a successful architecture, or proof that the optimizer can reach that solution. The oracle residual is privileged information, and the finite anchor subset is reused validation data. Picking the best-looking regularization on those anchors would add selection and must not be hidden as an independent test.
