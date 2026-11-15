# Interpreting the three comments: spectra, symmetry and the ISRU response

15 November2026. Analysis of the supplied professor comments, with an audit of
existing evidence. No new training or checkpoint sampling is performed here.

**Recommendation:** prioritize learned-weight spectra and full-field Ring8
equivariance, then one matched activation control. The previously proposed
CelebA K2/midpoint pair remains a later option. These comments target unresolved
mechanisms more directly than immediately changing the internal solver.

## 1. Broad updates do not imply broad learned weights

Muon approximately orthogonalizes the momentum-derived update, not the weight
matrix itself. See the [author's definition](https://kellerjordan.github.io/posts/muon/).
For zero decay, W_final=W_initial+sum(DeltaW). Singular values do not add across
these matrices. Updates diag(1,1) and diag(1,-1) both have rank2, but their sum
diag(2,0) has rank1. This is an algebraic illustration, not an explanation of
the particular trained trajectories.

We already measured actual post-step weights and actual parameter differences,
separately from gradients and momentum. The archived [cycle13 CSV](../autoresearch/professor-20261103/cycle13-results/updates.csv)
contains the following20k endpoint summaries, each averaged across six FFN
matrices. A=AdamW.0003, M=Muon.02; these are recorded configurations, not
update-magnitude-matched optimizer experiments.

| Task / model | Update stable rank A/M | Weight stable rank A/M | Weight entropy rank A/M |
|---|---:|---:|---:|
| Ring8 / Free ISRU |2.33 /33.46 |10.77 /9.06 |60.63 /45.89 |
| Product16 / Free ISRU |4.57 /32.78 |13.02 /10.53 |55.55 /43.79 |
| Ring8 / ISRU-D K1 |1.86 /34.23 |7.33 /11.40 |47.02 /48.56 |
| Product16 / ISRU-D K1 |1.65 /32.91 |9.70 /12.38 |47.51 /46.67 |

Thus the professor's suggested opposite behavior already occurs in the ordinary
Free ISRU control on both tasks. These are ordinary W matrices, not ODE P.
ISRU-D is not uniform: on Product16 stable rank rises while entropy rank falls
slightly. Do not summarize every definition as "the rank went up."

Definitions used by the source logger:

```text
stable rank = sum(sigma_i^2) / sigma_max^2
entropy effective rank = exp(-sum p_i log p_i), p_i=sigma_i/sum sigma_i
condition number = sigma_max / sigma_min
```

The first two describe spectral concentration, not exact algebraic rank.
Condition number is sensitive to the smallest singular direction, so it can
increase even when the energy spreads more broadly. In the earlier **scalar
K2** Product16 cohort, mean P condition number is940.8 for AdamW.001 and1017.0
for Muon.02, despite broader spectral energy under Muon.
[Existing condition-number analysis](../autoresearch/professor-20261103/spectrum/findings.md).
That cohort must not be relabeled as the later ISRU-D study.

Next read-only work: publish per-layer weight/update rank curves together and
recover sigma_min/condition numbers for cycle13 from saved singular values.
Log singular or near-singular cases explicitly; do not silently cap an infinite
condition number. Current remote availability of those full raw spectrum files
has not been checked in this pass. Curated aggregate data are already local.

For interpreting performance, also inspect P D(h) and the complete local
Jacobian. For ISRU-D K1 the latter is I+P diag(q'(h)); it is not P alone.
Current CelebA baseline/K1/K8 all use AdamW and cannot answer a Muon comparison.

## 2. Ring8 supplies an exact, testable symmetry

The implemented population has eight equally weighted centers
3(cos(k*pi/4),sin(k*pi/4)), covariance.04I per component, Gaussian sourceN(0,I),
and independent linear interpolation x_t=(1-t)x0+t x1. Simultaneously rotating
the pair (x0,x1) by45degrees preserves its distribution. Conditional expectation
therefore gives the exact population velocity identity:

```text
v_oracle(Rx,t) = R v_oracle(x,t).
```

Both the position and the velocity rotate. Rotating only the velocity vector
while leaving its position fixed is a different operation. This is C8 symmetry,
not invariance under every possible rotation angle.

For an equivariant field with a unique flow and a rotation-invariant source,
the generated population remains rotation invariant. This mathematical
connection is established in [Equivariant Flows](https://arxiv.org/abs/2006.02425).
It does not guarantee the correct mode radius or within-mode shape. A zero
field is also equivariant. Equal sector populations also do not require every
particle to keep its initial sector: C8 permits circulating flux. The native
oracle's zero boundary-normal velocity uses the additional reflection symmetry
(D8), as derived in the existing [mode-mass note](../autoresearch/professor-20261103/mode-mass-derivation.md).

### What is already known

Earlier [central-inversion diagnostics](../autoresearch/professor-20261103/symmetry-results/findings.md)
tested180degrees on older scalar/raw-diagonal models. That does not establish
45degree equivariance for later ISRU-D. No completed C8 field projection or
rotated-pair trajectory test was found in the scoped archived studies.

For ISRU-D from15k to20k, [existing transport results](../autoresearch/professor-20261103/cycle15-results/analysis-report.md)
show:

| Quantity |15k |20k |
|---|---:|---:|
| Oracle field MSE |.010256 |.009773 |
| Centered-residual SW2 |.017683 |.015018 |
| Sector-frequency L1 to equal weights |.038269 |.054504 |
| Pooled native SW2 |.117271 |.132610 |

The pooled source and target sector L1 values are only.006592 and.007996.
Free ISRU and selected SwiGLU also show growing sector imbalance. The open
question is whether a worsening nonsymmetric component of the field explains
this mass error while within-mode fitting improves. That causal explanation
has not been measured yet.

### Minimum diagnostic with saved models

Use the same six15k/20k checkpoints: ISRU-D, Free ISRU and selected Swi64.

1. Check ||v(Rx,t)-R v(x,t)||^2 at common oracle-path and learned-path states,
   separately by time. Verify the oracle as a zero-residual reference. Retain
   both absolute error and normalization by oracle energy; exclude undefined
   normalization near zero energy rather than treating it as success.
2. Start paired trajectories at z and Rz using unchanged Heun128. Plot the
   rotated original trajectory against the trajectory from the rotated input,
   and compare endpoint discrepancies and sector assignments.
3. Project the complete learned field onto C8 symmetry, without retraining:

```text
v_sym(x,t) = (1/8) sum_{k=0..7} R^(-k) v(R^k x,t).
```

Compare native and projected sampling on the original eight shared IID banks,
4096samples each, same Heun128 and target banks. Report pooled/bank SW2,
sector L1, mode centers and centered-residual shape. Group averaging costs
eight field calls: this is a diagnostic, not a free architecture improvement.
Use orbit-paired banks for equivariance checks only, not to manufacture equal
sample counts in the distributional comparison.

The projection is exactly equivariant. Under the invariant true path population,
Jensen's inequality and oracle equivariance imply its population field MSE
cannot exceed the original field MSE. This guarantee does not apply to arbitrary
asymmetric empirical banks or imply lower sampling SW2. If sector balance
improves but SW2 does not, shape/radius errors remain. If the original field
already has tiny equivariance error, investigate those other errors first.

## 3. The missing activation control is a matched bounded response

The [cycle13 experiment](../autoresearch/professor-20261103/cycle13-results/analysis-report.md)
already compares K1 ISRU against linear q(h)=h while retaining P, the rest of
the model, initialization and AdamW.0003:

| Task | Linear field MSE | ISRU field MSE |
|---|---:|---:|
| Ring8 |.01441079 |.00977337 |
| Product16 |.07540868 |.01605896 |

That supports using a nonlinear response. It does not show that ISRU's ODE
origin is necessary. The older cubic/SiLU split also changes oddness, local
slope and saturation. Free ISRU changes matrix parameterization, initialization
and residual placement, so it is not a one-factor activation substitute.

The clean new control keeps F_i(h)=h+P_i q(h), both K1 maps and middle SiLU,
and replaces only q:

```text
ISRU:        q(u) = u / sqrt(1 + 1.2*u^2)
scaled tanh: q(u) = tanh(sqrt(1.2)*u) / sqrt(1.2)
```

Both are odd, have slope1 at zero, and saturate at +/-1/sqrt(1.2). Their shape
and derivative decay differ; their cubic coefficients are -.6 and -.4, not
matched. A scoped search found no completed version of this exact control.
The activation itself is established [prior art](https://arxiv.org/html/1710.09967v2#S5).

Proposed minimum training, if selected: two new scaled-tanh trajectories,
Ring8 and Product16, seed0,20k updates,batch512,AdamW3e-4, same native synthetic
source/initialization/streams. Reuse the linear/ISRU/SwiGLU anchors only after
their provenance and replay compatibility are verified. Evaluate new and old
checkpoints with the same later Heun128/common-bank protocol, rather than
mixing it with the early single-bank SW2. Do not combine activation and
optimizer changes.

If tanh matches ISRU, a generic bounded odd response is a sufficient explanation
within this comparison. If ISRU wins, its particular curve deserves attention;
the result alone does not isolate its origin as an exact ODE flow. In the old
cubic split, q_r is the exact y'=-y^3 subflow and satisfies q_a(q_b(u))=q_(a+b)(u).
The later fixed-r K1 insertion does not explicitly exploit that composition law.
Current K8 approximates a different, entrance-frozen P D(h0) exponential.

Coordinatewise ISRU and tanh do not generally commute with a45degree rotation.
That fact does not attribute the complete network's symmetry error to either
one: P, SiLU, conditioning and hidden representations also matter. Measure
equivariance at the complete physical velocity field. The old trial named
"radial" used h^2/sqrt(1+mean(h^2)); it is not rotation-equivariant vector
ISRU, so it cannot be cited as a failed radial-equivariant activation test.

## Revised recommended order

1. Reorganize the existing update/weight spectra and recover missing cycle13
   condition-number curves. No training is required for these measurements.
2. Run the saved-checkpoint C8 field/trajectory/projection diagnosis, with
   ordinary controls and the original sampling banks.
3. Select the two small scaled-tanh fits to separate response shape from the
   broad bounded-nonlinearity explanation.
4. Revisit the CelebA K2/midpoint pair after these results. It remains a
   proposed test, not an already approved or submitted run.

The testable chain is optimizer/feature response -> learned transformation ->
full-field symmetry error -> mode allocation. Those arrows are hypotheses to
measure, not conclusions supplied by the terminology. This analysis does not
alter finalized Report1/2 or launch experiments.

## Verification in this pass

[Evidence receipt](2026-11-15-professor-comments-evidence.json) retains all eight
selected spectral rows and independent numerical witnesses. The analytic
Ring8 oracle's C8 residual is below1.5e-14; generic-field group averaging gives
a residual below3.4e-16. Matched activation slopes, the ISRU subflow composition
identity, local evidence links and all11 protected Report2 hashes pass.
These algebra checks are not trained-field equivariance measurements.
