# Registered conditional input-information attribution

The cycle07 thin/clean field gap is known. This read-only study asks whether that gap is mostly explained by the different information retained by the learned input projections, or by error remaining after that projection. It nominates no architecture and reopens no stopped setting.

Use every cycle07 checkpoint at0,1000,2500,5000 updates: thin/thick, ODE K2/Swi128/Swi192, clean/shifted/direct,72 total. Use every row of the existing20000-point field bank per law and its saved full oracle and prediction arrays. Use the existing unknown target basis only in evaluation. Preserve all earlier data, source, weights, training, metrics and sampler settings.

Let U=B X, where B is an orthonormal basis for the learned linear input row space. An input bias and the time embedding lose no additional information. Derive exact Gaussian-mixture conditional means E[Y|U,t], E[Z|U,t]. The best decoded field among arbitrary measurable raw functions of(U,t) is `a/delta*(E[Y|U]-E[Z|U])` for direct and `(E[Y|U]-X)/delta` for either skip head. The [theory](theory.md) states the assumptions and the distinction from the full field oracle.

For each fixed checkpoint and row write

```text
prediction - full_oracle = information_displacement + residual
information_displacement = conditional_best - full_oracle
residual = prediction - conditional_best.
T = I + F + C,   C = 2 * dot(information_displacement, residual).
```

Report T,I,F,C per coordinate, in the full space, target tangent plane and target-normal complement. Retain the finite-bank cross term even though its population expectation is zero. F includes all remaining output/representation/optimization restrictions, not merely an FFN optimization error. It is not an attainable loss guarantee for the actual network. Also publish fixed time bins[0,.2),[.2,.4),[.4,.6),[.6,.8),[.8,1). Save per-row components privately.

The sole endpoint attribution rule concerns the already observed thin/clean ODE gap to each Swi control. Define positive improvement `Delta=T_Swi-T_ODE` and differences DeltaI,DeltaF,DeltaC in the same orientation. Call the gap predominantly input-information-related only if DeltaI>=.8*Delta and abs(DeltaC)<=.2*Delta for both controls. Call it predominantly remaining-residual-related only if DeltaF>=.8*Delta and abs(DeltaC)<=.2*Delta for both. Otherwise retain mixed/unresolved. Report all signed contributions; ratios can exceed one or be negative. No new candidate can pass cycle07's failed criteria through this descriptive attribution.

Pre-outcome checks: compare conditional means with independent dense joint-Gaussian component conditioning for both laws; compare full-observation fixtures to the native full oracle; test rank-deficient input extraction and pointwise decomposition. Float64 fixture agreement requires max<=1e-8; algebraic decomposition<=1e-10. Rank threshold is1e-10*max(sigma_max,1). All bank times must be strictly below1; no endpoint covariance regularization is introduced.

Independent audit: obtain input row spaces via QR, compute the conditional means through2D Woodbury solves rather than the formal SVD of BQ, replay all72 decomposition arrays and summaries, and replay all18 native final predictions. Require max<=1e-6 between the two conditional fields, per-row component relative/absolute agreement1e-6/1e-8, source/prediction replay unchanged, and all prior seals intact. Keep failed audit versions and diagnose implementation issues without changing scientific attribution thresholds.

Register source before conditioned fitted outcomes. Bind source/settings and cycle07/08 seals. One single-thread Mac CPU process,600s execution ceiling and checkpoint/time/RSS monitoring; no GPU training or MBP/torch dependency. Runtime results.tsv records the diagnostic as retained evidence, never as a quality nominee. Publish small audited tables/plots; raw arrays/logs stay outside Git.
