# Registered diagnostic: complete-field Jacobian constraints

Registered before reading the new violation outcomes. This is one read-only analysis of existing Stage17 endpoints; it selects no architecture or training run.

Use final `vector_parent`, `swiglu_budget` and `swiglu` common-validation arrays from `shared_features_20261024_m1`: Ring8 and Product16, times .1/.3/.5/.7/.9, all 512 saved anchors per panel. This fixes 30 arrays and 15360 model-anchor records. Verify frozen source/data/array identities, common inputs and oracle fields across arms, and reproduce each native aggregate velocity-Jacobian error.

For each saved velocity Jacobian J, set a=1-t and A=I+aJ. The true denoiser Jacobian T=I+aJ_target is symmetric PSD for this independent standard-Gaussian-source path. Define S=(A+A^T)/2, K=(A-A^T)/2 and Q=PSD_projection(S). Record:

```text
D_skew = ||K||_F^2
D_negative = sum_i min(eigenvalue_i(S),0)^2
D = D_skew+D_negative = ||A-Q||_F^2
E = ||A-T||_F^2, E_projected = ||Q-T||_F^2
E >= D+E_projected.
```

Report eigenvalue minima, negative-eigenvalue violation fractions and these distances per anchor, panel and dataset/arm aggregate. A numerical violation means lambda_min(S) < -1e-10*max(1,||S||_2); also retain raw minima and the tolerance. Validate the oracle symmetry/PSD and the actual projection/Pythagorean inequalities using scale-aware floating-point tolerances; preserve the measured slack.

The primary aggregation uses velocity-Jacobian units: divide every denoiser squared distance by a^2 **before** averaging over anchors and the five times. Compare the resulting original error with the native Stage17 summary. Ratios use sums of errors, not averages of per-anchor ratios. Separately label any unweighted denoiser-space aggregate. No model, time or anchor is selected after seeing violations.

Write per-anchor/per-panel/aggregate CSVs, a concise numerical summary, separately hashed input/protocol/script provenance and two figures: aggregate error components and time-dependent structural-distance/PSD-violation fractions. Only mathematically necessary tolerances are used; no outcome-driven pass threshold or architecture nomination is introduced.

The PSD projection is independent at each saved point. It need not be an integrable Jacobian field, the derivative of a global denoiser, realizable by an ODELayer network, or reachable by training. Its distance is a structural component of the saved Jacobian error, not a trained-model improvement, an architectural approximation lower bound or evidence about generated samples. The purpose is to determine whether a complete-field constraint addresses a material part of existing error before designing another operator.
