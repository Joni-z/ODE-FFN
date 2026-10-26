# Scope and numerical checks

Frozen runtime: `/Users/mr.z/ode-workspace-prep/architecture-search/adjoint_composition_20261026`. All33 source files were frozen before measurements. Session42571 terminated normally with exit0; output/status.json is COMPLETE. The source and all checkpoint/bank/canonical-array hashes were rechecked at the end. The previous orientation report manifest was checked before reuse; no prior checkpoint, run or submitted report was changed.

The experiment covers18 actual learned states: ODE K2/SwiGLU128/SwiGLU192, Gaussian/Ring/Product, trained under A or B. It evaluates72 complete partial-sum fields and360 complete partial-sum derivative panels, including all512 anchors at each of five times. Saved per-input components, Jacobians and exact risk polarization arrays remain under runtime/output/cases. No test bank or new training label was used.

The complete sums reproduce independently saved canonical A values within1.07e-14 and complete input Jacobians within3.56e-14. The four partial sums each pass finite differences at every declared case/time,360 checks total, maximum relative error4.39e-10. Full-field input/time/all-parameter loss-gradient comparisons on every learned state have maximum absolute difference5.69e-13. Four focused unit methods additionally enumerate all eight paths at small width and check each component's own parameter/input/time gradients, nonzero-head batch/inference behavior, symmetric-P clock skew, and noncommuting local gradient flows.

Component field/skew Gram matrices retain all cross terms. Summed component norms are not the norm of the sum. Finite-change alignment and change energy exactly account for the error of every partial sum relative to the full A evaluation at the same weights. Neither these decompositions nor one-probe finite differences prove that a smaller freshly trained model cannot compensate.

The single one-thread CPU analysis completed in about41seconds. Case-boundary telemetry sampled a maximum425.25MiB RSS and at least4.738GiB available memory; it is not a continuous peak-memory bound. No GPU was allocated. Lenovo remains offline in the authoritative Tailscale state, so no remote work was launched.

The raw analyzer's first heatmap has an overlapping colorbar. The report-only renderer places the colorbar on a separate axis and shares row labels; both raw numerical output and the original figure are preserved. The corrected heatmap and skew plot were visually inspected. There is no model rerun or changed numeric threshold for this layout correction.

The primary GradNet citation supplies an established architecture construction and its assumptions; it does not validate our ODE quality claim. A discovered EqM workshop paper was excluded from empirical interpretation because direct full-text access returned a browser challenge and its training objective differs. No main-conference or state-of-the-art claim is inferred from that search result.
