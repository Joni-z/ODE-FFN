# Measurement and verification scope

## Units
The diagnostic contains 24 fixed functions: eighteen final students and six local teachers. Each is evaluated at eighty pre-existing raw inputs, sixteen from each of five time panels. The three positions belong to the same common prefix model. Times, coordinates, positions and anchors are not independent training repetitions. There is no new training, random sampling, test set, p-value or inference about a training population.

All complete derivatives hold time fixed and differentiate with respect to the sixteen physical input coordinates. The common prefix Z_j maps those inputs into 64 feature coordinates. Its first and second derivatives are included, so arbitrary ambient hidden perturbations are not substituted for the data-composed function. Parameters/buffers preserve their exact stored values when promoted to float64. Native float32 prefix arrays replay all original 512-row panels bitwise; the largest double-promotion change on selected anchors is 1.758e-5 absolute, within the predeclared scaled tolerance.

## Metrics
Curvature errors are squared Frobenius norms summed over anchors/output/input pairs, divided by the corresponding teacher energy. Displayed RMS percentages are 100 times the square root of those ratios. Pooling uses summed energies rather than an unweighted average of per-time ratios. Function risks from the earlier training study remain distinct from derivative errors.

The clock and origin-omission criteria use ratio at most 0.01 in every time panel for a given ODE student. This corresponds to 10% relative RMS discrepancy. One of six students satisfies the uniform frozen-clock criterion; two of six satisfy the uniform origin-omission criterion. The full continuous 45-row table includes three ODE teachers as context. No criterion is a statistical test or architecture acceptance rule. Energy below 1e-18 would be indeterminate; no such panel was omitted.

Curvature route energies are not independent contributions. Their signed inner products are retained. Local/prefix error decomposition also includes the cross term, and sums reconstruct the complete error. No mediation or causal attribution is inferred from these algebraic decompositions.

## Exact identities and supplementary scope
The registered identity removes at most two physical-input clock covectors and two propagated output vectors. All measured frames remove exactly two of each; all prefix Jacobians have rank sixteen. Frame bases use normalized constraints and a fixed 1e-10 relative SVD tolerance, with a 1e-15 zero-vector tolerance. Frames are anchor-dependent and derived only from the current ODE module and prefix, not optimized using target curvature.

The complete-prefix corollary was derived after the registered results, recorded separately, and checked at all 720 ODE-function/anchor conditions. Native local Jacobians verify the rank-two difference and output projection. Maximum complete-Hessian identity residual is 6.73e-15; maximum projected Jacobian residual is 1.95e-15, and the largest third singular value of the native Jacobian difference is 4.96e-15. See [composed-identity-receipt.json](composed-identity-receipt.json). These identities do not impose one global fixed frame or establish a retrained-architecture lower bound.

The global upper-clock intervention is another separately declared exploratory analysis. It fixes both durations to 1 without changing any parameter, selection sweep or training. All nine native validation functions replay sealed arrays bitwise. The provided-duration native implementation agrees with an independent effective-map expression within 1.42e-14 in float64. All nine clamped arrays are preserved and hashed. See [upper-clamp-receipt.json](upper-clamp-receipt.json).

The polynomial-differential eigenpair screen was declared after aggregate curvature results and prior P minima were viewed. All 1440 duration/matrix conditions are retained. Each minimum eigenvalue modulus is checked using its full 64-by-64 eigenmatrix under the actual Sylvester operator; maximum residual is 5.72e-15. Eigenvalue-modulus ratios are conditioning lower bounds, not computed singular spectra or native residual-gradient alignments. None meets the combined exploratory threshold; the maximum lower bound remains 128.6. No reparameterization or optimizer is selected from that value alone.

## Independent native verification
The preflight verifies all 24 composed functions at the fixed middle-time first anchor, including frozen-clock counterfactuals and reduced identities. Native discrepancies are at floating-point precision.

The formal independent audit uses pre-hooks on the actual complete backbone to reconstruct all eighty prefix values/Jacobians/Hessians, matching the explicit prefix traversal exactly. Native automatic differentiation checks all 1920 complete Hessians and all 1920 local pullback Hessians, plus 720 frozen-clock complete Hessians and 720 reduced identities. Maximum complete/local/frozen Hessian differences are 1.78e-14, 1.60e-14 and 1.38e-14. All 119 ready bindings and all 34 previous-bundle files retain their hashes. The audit receipt is `/Users/mr.z/ode-workspace-prep/architecture-search/data_curvature_20261026/audit.json`.

The registered source, theory and protocol were not edited when the post-result corollary, eigenpair screen or upper-clamp intervention were added. Separate source/receipts label those extensions. A passing numerical audit establishes consistency in this scope, not causal explanation, empirical superiority of a new architecture, or transfer to images.

## Exports
- [Clock approximations](clock-approximation.csv): 45 rows.
- [All ODE anchor geometry](approximation-anchors.csv): 720 rows.
- [Time-pooled route inner products](route-gram.csv): 720 rows.
- [All student/time curvature risks](curvature-fit.csv): 90 rows; [pooled risks](fit-aggregate.csv): 18.
- [Registered local reduced risks](reduced-fit.csv): 90 rows; [pooled local risks](reduced-aggregate.csv): 18.
- [Exploratory eigenpair screen](branch-screen.csv): 1440 rows.
- [Promotion checks](promotion.csv): 80 rows.
- [Complete-prefix identity checks](composed-identity-checks.csv): 720 rows.
- [Supplementary complete reduced risks](reduced-composed.csv): 90 rows; [pooled risks](reduced-composed-aggregate.csv): 18.
- [Upper-clock intervention](upper-clamp.csv): 9 rows.
