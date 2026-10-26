# Data-composed curvature diagnostic

Fixed before inspecting these derivative outcomes, 26 October 2026. This follows the completed [local FFN study](../research-local-ffn-teacher-2026-10-26/analysis-report.md). It is a derivative diagnostic and mathematical check, not an architecture trial.

## Question and rivals
The local fitting gap is mostly outside an empirical affine component. The previous origin identity does not establish what happens on post-normalization inputs. Candidate descriptions are: clocks contribute little so the local response resembles two fixed linear maps around SiLU; clock derivatives materially alter the response; or feature-prefix curvature makes local ambient-space curvature a poor account of the data-composed function. These can coexist. No description is treated as established.

## Fixed models and anchors
Use all eighteen final cold-student modules from the sealed local-FFN study and all six original-ODE/SwiGLU128 local teacher functions. Every module retains its exact saved parameters and actual buffers. At each of the five existing times 0.1/0.3/0.5/0.7/0.9, take panel rows 0:16. These eighty fixed raw inputs are common across three module positions; no test set, optimized direction or new random sample is used.

The common feature map Z_j(x,t) is the unchanged SwiGLU128 teacher prefix used to create the local study's inputs. Evaluate all derivatives with time fixed, with respect to the sixteen physical input coordinates. Compute both J_Z and H_Z. A module F then has

```text
J_(F∘Z) = J_F J_Z
H_(F∘Z)[u,v] = H_F[J_Z u,J_Z v] + J_F H_Z[u,v]
```

Both terms must be retained. Arbitrary directions in the 64-dimensional hidden space can leave the feature distribution and are not the primary measurement.

Promote the exact stored float32 parameters/buffers to float64 for derivative algebra. Check native float32 prefix outputs against the historical cached panels under the original batch boundaries, then record the float32-to-float64 change. Before checking, allow at most 2e-5 times max(1, absolute feature maximum) for that promotion difference. This checks numerical correspondence; it does not create a new trained model or replace the saved function-risk outcomes.

## Exact local decomposition
For L(h)=Q(a(h))h, Q=I+aP+(a²/2)P², C=P+aP², g=grad a, the exact bilinear Hessian is

```text
H_L[u,v] = (Cu)(g·v) + (Cv)(g·u)
           + (P²h)(g·u)(g·v) + (Ch) H_a[u,v]
```

The last two terms vanish at the origin but need not vanish on data. For F=L2∘SiLU∘L1, decompose H_F into middle-activation curvature, first-clock curvature propagated through later derivatives, and second-clock curvature. The middle route includes full clock-dependent first derivatives; it is not the frozen-clock map. Derivation and notation are in theory.md.

At each anchor also define a local counterfactual that freezes both clock values at that anchor. Its Jacobian J_0 and Hessian H_0 are those of Q2 SiLU(Q1 h), with the native clock values held fixed. Its data-composed Hessian includes J_0 H_Z. Compare full and frozen-clock data-composed Hessians. Separately compare a diagnostic approximation omitting the two single-solve Hessian terms that vanish at the origin, while retaining native first derivatives and activation values. Neither is an updated parameterized predictor evaluated for generation.

Primary descriptive ratios are sums of squared Hessian differences divided by sums of squared full Hessian entries within each fixed-time sixteen-anchor panel. A uniformly accurate frozen-clock description requires ratio at most 0.01 at every time for each of the six ODE students; the same rule separately operationalizes the origin-term omission. The 0.01 squared threshold means at most 10% relative RMS discrepancy. A denominator below 1e-18 is indeterminate and reported, not omitted. Include the three ODE teachers as procedural/context controls. Report continuous ratios, all five times and all cases; these criteria are not statistical tests or architectural acceptance gates.

## Exact reduced-space identity
There is also an any-input algebraic statement that does not require small clocks. Inside the physical-input tangent space, remove at most the two scalar-clock first-derivative directions. In output space remove the two propagated clock-Hessian vectors. The remaining local Hessian equals the correspondingly projected fixed-clock SiLU curvature exactly. Construct the frames from the ODE module and common prefix only, without using teacher outputs to optimize them. Full definitions and proof are in theory.md.

Check the identity for all nine original-ODE modules on all eighty anchors. In each student frame, report the retained teacher local-curvature energy and all three students' projected local-curvature error. This is a conditional property of the current trained geometry, not a lower bound on every ODE parameterization. The prefix-curvature term is reported separately and not silently included in that local identity.

## Comparison and validation
Report all eighteen students' value, first-derivative and complete data-composed Hessian errors relative to their corresponding local teacher, with local/prefix error contributions and cross terms. Treat positions and times as dependent conditions, not training replications. Use descriptive exact empirical quantities, no significance or training-variability claims.

Preflight all 24 functions against independent native derivatives of the complete composition at the fixed middle-time first anchor. Compare values, Jacobians, Hessians, analytic route sums, frozen-clock maps and reduced-space identities, using float64 rtol 2e-8 and atol 5e-10. Formal audit checks every complete native Hessian on all eighty anchors for all 24 functions, plus every reduced-space identity and cached prefix derivative. Do not loosen thresholds after outcomes. Preserve technical failures and freeze any corrected source version explicitly.

## Execution and stopping
Cache common prefix derivatives once. Six jobs, one per teacher family/position, each evaluate the target and all three students. Use at most two single-thread CPU children with five-second actual CPU/RSS monitoring; no GPU reservation. Bind all source, checkpoint, input and protocol hashes; never restart on an observation timeout. No architecture, loss or training schedule changes follow automatically from this diagnostic.

If the frozen-clock/origin approximation fails, stop using that approximation as the explanation. Even a valid reduced-space identity does not establish task relevance without retained teacher energy and error evidence. A later design still requires a distinct mechanism, matched ordinary controls and ordinary FM/transfer/cost validation.

Procedural guidance: hypothesis-generation, experimental-design, planning-with-files, daily-coding and results-analysis. Workflow provenance is [Kassis et al. (2026), Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065), with metadata verified in the preceding review; it is not model evidence.
