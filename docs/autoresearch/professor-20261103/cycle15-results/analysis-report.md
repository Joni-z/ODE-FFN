# Centered residuals improve while mode allocation drifts

The next research object is the probability mass sent to each mode. Across
the saved15k/20k Ring8 checkpoints, field fitting and centered-residual SW2
improve in all three models, while sector frequencies move farther from the
known equal probabilities. ISRU-D and selected Swi64 also worsen in the larger
pooled native SW2 readout; Free ISRU changes little. This is a concrete shared
tradeoff to explain before another architecture.

## Compare the same models at the same checkpoints

The original native validation MSE is kept separate from the new common-path
diagnostic. Pooled scores below use all eight saved banks,32,768samples and
the original256directions, with no new model calls. Assigned-mean SW2 applies
the same formula after moving each sample to its assigned target mean; centered
residual SW2 applies it after subtracting that mean. These two diagnostics
separate statistics and are not additive terms in the native score.

| Model | Step | Native validation MSE | Native pooled SW2 | SW2 of assigned means | SW2 of centered residuals | Sector-frequency L1 to1/8 |
|---|---:|---:|---:|---:|---:|---:|
| ISRU-D | 15k | 0.010256 | 0.117271 | 0.224172 | 0.017683 | 0.038269 |
| ISRU-D | 20k | 0.009773 | 0.132610 | 0.245073 | 0.015018 | 0.054504 |
| Free ISRU | 15k | 0.011684 | 0.143685 | 0.256844 | 0.016521 | 0.036194 |
| Free ISRU | 20k | 0.009998 | 0.144674 | 0.262740 | 0.009230 | 0.073730 |
| Selected Swi64 / Muon | 15k | 0.013655 | 0.109579 | 0.217115 | 0.032057 | 0.039062 |
| Selected Swi64 / Muon | 20k | 0.011279 | 0.136426 | 0.250452 | 0.012577 | 0.056030 |

The pooled target sector L1 is0.007996; pooled Gaussian source L1 is0.006592.
The native oracle preserves source sectors. The learned20k frequencies are
therefore visibly different from both analytic equal weights and this small
source/target sampling imbalance. Assignment confidence is retained, including
the few points near boundaries; none is discarded.

All three models have smaller component-centroid errors at20k. Conditional
covariance errors are mixed: their equal-component means increase for ISRU-D
and Free ISRU and decrease for Swi64. Thus the centered-residual improvement
is a pooled residual-distribution result, not a claim that every within-mode
statistic improves. Complete moments are in sectors/component-moments.csv.

## First resolve the sample-bank reversal

The mean-of-eight4,096-sample curves worsen from15k to20k, but banks0,1,5
improve for every model and the other five worsen. On preselected bank0,
ISRU-D SW2 changes0.217721 to0.120893, Free ISRU0.236997 to0.138823, and
Swi64 0.195107 to0.103981. We preserve this reversal rather than choose another
bank to recover the mean trend. All24paired checkpoint changes are published.

Pooling retains a native SW2 deterioration of0.015340 for ISRU-D,0.000988
for Free ISRU and0.026847 for Swi64. Across all eight leave-one-bank-out pools,
the corresponding differences range[-0.002480,0.042604],[-0.019422,0.027826],
and[0.004309,0.055289]. The selected ordinary model gives the most consistent
pooled deterioration; ISRU-D has one omission reversal and Free ISRU has
several. These overlapping pools are sensitivity cases. The initial eight-bank
estimator remains unchanged, and no training replication is inferred.

## Common paths locate different parts of the error

All three models use the same4096noise points and native128step oracle path.
Reference128/256/512 endpoints differ by coordinate RMSE9.50e-5 (128vs256)
and2.62e-5 (256vs512);128vs512 is1.21e-4. Their SW2 values are
0.140937/0.140955/0.140959. Common-state error profiles span33times.

| Model | Integrated common-path MSE15k | At20k | Early MSE15k | At20k | Late MSE15k | At20k |
|---|---:|---:|---:|---:|---:|---:|
| ISRU-D | 0.010519 | 0.009987 | 0.007435 | 0.009715 | 0.014825 | 0.011814 |
| Free ISRU | 0.011974 | 0.010319 | 0.010606 | 0.010369 | 0.014776 | 0.008444 |
| Selected Swi64 / Muon | 0.013909 | 0.011074 | 0.009262 | 0.011145 | 0.020259 | 0.010716 |

Early/late columns average the saved times in[0,.25]and[.75,1]. ISRU-D and
Swi64 improve late errors while their early errors increase. Free ISRU has a
different early change. Radial/angular components and full2x2Jacobian data
are retained in profiles.csv. Median actual error-direction step gains are
close between checkpoints; the recorded data do not show a median amplification
blow-up. Extremes and counts remain in step_decomposition.csv.

The copied native step gives the exact discrete identity
`e_(i+1) = [Psi_v(y_i)-Psi_v(z_i)] + [Psi_v(z_i)-z_(i+1)]`.
Every saved sample/step satisfies it with zero replay residual. This separates
actual propagation from a defect introduced at the common reference state.

## Oracle replacement and sector cases

All six cuts/policy combinations plus full learned/oracle anchors are retained
for each checkpoint. In the learned-prefix/oracle-suffix cases, keeping the
learned first75%leaves almost all common-source endpoint error. Using the
oracle for the first75%and the learned field only in the last quarter leaves
endpoint MSE about0.00037–0.00148, compared with full learned0.044–0.076.
This identifies an early/middle trajectory effect in the chosen source bank.

For example, ISRU-D20k has endpoint MSE0.049340 to the common oracle.
Oracle-prefix75%reduces this to0.000752; learned-prefix75%followed by the
oracle gives0.050086. Later accurate dynamics do not recover an already
different particle-to-mode assignment in this case.

The sector accounting makes the change tangible:97.5–99.7%of gross crossing
events occur by t=.75, and94.7–97.3%of common-source endpoint squared error
comes from particles assigned to a different final oracle sector. The oracle
path has zero crossings. Gross crossings include recrossings; net sector
masses, which determine balance, are saved separately. Equal populations can
coexist with particle reassignment, so these are different quantities.

A finite-bank counterexample also matters: on bank0, the full oracle has
SW2 0.140937, while ISRU-D20k learned-prefix25%then oracle gives0.077587.
The latter has worse empirical L1 to true sector weights (0.025879 versus
oracle0.021973) but better L1 to that particular target bank (0.023438 versus
0.037109). This is why a low single-bank distance is not used as a standalone
causal ranking. Oracle-relative endpoint MSE describes the same-source map,
not the optimal distributional Wasserstein coupling.

## Mathematical direction and next test

The [derivation package](../mode-mass-derivation.md) uses sector probability as
the object. Native Ring8 reflection symmetry makes the oracle normal velocity
zero on sector boundaries. Continuity then expresses mass drift by net normal
flux. Around the oracle, the first-order endpoint mass change is the
time-boundary integral of `-p_t (v-u) dot n`. It is a different functional from
bulk quadratic field error. The exact-law boundary check is below2.61e-13.

For a K1 matrix change, `D_P v[DeltaP]=J_downstream DeltaP q(h)`.
Consequently the first-order sector response depends on q, downstream
derivatives and boundary geometry; for eight sectors its matrix-update
quadratic form has rank at most7. This supplies a concrete link from the
measured mode drift to feature/update geometry. The first-order predictor
has not yet been numerically calibrated at these trained fields.

Next finite question: does the analytic-density-weighted boundary-normal flux
predict the observed net sector drift, first under weak oracle/model mixing
and then at the native model? Calibrate that before choosing a mass-sensitive
ODE-layer or optimizer control. Existing Free-ISRU initialization/placement
differences remain explicit; no new architecture, training seed or JiT run
is selected. The completed Report2 PDF/TeX remain unchanged in this pass.

## Verification and artifacts

The original source is6fe9e7f. Fresh documented CUDA readiness passed all six
4096sample/checkpoint/helper replays. All48saved endpoints,198profiles and
768step decompositions independently replay; SW2 error is at most2.78e-17.
The actual oracle finite-difference Jacobian error is6.57e-10. Original366
cycle14files are rehashed unchanged. The72,270,540byte runtime archive has
SHA256 ff1acce5672c701136e848322e3b0917780114d6b7c171e4a7686f931aafaedf.
Raw arrays stay outside Git; CSVs, figures, source and receipts are published.
