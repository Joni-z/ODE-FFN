# ODE autoresearch notes

## Evidence used at setup
- Canonical starting commit: `2d2259a8e34fb15d48e57f9492f18f4d026e57c8`.
- Official autoresearch: https://github.com/karpathy/autoresearch, local `/Users/mr.z/autoresearch`, revision `228791fb499afffb54b46200aca536f79142f117`.
- `docs/research-joint-dependence-2026-10-26/next-question.md`: the ISRU advantage did not transfer; no new architecture was selected.
- `docs/research-architecture-search/coupled-generator-prior-review.md`: frozen-gate feedback alone was rejected without independent task evidence; a complete nonlinear derivative differs from its frozen-generator approximation.
- The native synthetic backbone uses pre-normalization, time FiLM and residual scaling. The complete hidden field includes their input derivatives. A derivative of the bare FFN is insufficient to describe a higher-order solver for the actual block.

## Interpretation boundary
The next diagnostic tests whether changing the integrator of an already fitted Euler field helps. Its negative outcome would reject a post-training upgrade, not all possible higher-order architectures. Its positive outcome would justify a separately registered training comparison, not an ODE-specific advantage on its own.

## Cycle01 independently replayed outcome
- Registration commit: e0ad302, created before preflight and new outcomes.
- Runtime: `/Users/mr.z/ode-workspace-prep/autoresearch/oct27-cycle01`.
- Preflight passed12 native checkpoint replays and36 float64 complete-field tangent checks.
- Formal process PID32883 exited0 after23.113seconds;204 field cases,100 time panels,20 common-source generation checks.
- Final all-block complete-Taylor MSE / unchanged Euler MSE: Gaussian47.54, Ring15.68, Product8.32, dependent-sign33.81. The registered premise criterion fails.
- Actual complete-Taylor and midpoint dense inference MACs match:130752 per16D example, versus81792 Euler; training MACs389120 versus242240. This is a diagnostic of frozen models, not a matched-training-cost result.
- The common extra-compute controls also worsen all four endpoint field errors. This suggests that treating a fitted Euler residual update as a solver-independent field is unsupported here; independent verification and precise negative-claim scope still matter.
- Audit-v6 completed204 cases/34,680,000 prediction coordinates,100 panels and20 samplings, preserving91 inputs and333 formal files. All failed audit sources/logs remain recorded under the private runtime; deviations.md explains each numerical/assumption correction.
- Exact output-error decomposition: Product cross term-0.0086382, displacement energy0.6848118; sign-code cross term0.0250104, displacement energy1.3681410. The fixed correction's output displacement dominates the small alignment benefit on Product.
- Do not turn this failure into a sign/step-size sweep or start Taylor training without new evidence. The full-support mixture control proposed in the cycle analysis addresses an unresolved interpretation of prior data dependence, not a new architecture claim.

## Cycle02 independently replayed outcome
- Registered/published4016837 before data-law training. Runtime `/Users/mr.z/ode-workspace-prep/autoresearch/oct27-cycle02`.
- Five unchanged native models completed5000 seed0 updates; PID34331 exited0,56.17s, two one-thread CPU workers, sampled peak summed RSS0.802GiB.
- Oracle enumeration65536 components agreed with native values/Jacobian to2.92e-13/2.28e-12; five cold-state and exact-resume checks passed.
- New-law field MSE: original ODE0.398957, ordinary ISRU0.163557, Swi64/128/1600.192344/0.138033/0.113254. ISRU/best-Swi ratios: field1.4442, Jacobian1.0484, SW2 1.3287.
- Independent audit passed20 checkpoints,25 panels,five generated batches,48 matrix spectra and5000 common-stream batches;310 inputs and prior sealed evidence unchanged.
- Literal Gaussian-component-count change alone is insufficient to explain ISRU's reversal. Component weights, label entropy and posterior ambiguity are not controlled by equal count.
- Post-outcome Bayes covariance decomposition: between-branch response energy / total nonlinear response energy0.1332, not an additive attribution. At the largest-term anchor ISRU outperforms Swi160, preventing a simplistic pointwise causal explanation. Uncertainty bins also differ in time.
- Existing hidden covariance-field failure was inspected. Do not reuse that construction under a new name; no third architecture follows automatically from these facts.

## Publication and next-review boundary
- GitHub main b35497b contains both completed cycles, source, figures, ledgers and audits; MBP is synchronized. All four new scientific commits use Zhizhe Zhang. No data arrays, checkpoints or raw logs were uploaded.
- Main model code, configs, tests and submitted reports have no changes relative to2d2259a; the new studies use frozen native code. Runtime storage is151MiB for cycle01 and68MiB for cycle02.
- Read the old complete-field constraints, conservative-readout and recurrent-potential protocols again. Full-map monotonicity is different from hidden covariance or mere gradient readout, but the existing evidence does not justify a scalar-potential variant or a revived coupling setting. No such architecture or training is selected.
- The next review must demonstrate a complete-map computational purpose beyond a valid internal ODE identity, with an ordinary matched-cost comparator. Keep current failed directions stopped. Do not equate posterior covariance in data coordinates with a learned hidden P.

## Cycle03 registration rationale
- The next bounded step is a compatibility theorem, not another altered quality candidate: ordinary SwiGLU has an exact lifted conditional shear generator with P²=0, no higher series terms, and identical useful computation.
- Complete FFN derivatives retain the gate path. Nonnormality permits arbitrarily large norm amplification despite all generator eigenvalues being zero. Encoder/conditional-generator/readout composition prevents importing conditional latent volume/inverse claims into the full data-space map.
- Primary sources read: Shazeer2020 Section2; NICE2015v6 Sections3.1–3.2; U-Former ODE2026v1 Section3.2.1 (SwiGLU CDE field, not the exact shear); Wendin/Altafini2025v1 Sections2–3 (weight-training dynamics, not an inference theorem).
- The exact ordinary baseline is retained, not weakened, and no failed scalar-potential or feedback setting is revived. Positive compatibility cannot be called a performance improvement.

## Cycle03 completed evidence
- Registration commit dfd416f. Formal PID35582 exited0 in42.33seconds; independent replay exited0 in9.37seconds. Sampled peak RSS0.331GiB on one Mac CPU thread.
-44 historical checkpoints,55 full-network-Jacobian panels and11 generated batches agree bitwise. All11 fitted full-parameter probes and the paired5000-step optimizer trajectory match native behavior; all four saved states match historical half-mixture SwiGLU128 exactly.
- Independent literal matrices at132 real FFN contexts give exactly equal values, and33 local gradient probes agree within9.54e-7 in float32. The separate literal exponential float64 check agrees in gradients to2.78e-17. All369 bindings and prior seals remain unchanged.
-132 conditional-generator singular quantile records are published; raw arrays stay private. Final Swi128 probe flow-norm maxima: Product7.93, sign-code9.65, mixture6.72. All generator eigenvalues remain zero. These are finite probe observations, not global bounds.
- User clarified experimental hosts: Mac mini and Lenovo. Lenovo SSH still times out. MBP/torch are removed as publication dependencies; GitHub CLI2.103.2 was installed on Mac mini. Browser login is pending. No secret token is printed or committed.
- The next candidate must add useful computation. The gate-transport design describes exact K³=0 coupling with a small rectangular C, a fixed-feature rank witness, explicit full gradients and matched wider-Swi controls; it is not yet implemented or selected for training.

## Cycle04 completed and stopped
- Registered8dba424. Fifteen seed0 trajectories completed (PID36784,exit0,170.20s), two one-thread Mac CPU workers, sampled summed peak RSS0.826GiB. Lenovo was unavailable; no MBP/torch/GPU execution.
- Preflight passed15 cold/resume cases; literal exponentials and complete derivatives agreed to2.78e-17. Actual16D costs: transport89016params/87210MACs; Swi14389418/87552.
- Field MSE transport/Swi143: Gaussian.0038848/.0043525, Ring.0162897/.0153686, Product.1013762/.0846540, sign-code.0411181/.0389728, mixture.1355216/.1259118. Strong reused Swi160 remains included. No16D10% gain; stop without partition/coefficient/rank/feedback rescue.
- Independent audit-v2 exited0 in13.30s:60 checkpoints,75 panels,15 samplings,25000 stream hashes,60 C spectra,120 cube-zero generators and10 historical exact streams. All345 bindings and prior seals unchanged.
- C is active: final Product singular maxima1.02,.80,1.56 and local FFN correction RMS ratios.42,.19,.47. Local amplitude is not a causal explanation of full prediction error.
- Post-outcome source-gate coefficients all have numerical rank43, but median participation ranks1.050Product/1.072sign/1.061mixture. These are fixed-dictionary linear coefficients, not full-network/Jacobian ranks or a demonstrated failure mechanism.
- Next review is unselected: evolve preactivations before integrating the activation, and compare against an absorbable midpoint reparameterization. Pointwise nilpotency need not terminate nonlinear Lie derivatives. Review old stops and primary literature before new training.

## Cycle05 pre-outcome selection
- One fixed candidate integrates SiLU along an exact square-zero preactivation path; the paired midpoint control absorbs into ordinary SwiGLU. Both preserve native Swi128 at C=0, including the complete C tangent. This changes nonlinear feature directions rather than linearly recombining cycle04's existing activated dictionary.
- Literal values, full parameter/input/C derivatives and analytic Jacobians agree to2.78e-17. Independent joint RK4 agrees with64-node integration to2.50e-16. The pointwise joint Jacobian is cube-zero yet the nonlinear third-time derivative is nonzero. A single-receiver Hessian witness has determinant1/48; no whole-network lower bound follows.
- Read Shazeer2020 §2, Biloš2021 §2/limitations, Norcliffe/Deisenroth2023 §3 and NIST DLMF3.5(v). These support the comparisons/numerical interpretation, not a new or FM-specific architecture claim. Limited targeted searches are not exhaustive novelty verification.
- Select only10new trajectories: quadrature2/midpoint across the existing five laws, with all15 historical Swi128/143/160 controls. Keep seed0 and every native training/evaluation setting. Require the fixed quality rule AND finite-bank two-node/32-node accuracy checks before any flow nomination; failure forbids node-count or other parameter rescue.
- Lenovo SSH was rechecked with an8-second connection limit and timed out. Mac CPU execution remains available. GitHub login is still absent; local main now fast-forwards to47a85cb, without claiming remote publication.

## Cycle05 completed and stopped
- Registered5cab241;10trajectories completed on Mac mini, PID37824,exit0,121.67s, two one-thread workers, peak sampled summed RSS0.827GiB. No Lenovo, MBP, torch or GPU execution.
- Candidate/midpoint/Swi143 final field MSE: Product.096584/.097085/.084654; signs.043027/.043061/.038973; mixture.132280/.132941/.125912. All stronger Swi160 and narrower Swi128 references remain visible. No quality nomination.
- Independent audit exits0,14.36s:40checkpoints,50Jacobian panels,10samplings,25000batch hashes,120C matrices,240pointwise field-Jacobian spectra,15historical streams. All466bindings and1051older sealed files match. No audit retry/tolerance change or training rerun.
- The independent32-node reference passes every fixed numerical gate: maximum gate relative RMS.00016859, field-MSE relative change.00017524, complete-Jacobian relative RMS.00035981. Probe32-vs64 references agree to2.41e-16. These are finite-bank observations, not global bounds.
- Final local two-node/midpoint receiver differences are.42–.58%RMS on Product,.51–.74%on signs,.52–1.01%on mixture despite nonzeroC. At C=0 all C tangents are exactly ordinary Wg/bg tangents, by dWg_left=.5*dC*Wg_right and dbg_left=.5*dC*bg_right. Extra nonlinear responses begin at second order; this is not causal proof of the fitting gap.
- Runtime320files sealed, SHA2562a7a5fd06458ee4578db8cb52ed8076178b4b96046c2e1832fc267db697d6885. Curated copies and visually inspected figures pass archive checks. Report1 and earlier runtimes are untouched.
- Public remote main remains800d09f, verified with HTTPS ls-remote. User was asked to authenticate gh on Mac mini directly; no other-machine relay or new device code was started. Local results remain unpublished pending that external step.

## Cycle06 frozen curvature attribution
- Registered0f39045 before counterfactual outputs. No training or coefficient/initialization selection. One Mac CPU process, PID38295,exit0,3.27s, terminal RSS0.292GiB.
- Replayed80000actual field rows/25Jacobian panels and computed25midpoint counterfactual panels at fixed learned tensors. Independent NumPy audit checks exact risk decomposition,25counterfactual Jacobian metrics and15predefined largest-change cases. Its scope is saved arrays, not a second neural-network implementation.
- Retaining nonlinear curvature changes full-bank risk by−.232%Gaussian,+.721%Ring,+.0368%Product,−.453%signs,−.104%mixture. All16D variants remain worse than matched Swi143. Tiny numerical error and this small mixed curvature contribution cannot be presented as achieved task gains.
-20complete C=0 tangent identities at nonzero fitted heads pass with maximum discrepancy1.11e-16. The ordinary gate-weight/bias tangent mapping is exact, but this does not measure the fitted nonzero-C tangent rank or establish why Adam failed.
- The largest3sign-code case effects are adverse despite a helpful average. Keep all cases and do not infer a posterior-mode mechanism from selected extremes. No next architecture or training extension is selected by these observations.
-53runtime files sealed (SHA2560d0118ab6febf04a486b18cbaee3771a5c04176c1d4911ea44629c9835faf7f1);1371older sealed files, curated copies, Python ASTs and relative report links pass verification. The plot was visually inspected. Mac mini gh still has no authenticated host; publication remains pending independently of completed local research.

## Cycle07 pre-outcome compatibility question
- Previous turn is verified progress: cycles05–06 completed, audited and sealed, local main9e53806. Current worktree was clean at revalidation. Local gh remains unauthenticated; publication is pending, not research execution.
- Re-read the unselected head review and output-envelope result, JiT v2 §3.3, Gagneux2026v3 controlled Fourier manifolds and Liang2026v2 residual-stream work. The new paper is under review. None predicts an ODE-FFN advantage. A first guessed head-protocol filename did not exist; the actual head-envelope protocol/review were inspected.
- The128D paired thin/thick Ring8 benchmark keeps a common unknown orthogonal2D latent plane and changes only target normal variance0/.04. It is additive, not a replacement for previous full-support datasets. Total target energy changes with normal variance, so no equal-energy or universal intrinsic-dimension claim is made.
- An input64D linear bottleneck erases at least62normal Gaussian directions. The resulting population field-risk floor is(62/128)*beta²*d for direct prediction and(62/128)*gamma²*d for analytic-skip heads. Final LayerNorm gives corresponding full-Jacobian floors63*beta²/63*gamma². Thin direct uniform-time floor is0.468229; hidden-P constraints/integration accuracy cannot recover erased directions.
- Dense128D Gaussian conditioning independently matches the efficient oracle to4.89e-14 in values and1.48e-13 in complete AD Jacobians. Rank-two/endpoint checks, head parameter tangents and a deterministic missing-normal-coordinate fixture pass. No training outcome was used to choose these laws or heads.
- Select18compatibility trajectories, not a new architecture: native professor K2/Swi128/Swi192 crossed with clean/shifted/direct and both laws. Same capped pair target for every head; no oracle/covariance inputs, new FFN, sampler/optimizer tuning or old leaderboard substitution. Capped-sampler bias gets an explicit oracle reference. Generic head effects cannot complete the successor goal.
- Mac mini has16GiB RAM and76GiB disk free. At most two one-thread CPU workers; expected private data/results below2GiB. No MBP/torch/GPU dependency.

## Cycle07 completed compatibility evidence
- Registeredb19bcb8;18trajectories finished on Mac mini, PID39080,exit0,331.57s, sampled summed peak RSS1.314GiB. Independent audit exits0,34.51s:72checkpoints/90complete panels/18samplings/10000stream hashes/144P matrices/two oracle samplers. All65bindings and prior seals match. Private storage about1.4GiB. Lenovo SSH was rechecked and timed out.
- Thin/clean field errors: ODE.00272857, Swi128.00299853, Swi192.00309514. The9.0%/11.8% advantages are retained, but do not meet10% against each control. Thick/clean ODE tangent error.353135 vsSwi192.287896 fails the10% guard. No nominee or head/cap/dimension rescue.
- The generic head effect is large and theory-consistent: thin/direct errors~.505 versus~.003clean, population rank floor.468229. Thick field ordering reverses (~.38direct vs~.70clean), yet direct generation has much larger normal spread. Keep full128D and2D tangent metrics separate.
- Generated skip-model samples largely form continuous rings rather than eight resolved clusters. Original ODE thin/clean fullSW2.02966 but tangentSW2.24985. Oracle Heun32 resolves clusters and retains normalRMS.0212 due the cap/grid; continuous cap predicts.01839. No distribution-recovery claim follows from the small ambient field MSE.
- All144P histories/9216singular values and clock probes are curated. Thin/clean sigma maxima1.15–1.44, CV.62–.66, mean clocks.607–.746. These are not whole-map stability guarantees.
- A new unselected mathematical review derives a final-readout affine-subspace invariant: raw output lies in c0+range(W diag(gamma)C), rank<=63. Direct sampling preserves source variance in its left null space; clean/shifted contract it with an exactly computable Heun factor. This has not yet been verified in actual fitted null-space coordinates. It selects no new head or method; fullW2 bounds must not be confused with empiricalSW2.
- Uniform-time integration of the proved field bounds yields thick/direct floor.351341 and thick/skip floor.620008.64/128-node split quadrature agrees within1e-11. These are population/time-average bounds, not finite-bank or five-fixed-panel Jacobian bounds.
-584cycle07runtime files sealed (SHA2562638989d5b46798e4150d933140ad439c2ba62ea3dc2514e4a88cfd92e83e0b4). All65bindings/1424older sealed files, curated bytes, Python ASTs and report links pass. Four figures were visually reviewed; generation-panel spacing was corrected so both normal-RMS rows are readable. Report1 is unchanged. Mac mini gh remains unauthenticated.

## Cycle08 pre-outcome registration
- Revalidated cycle07 at019e676 with clean worktree. Previous goal turn is progress;18models/audit/seal are present. Local gh remains unauthenticated, independently of research execution.
- Select only a read-only invariant verification:18cycle07 endpoints, six fixed zero-P interventions and two old2D/16D readout-rank controls. No training, head/solver changes or quality-based selection. Checkpoints are hash-bound, including old controls' original cycle05 reference digests.
- The proof uses raw=B*z+c0 with B=Wdiag(gamma)C, rank<=63. Projection onto ker(B^T) annihilates every hidden operation. Standard-Gaussian projected variance is preserved by direct prediction and shrunk by the same head/solver-dependent factor for both skip heads. Target-normal intersection dimension>=63 gives a fullW2 lower bound, not an empiricalSW2 bound.
- Small literal nonlinear-flow/Heun fixtures pass to5.78e-15, including zero-gamma/bias and full-row-rank cases. Formula/scalar Heun coefficients agree to1.39e-17. Fixed Heun32 skip factor.02116394043; continuous.01839397206. Source-wise tolerances are registered before fitted null-space outcomes.
- The theorem assumes a frozen finite predictor, positive normalization epsilon, local state Lipschitzness and standard-Gaussian source. Final raw output is bounded; capped skip has linear growth, ensuring global finite-horizon solutions/moments. It does not automatically apply to older full-row-rank outputs or images.

## Cycle08 verified readout restriction
- All18fitted readouts rank63, null65, common target-normal null63. Source-wise sampling errors<=9.21e-7; covariance discrepancy<=2.29e-7. Independent Householder/QR projectors match to2.51e-11.
- Six interventions zero only hidden P. Whole-sample RMS changes0.108–0.355; null-coordinate max changes<=1.30e-6. P is active but cannot repair fixed-readout directions. No zero-P quality comparison or selection.
- Thin/skip population fullW2² floor≈0.02822; thick/skip≈2.015; direct≈63thin/40.321thick. These are unnormalized populationW2², not empiricalSW2.
- Actual olderRing2D/Product16readouts have full row rank. No claim that this theorem explains old negative findings.
- Formal PID40649 exit0,6.57s; independent audit terminal log/receiptPASS,6.00s; process absent, original handle lost during observation truncation. No audit exit code asserted and no restart.
- Next possible read-only question: exact conditional-mixture oracle after each learned input projection, separating information loss from fitted residual error. Not run or a new method nomination.

- Cycle08 archive sealed61files; SHA256384d9b6974dddd969bd48509c9f889763d5e5a08b41585f2be543ce53fd2d650. Curated audit files match exactly; Python AST and relative report links pass; one PNG/PDF figure visually reviewed. Prior2008sealed files and9bindings match.

## Cycle09 pre-outcome conditional-mixture derivation
- For orthonormal observed rowsB and timea=1-t, conditional component covariance isD=a²I+t²B Sigma Bᵀ. Exact joint-Gaussian conditionalY/Z means lead to decoder-specific best fields.
- Skip heads share the same decoded conditional optimum at fixedB; direct conditions the pair velocity itself. Population cross term vanishes by tower property, while the finite-bank cross is retained.
- Remaining residual includes output capacity/approximation/optimization. No causal P attribution, no renewed head training, no stopped setting is reopened.
- Independent dense-conditioning fixture maximum9.06e-14; full-observation/native-oracle maximum1.78e-14. No fitted conditioned outcomes inspected before registration.

## Cycle09 audited conditional input attribution
- Formal72checkpoint study PID41562 exit0,14.15s,0.408GiB; audit-v2 exit0,17.73s. QR/Woodbury fields agree to1.13e-13, all72per-row decompositions/1296summary rows and18native endpoints replay.
- Thin/clean ODE input component3.1646e-5 vs total.00272857; Swi1284.5693e-5/.00299853; Swi1924.2554e-5/.00309514. Residual explains95.16%/97.63% of known gap, input5.20%/2.98%, finite cross-.37%/-.61%. Predeclared classification: predominantly remaining residual.
- Thin/clean tangent residualsODE.17078/Swi128.18730/Swi192.19372. Largest fixed-time-bin residuals in.6–.8; ODE loses early and wins later. No loss/time-distribution change selected.
- Thick/clean tangent residualODE.35134 vsSwi192.28473; inputI.00186 vs.00285. Transfer failure cannot be attributed to ODE losing more tangent input information.
- Residual includes output/representation/optimization limitations and does not causally identify P. Cycle07 failed thresholds and all earlier stops remain; no useful new ODE model is established.

- Cycle09 sealed108files, SHA2568e572acd096f1310fe2ff2cab337342c981a6fcfa4d265a1a34cda3a94e984f5. All8bindings and2069prior sealed files match; curated audit copies/relative links/Python AST/figure review pass. Private runtime126.3MB, curated evidence0.46MB. GitHub main still800d09f; auth check remains unauthenticated.

## Cycle10 pre-outcome derivative derivation
- Exact posterior clean Jacobian J_m=t/a² Cov(Y|U) Pi. Its skew part is t/(2a²)[Cov,Pi], generally nonzero under a compressed input. Penalizing all model skew can penalize the field-optimal reference itself.
- Direct conditional field equals((aI+tPi)m-Pi*x)/delta; skip reference equals(m-x)/delta. Derivative information/residual/cross decomposition has no population orthogonality claim.
- Float64 pre-registration fixtures pass: analytic/independent denseAD1.05e-12, finite differences7.55e-9, full observation/native oracle3.30e-12, commutator4.45e-16. No fitted conditional derivative outcomes viewed.

## Cycle10 complete-field derivative outcome
- All90existing final panels analyzed; formalv2 PID42271 exit0,34.22s,0.410GiB. Audit exit0,17.95s:5760nativeJacobian anchors bitwise,180denseAD conditional anchors max2.49e-14, independentfields6.27e-14,96formal files unchanged.
- Late clean tangent residual symmetric shares: thinODE99.9687%/Swi12899.9786%/Swi19299.9743%; thickODE99.8050%/Swi12898.9898%/Swi19299.9592%. Skew-only diagnosis is unsupported; these are derivative-energy shares, not field-risk/sample-quality attribution.
- ThinODEtime.7residual18.7312 vsSwi19219.9302, buttime.9ODE7.1397 vsSwi1926.7279. ThicklateODE14.2298 vsSwi19213.5848. No uniform derivative win or new method.
- Fulloracle tangent expansion counts bytime.1/.3/.5/.7/.9 are0/64,64/64,49/64,4/64,0/64 for bothlaws. Complete decoded fields sometimes need positive local eigenvalues. HiddenPconstraints remain a different question.
- Registration9517070 fixes actual storedfloat32 paneltime before fitted outcomes; initial5af02d7preparation/preflight-only snapshot is preserved. Neither dataset nor tolerance changed.

- Cycle10 sealed135files SHA256e8c8278b77ce7fd6550c0cd684936b56c1e61bf62963efc8cf11d3449349c9f6;7bindings/2177prior sealed files match and10initial preparation-only files are copied/verified. Private728.1MB/curated0.29MB. Source AST/relative links/exact audit copies/figure review pass. GitHub remains800d09f, local gh unauthenticated; connected create_commit still has no author override.

## Cycle11 pre-outcome conditional-frame design
- One input-dependent orthogonal similarity QPQᵀ around each unchanged originalK2 polynomial. Q=H([1,g])H(e1), g=zero-initialized affine readout of existing clock-controller features; no new activation, clock range, loss or sampler. Fixed conditional spectra/norms are preserved, full-map stability is not implied.
- Complete differential includes commutator[Omega,A]x and clock derivative. WhenAis symmetric, the frame commutator is symmetric; the design can change stretch directions without assuming the skew residual dominates.
- Householder weight adaptation/input-conditioned reflections/isospectral ODE parameter flows have prior art (HOFT2025,DeltaProductNeurIPS2025,HouseGNN2026preprint,ODEtoODE2020). No exhaustive novelty/FM benefit claim. Global frames absorb intoP; candidate equals an ordinary rank<=4 conditional operator update.
- Independent small-layer prototype passes: all-output active state/parameter gradients4.45e-15, SV4.45e-16/eigenvalues2.34e-15, full commutator differential1.78e-15, global absorption2.23e-16. Zero-frame values/inherited gradients and RNGexact. Local response rank4 plus scalarclockrank1 spans5in the explicit5D witness; not a whole-model capacity theorem.
- Measured128D parameters/inference/backward: frameK2(122880,145472,426112),globalK2(122880,145472,401920),frameK1(122880,120896,352384),Swi257(148098,145664,426688). Swi257 is the smallest nominal rounding above candidate dense work. Global inference/params match but backward is lower, explicitly retained. No quality outcomes inspected before registration.

- Before fitted training, cycle11 initial preparation/cache and verifier-namespace issues were corrected in new snapshots. V3preflight then exposed float32 identity-backward sign-cancellation max5.82e-11; algebraically equivalent rank-two residual Q implementation restores exact whole-model inherited gradients on both laws/modes. No tolerance, parameterization, initializer, hypothesis or fitted result changed. New active derivative maximum5.11e-15; all failed preparatory sources/logs remain private.

## Cycle11 trained and audited result
- Formal PID43415 exit0,284.07s, two CPUworkers, peak sampled summedRSS1.349GiB. Audit-v2 exit0,38.56s:32new/16reference checkpoints,40new/20reference panels,8new/4reference samplings,10000stream batches,144P matrices,36two-anchor dense-frame checks;235formal files unchanged.
- Thin frameK2 field.00068508 vsoriginal.00272857/global.00274769/Swi192.00309514/Swi257.0032595. TangentMSE.0428035 vsSwi192.196558. Visible eight-cluster recovery replaces the original/Swi rings, with residual bridges/shape errors; capped normalRMS≈.021 stays.
- K1frame field.00071312 and tangentSW2.19501 vsK2.20410: K2only3.93%field advantage, failing fixed5%mechanism gate. Thick frameK2 field.700198, tangent.282518 and full/tangentSW2.06366/.31734; betterSwi.697991/.255320/.054845/.247082. Thick sampling/transferguards fail. Decision stop_this_frame_setting, no historical wave/rescue/nomination.
- Frame-zero replay changes outputsRMS.364thin/1.303thickK2 and raises risk.13168/2.36076, preservingP/all non-frame tensors. This is joint-fit dependence, not a retraining result or proof of an ODE-specific effect.
- Last-frameK2 rawP sigma_max1.173–1.327thin,1.126–1.800thick; CV.614–.656/.623–.689. Stateframeg varies acrosssamples. Spectrum preservation acrossinputs does not imply same spectra across trained models or full-map stability.
- Cellend-to-endCPUtime includesdiagnostics: K2frame75.5/84.3s vscompute-matchedSwi25737.2/46.1s. Pointwiseoverhead makes denseMACmatching an incomplete wall-time comparison.
- Next separate question: signed propagation of field defects along frozen model trajectories through the unchanged oracle Heun suffix; no new training/head/cap/order/frame setting is selected.

- Cycle11 seal317files SHA256392371988605176496b8af7b6876751a7370a911adf40e709a44d1c95b922449. All12bindings/2312prior files match;144derived full spectra/132training log rows match source exactly. Fourfigures visually reviewed, sourceAST/links/auditcopies pass;33preparation-failure files retained. Private592.3MB/curated1.61MB. GitHub remains800d09f pending local authentication.

## Cycle12 pre-outcome discrete sampling derivation
- p_i=oracle_suffix_i(model_prefix_i(source)); E=p_32-p_0=sumDelta_i exactly. Signed contributions<E,Delta_i>sum to pairedMSE, retaining cancellation. This coupling-dependent error is not SW2 or a distribution-quality attribution.
- Oracle normal field is beta(t)Nx; exact scalar Heun suffix G_i transports each normal defect. A clean-head output-null/target-normal projectorT also predicts every bridge: p_iT=G_i[R_i sourceT+(1-R_i)c0T]. This separates a fixed-readout forced component from the normal remainder without excusing every relative model difference.
- Float64 nonlinear-mixture fixture vector telescoping1.09e-19/normal affine5.94e-16, forced-readout bridge3.89e-16. Signed cancellation, expanding suffix, correct126normal normalization and nativefloat32 sampler arithmetic pass. No new fitted prefix/suffix outcomes viewed before registration.


## Cycle12 audited path defects
- Previous host-clarification turn was no progress toward the research goal; current work resumes the already completed formal diagnostic, without restarting it. Fresh Lenovo SSH check times out; Mac mini remains available. Local GitHub auth is absent; no MBP/torch relay is used.
- Registered7a6c2db; preflight14.74s, formalPID44962 exit0/126.32s/1.340GiB, auditexit0/61.88s. All12endpoints/384learned steps/396oracle suffix branches replay bitwise, signed source projections checked;21formal files unchanged. No new training/solver/cap/head/loss.
- ThinframeK2 tangent signed shares20.3846/49.6867/28.9215/1.0071%; thick22.8275/38.6117/22.6290/15.9318%. Large late-time uniform-bank residuals do not locate endpoint discrepancy. Finite tangent gain maxima16.77/16.52 refer to external oracle suffixes, not hiddenP stability.
- Every fitted target-normal/output-null intersection is63D. Fixedreadout bridge identity error<=2.89e-7, normal scalar propagation<=6.41e-7, independentQRprojector<=7.42e-12. Thick forced fraction91.53–93.15% acrossall6families, sharedrestriction notexplanationofeveryrelativefailure. Clean homogeneousfactor.02116394 versus thickoracle.19701598.
- ThinframeK2 paired tangentMSE.165753 exceedsoriginal.158101 while tangentSW2.204103 improvesversus.249849; centroidassignmentdisagreement11.72%versus8.45%. Paircouplingmetric mustnotreplace distributionquality orbecomeanewobjective.
- Atthicktime.875,frameK2 learned-pathfirst-stage tangentfielddefect1.57415 versusoraclepath.31601,normalRMS.17595versus.21601. Visited-state sensitivity,notisolatedcausality.
- Strictanalysisreport/statisticsappendix/figurecatalog written,fourPNG/PDFfiguresvisuallyreviewed.48reporting-onlyconditionalMCrows havefixedpredictor/IIDpointwiseapproximations,notselection-adjustedtests.12negativeaggregatestepcontributionsretained.NoK2rescueornewnominee.

- Cycle12 sealed77files SHA256c5c17339156af984d03edaea508bd4a007aaad2a9719decdd1a2d1872c02675b. Sevenbindings/2629priorsealedfiles/exactauditcopies/48conditionalprecisionrows/78relativeMarkdownlinks/sourceAST/fourfigurechecks pass. Private1208639308bytes/curated1146453bytes; allprotectedmodel/evaluator/Report1paths unchanged.

- Cycle12 final milestonee228381 committed underZhizheZhang, localmainfast-forwarded. Directpublisher session45821 exited1 aftersuccessfulfetch/prepushauthor/ancestrychecks: HTTPSUsernameunavailablewithterminalpromptsdisabled. No remoteupdateclaimed,noliveretryprocess. Publicationpendinglocalauthentication,separatefromnextresearchreview.


## Cycle13 registered question and mathematics
- Revalidatedclean1329914/cycle12sealed77files. Previousgoalturnmadeprogressbystrictanalysis/archiveandsynchronization; currentgoalcontinuationpreservesthiscourse,notanewtask.
- ExactlocalK2incrementd=s*u+.5s²*w,u=Pz,w=P²z. BestfreeK1scalaralongv is<v,d>/||v||²; originalboxusesclip[.2,1], includingclosedendpointinfima. FixedA=P+.3P² absorbsnativec=.6; sameclockremainder=.5s(s-.6)P²z. Boundsarefrozen-Pper-inputrelaxations,notfinitecontrollercapacityorretrainedK1results.
- EqualSVDvaluesdonotremovequadraticdirections: Iand90degreerotationhaveidenticalsingularvaluesbutorthogonalinnovationfractions0and1. SymmetricnarroweigenvaluebounddoesconstrainP²-lambdaP. No flatteninginterventionselected.
- Smallalgebra/grid/degenerate/boundaryfixturespass; liveframe/clockfullADmax4.44e-15. Actualnativeevaluationchunk1024verifiedfromsealedpriorauditor,asdistinctfromtrainingbatch512. Cohort24checkpoints,144layers,all20000fieldstatesplus32-rowclockreplays,steps0/1000/2500/5000,two laws/threeK2families.
- Primaryliteraturereview: AREX2610.03483v1isanexternalsamplerwithdata-momentaffinebackbone; Koopman2506.22304v2usesateacher-consistentlift; neitherjustifiesfreehiddenPpolynomialorders. Al-Mohy/Higham2011isclassicalmatrix-actionprior,notafrontierrecommendation.
- Pre-registrationdiscoverycorrectedguessedoldfilenamesand512-vs1024inferencebatchbeforeanyfittedenvelopeoutcomes. Earlierstops/immutabledata/Report1remainunchanged.


## Cycle13 execution and scientific outcome
- Registration35cba88; preflight7.71seconds, formalPID46900exit0/78.04seconds/0.768GiBsampledRSS, independentauditexit0/58.67seconds/0.848GiB. Nativefields/hiddeninputs/substitutedfieldsbitwise;144layers/2.88Mscalarrows/1.92Mdenseframerows checked, maximumindependentsampledifference1.14e-12;56formal files unchanged.
- Rawquadratic/incrementrelativeRMS14.08–34.69% atfinalsolves. UnchangedPbounded-clockresidual12.27–33.85%; afterfixedA=P+.3P² only8/36finalsolves below1%, predominantlythininputframe. No universalquadraticredundancy.
- Same-clockcentredintervention: thinin-frame tangentdisplacement.44%, tangentrisk-.62%; thickoriginal tangentdisplacement21.71%,risk+25.17%,despitefullrisk+.40%. Thickglobal tangentrisk+36.69%. FrozenfunctionalroleisnotaretrainedK2advantage; K1mechanismcontrolandallstopsremain.
- Tangent/normal reporting split was added afterregistration toavoidnormal-dominatedfullratios; same saved predictions, no newintervention,thresholdorsampler.72exactrisk-accountingrows; no p-values ormodelwinner.
- Threefiguresvisuallyreviewed; strictanalysis/statistics/catalogwritten. Future order-specific rationale must accountforfixedmatrixreparameterization,strongretrainedK1/SwiGLU,andseparatetangentdiagnostics. No new architecture selected; no spectral flattening or order rescue.

- Cycle13 archive112files, SHA256c1a7dd5187106378bb1156645e15c3adcae24253061823c08c5327e797f20b91. Eightbindings/2706priorsealedfiles/144P-clockrecords/72projectedriskrows/67links/AST/threefigurechecks pass. QRnormal-coordinate replay avoids cancellation in cold near-exact normal fields; no tolerance or formal result changed. Private1368991282bytes/curated708878bytes. Fresh gh remains unauthenticated, LenovoSSH times out; no MBP/torch used.

- Verifiedcycle13milestonea991af0 committedunderZhizheZhangandlocalmainfast-forwarded. Directpublisher67397 exits1 atpush (missingHTTPSUsername;terminalpromptsdisabled), remote800d09f verifiedread-only. No job restarted, no remote update claimed; sealed runtime unchanged.


## Cycle14 independent coefficient: pre-outcome review
- Previous turn is progress through the cycle13 diagnostic, archive and synchronization. Revalidatedclean e608c9c. Old external-time adapters, constant-duration effective maps, midpoint feedback and input frames do not implement one independent internal second coefficient. All their stops remain.
- Proposed Cjet is x+sPx+.5s²(P²+C)x, Cstarts0. It is a second Taylor jet for z'=(P+uC)z; B=P²+C is a bijective ordinary gated-polynomial coordinate change. The direct-B control has the same real-arithmetic class and fewer matrix products; no ODE-specific novelty or solver-accuracy advantage is assumed.
- Matrixfamilycommutator=.5sr(r-s)[P,C], completeclockgradientsretained. Small activefixturespass2.22e-16; C-familyvalues/inheritedgradients/RNGbitwise. DirectBfloat64equivalent, float32initialvalues/gradientdifferencesexplicitlytestedratherthanclaimingexactpairing.
- Actual128Dwholemodelcosts: C/affine122886params/145856inference/427264training; free123276/146240/428416; directB122886/121280/353536; Swi258148680/146240/428416. FreshSwi258coversfreecontrolwork; oldSwi257isslightlybelowCbudget. Tennewcellsandfoursealedoriginal/Swi192refsselectedpre-outcome.
- Calibration source/log preserved. Scalar reporting warning removed via detach before frozen registration, no numeric/tolerance/initializer change. Generator B is labeled a fixed reference for affine/free controls; it is not their actual input-dependent second coefficient. P/B/C diagnostics form float64 matrices from native stored tensors; activation telemetry remains native.


## Cycle14 trained independent-coefficient result
- Registration19af9a8; all10preflightcells pass inherited C-family gradients, active nonzero-head checks and exact native three-update/resume. Direct-B float32 value differences2.38e-7/4.77e-7 and output-head gradient differences3.73e-8/4.47e-8 are explicit, not matched-kernel claims.
- FormalPID47790 exit0/234.80s/twoCPUworkers/1.240GiBsampled totalRSS. Auditexit0/44.13s,40new+16refcheckpoints/50new+20refJpanels/10new+4refsamples/10000streambatches/192P-Crecords/6144fittedlocalJacobiananchors;293formal files unchanged. Local maxvalue3.55e-15/J9.99e-16/Pgrad1.07e-14/coefficientgrad0.
- QuadraticC thinfield.0025979 versusoriginal.0027286/Swi192.0030951/Swi258.0030242: modest14.10%vSwi258. Thick tangent.3347408 versusSwi192.2878964(+16.27%); tangentSW2.3027826 versus.2470819(+22.54%). Fixed guards fail. DirectB also fails.
- QuadraticC geometric-mean tangent ratios.990919vaffine/.997564vfree fail.95mechanismgate. Extra C is active (thickaddedRMS/input.086–.535), commutatorsnonzero, freeclockr-sRMS.032–.093thick. Activity/noncommutationdoesnotimplyusefulness.
- Direct B has the same real-arithmetic class and fewer matrix products, but worse finite-budget endpoints. A separate post-result gradient-flow coordinate-metric addendum/fixture explains why equivalent classes can have different update directions, without attributing Adam's outcome uniquely or selecting a new optimizer.
- FreshSwi258 is worse than reusedSwi192onthick; both referencesretained. StrongerhistoricalSwi257fromsealedC11audit-v2iscontextonly,no post-outcome gatechange. Samplesstillmostlyannuli,not8modeclusters. Fourfiguresvisuallyreviewed; all192wideP/Bref/Cspectraand154windowedtelemetryrowscurated.

- Cycle14 sealed366files SHA256182a3daa0a237b1d2c7e6b509005869f6002a25c15649668772e24d2069b8ac7. Fourteenbindings/2818priorsealedfiles/192derivedspectra/154trainingrows/exactauditcopies/77links/AST/fourfigurechecks pass. Threeexecutedcalibrationsources preserved; private740654995bytes/curated2161726bytes. Native model/evaluation/Report1 unchanged; only coefficient_jet.py was added to production source. GitHub tool capability rechecked: create_commit still has no author override; local gh remains unauthenticated.

- Verifiedcycle14milestone51fedd5 committedunderZhizheZhang/localmainfast-forwarded. Directpublisher11320 exited1 atpush: HTTPSUsernameunavailablewithterminalpromptsdisabled. Remote800d09f reverified; no remote update claimed. No formal/audit process restarted and sealed366-file runtime remains unchanged.


## Cycle15 question, before fitted gradient outcomes
- No study process is live. Lenovo SSH timed out again; only Mac mini/Lenovo remain authorized. Current clean checkpoint is cd09851.
- Existing cycle14 window telemetry has zero clipping in thin final windows but63.4–93.8% in thick. A large readout-null residual cannot directly drive hidden P at a fixed head; indirect clipping/moment effects require measurement.
- The next bounded diagnostic is frozen next-batch gradient routing, not a new head/loss/optimizer architecture or a rescue of stopped settings.

- Cycle15 formal49158 exited0 in35.58s,88checkpoints/1.76Mnativefieldrows/440groups,0trainingupdates,0.429GiBsampledRSS. Audit-v1 stopped at a cold thin cosine: remaining-normal route norm4.92e-15 makes direction meaningless. V2 keeps gradient tolerances and independently validates raw routes, then recomputes reported ratios from the saved vectors. No formal rerun, source binding mutation or threshold change.

- Cycle15 audit-v2 session55421 exits0/29.95s; all88nativefieldreplays/264gradientroutes pass, maxindependentgradient1.26e-15/projector9.06e-12.181formal files unchanged. No formal rerun and no fitted optimizer step.
- Thick final next-batch forced-loss fraction72.79–73.13%, hiddenforcednormmax6.20e-16. Step1000hiddenupdateeffects4.44–23.77%, step2500.41–5.45%, finalnonebecausebatch5001doesnotclip. Existing windowcliprates and singlebatchrates differ; no endpointcausalclaim. FinalthickPcos.089–.234reinforcesratherthancancels; no head/loss/optimizerrescue.
- Archive32847 exits0,225files seal e36f77d3b56196069816003687cec9835c5887fc091072c3af3540edf9960a71.10bindings/3184priorfiles/allcuration/AST/links/twofigures pass; private470497166bytes/curated458795bytes. Source/evaluation/Report1 unchanged. LenovoSSH timesout; MacminiCPUonly. gh authentication still absent. No useful new ODE architecture is nominated.

- Cycle15 milestonea7d1c76 committedunderZhizheZhang and localmainfast-forwarded; bothclean. Final225-file sealrecheckpasses. Directpublisher8587 exited1 atpush because HTTPSUsername is unavailable (Device not configured). Independent ls-remote still800d09fe08f38ecf132f2f06b707a829426a2cd8. No remote update claimed; no task process restarted.


## Cycle16 design review, before new training
- Clean8f459e5 and completed cycle15revalidated. Primary Hyper-SET and basin-preserving Hopfield papers do not supply a new hidden covariance mechanism: Stage9already tested it and explicitly stopped higher-order/temperature/atom rescues. Reject proposed softmax-field K2 before implementation.
- Complete-gradient readout and whole-vector recurrence are also stopped. They do not test token attention. The next necessary coverage is a small actual image/JiT baseline bridge, not a rescued covariance, frame or independent-Ccandidate. No new architecture selected.
- LenovoSSH fresh attempt times out. Macmini has Torch2.10/MPS and PIL, no torchvision/timm/torch-fidelity in the frozen synthetic environment. Use existing dependencies for feasibility before choosing any quality run.
- Canonical CIFAR-10page confirms60000RGB32images,50000train/10000test, PythonarchiveMD5c58f30108f718f92721af3b95e74349a. Real data provenance is explicit; classes will be discarded for unconditional training. JiTpaper inspected at arxiv2511.13720v1, not treated as evidence for small-CIFARODEgains.

- Cycle16 preflight-v1 handle14271 consumed exit1 for settings-shadowing before timing; immutable source/bindings retained. V2registration22d39ad; preflight91300/PID50045 exits0/31.13s. Four CPU model+moment resume checks bitwise, Jacobian max2.22e-16, all common tensors/buffers paired. Eight CPU/MPS calibration rows; no CIFAR quality updates.
- New image env installation1059 exits0, original synthetic env untouched. Official serial download61679 deliberately terminated143 for throughput; prefix preserved. Eight-range continuation54825 exits0/170.39s with full officialMD5,170498071bytes, train45000/val5000. Independent preparationaudit18685 exits0, all50000training images redecoded and pixel-identical, test not decoded.
- Semantics33414 exits0: current legacy condition changes local output.03707 and leaves input-controller unused; literal adapter ignoreslocalc and activates input-controller. Separate NxD witness66793 exits0: N=D=64 routesc as token scalars and bypassesbothlegacyMLPs; adapteravoids. These are current-source observations, not historicalcheckpointattributions. Source/model/Report1 unchanged.

- Cycle16 preparation archive exits0:92files seal c1f13c9a91b1599439670918315dc891bdce38adce86dfc4ba775bdb6c3d7145;23sourcebindings/8closurebindings/225priorC15files/links/AST/2figures verified. Private805174bytes, data324500294bytes, serialprefix preserved separately. Mainrequirements and fullenvironmentlock curated; no real CIFARtraining and no newmethod nominated.

- Verifiedcycle16preparation milestone8baec48committedunderZhizheZhang/localmainfast-forwarded. Directpublisher22601exits1 atHTTPSUsername; independent remote main800d09fe08f38ecf132f2f06b707a829426a2cd8. No source/runtime rerun or remote update claimed. No quality-training process is live; the next authorized phase is its fixed protocol and ordinary-control image screen.


## Cycle17 before image quality outcomes
- Clean98730e0/cycle16seal and no quality process verified. LenovoSSH timesout. Four-arm fixed image baseline bridge selected; original K1/K2 and Swi86/129 only, not a new architecture or failed-family rescue.
- run-experiment skill applied. Initial metrics env fails SciPy1.15.3 Mach-O loader on macOS27; source/spec/log/environment preserved. New specdc729a50 uses locally verified SciPy1.18.1 in separate .venv-image-metrics-v2. Training env remains unchanged.
- Parent actual trainingCPU and metricsMPS witnesses exit0; fresh-agent image_env_doc_pass independently executes both documented commands verbatim, exit0/no mismatches. Inceptionweights95628359bytes SHA2566726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2.
- Metric fixture84046 exits0: exact empirical low-rank FID agrees with covariance-formula fixtures (singular case difference2.13e-7); identicalfeatureszero, byte roundtrip exact. CPU/MPS pool3max5.72e-6/RMS5.72e-7 under fixed1e-3/1e-4gates. No CIFAR predictor trained or generated samples observed.

- Cycle17 final registrationcafa5af, active runtimeoct28-cycle17-v2. Parent51294/session39868 launches children51302/51303forODEK1/K2, Swi86/129queued. Live process/status/resource records verified. Metric reference64923exit0/26.78s; v2reuses byte-identical features after hashes. Fresh-agent trainer doc pass exits0, physical checkpoint/RNG/buffer resume bitwise; coldHeunfactor.0211639404296875. ActualFFNtrainingMACs K1/Swi86=49536,K2=74112,Swi129=74304.

- Postprocessor session9920 waits on original parent51294 and verifies process creation time/terminal child exits. Its separately hashed frozen sources will run fixed metrics and checkpoint audit, then independent feature/FID audit once. No automatic training retry. At546s K1completed5000+samples, K2at5000sampling; SwiGLUs stillqueued.

## Cycle17 completed image evidence
- Training session39868 consumed exit0: four final5000-update models, 1,107.813s total, Mac CPU two one-thread workers. No MBP/torch. Training audit verifies44checkpoints,80panels,48P,240clocks,all4096generated images and5000paired batches while preserving223native files.
- Original postprocessor9920 exits1 at string/Path conversion in feature audit before numerical work. Its source/log/status remain intact. Separate path-only audit-v2 session68967 exits0; all6144features bitwise, separate SVD FIDs agree within4.3e-13,6metric files unchanged. Terminal recovered status is separate from old failure.
- WeightedMSE K1/K2/Swi86/Swi129=.17639285/.17684240/.17292114/.17123427; FID184.9081/183.3952/195.4619/194.1005 with1024generated/2048reference. Lower ODEFID coexists with worse regression. K2 fitting/order gates fail; no useful successor claim.
- Same-score post-result FID accounting: K2 vs Swi129 improves feature-mean term11.0448 while covariance term worsens.3395. No diversity, pixel-mean or causal mechanism claim. Preserve all P spectra and active quadratic telemetry, which do not by themselves prove usefulness.
- All five curated figures visually reviewed: final metric comparison, learning curves, P/clocks, same-source generation, first held-out denoising case. English report/statistical scope/catalog written. No model, optimizer, loss, clock, sampler, metric or acceptance threshold changed after outcomes.

- Cycle17 archive session73203 exits0. Seal392files SHA25662d178c4e2b88780495c9ea45c6aeef96453ce7098a8aeee84076cf58f081172; private805570107bytes/curated1193604bytes. All71input bindings,3original postprocess sources and separate v2audit source unchanged. Every public P/clock/telemetry row matches native replayed records.71local links and sourceAST pass; src/synthetic/configs/scripts/Report1 unchanged versus98730e0. Runtime is now read-only.

- Cycle17 a267210 committed and local main fast-forwarded. Publisher20200 exits1 (missing HTTPS Username), remote main independently800d09f; no remote publication claimed. Fresh Lenovo85331 exits255/SSH timeout. Cycle18 selected as a read-only all-model/all-source sampler-resolution diagnostic; it cannot rescue cycle17 fitting/order failures.

- Cycle18 prior turn only confirmed compute scope; this continuation revalidated files and advanced the evidence. First native32arm replay takes32.96s, before finer outputs. A separately bound pre-outcome resource addendum raises only serial sampling-audit wall ceiling600to1200s; original prepared source/protocol retained. Scientific metrics, tolerances and cohort unchanged.

- Cycle18 prepare80542 exits0; preflight89758 exits0 in129.1s, all4x1024native32outputs bitwise, scalar zero-clean recurrence and constant-velocity fixtures pass allthreegrids. Warm installed packages match bothspecs/MPSavailable. Standalone5e2c320andresourceamendment100f322committed before any finer outputs.

- Cycle18 controlled pipeline launched from registered9ea74f7, tool99684. Sampling parent52386; K1/K2children52387/52388 verified live, matching process creation times and15.7CPU-seconds each at15s. The separate controller snapshots its source/amendment SHA, commands, process resources and terminal codes; no restart. It will independently replay allfiner samples, then features/FID.

- Cycle18 sampler parent52386 exits0 with all4children0,481.075s; native N64/128 banks preserved. Controller99684 consumed exit1: audit52723 fails importing SciPy before making an output directory or numerical replay. V2 only defers SciPy to the FID function; immutable original audit/source/log/status retained. Register separate v2source hashes plus all existing sampler outputs, and a one-pass recovery controller. No training/generation retry, environment installation or tolerance change.

- Bounded primary reading: JiT v1 Sec3.2/4.3/Table9/B.1 explicitly adopts.05cap; NeurIPS2025High-OrderFM pages1–8/Eq3.3 concern external derivative consistency. OpenReview challenge and browserPDFtimeouts bypassed with officialPDF pdftotext, excerptSHArecorded. Gaussian acceleration and cap/path distinctions already appear in canonical complete-field-constraints.md; link them rather than claim novelty. New closed zero-data normal example gives terminalscale.05/e=.01839397 and variance.0003383382 despite perfect clean regression. D=-y increases regression risk yet reduces squaredWassersteinto a^4; this is a scalar counterexample, not a trained FFN, imageFID explanation or proposed fix. Numerical identity/AD fixtures pass.

- Recovery tool69697 verified live at audit52792 with matching creation time; K1 N64/128 and K2 N64/128 raw arrays all bitwise independently replayed. Original controller/status remain FAILED; no overwritten failure or resampling. All8finer banks completed before recovery; metric conclusions pending ordinary controls and feature audit.

- All8finer sample banks pass independent bitwise replay: audit52792 exits0/765.135s,20sampler files unchanged. Original pre-outcome audit ceiling1200s respected. Recovery69697 moves to the unchanged MPS metric producer53149; final feature/SVD check remains pending.

- Cycle18 recovery69697 consumed exit0. Samplingaudit765.135s/20sampler files unchanged; metrics110.242s; featureaudit188.166s/all12288generated+2048reference features bitwise/all12FIDs maxdiff6.54e-13/30filesunchanged. No ranking reversal: at128 K1/K2/Swi86/Swi129 FID185.37154/183.69807/193.89102/192.01794. K2-Swi129 lead10.70533to8.31987; finer sampling raisesODEscores and lowersSwiGLUscores. K2fitting/order failure remains.
- Curation76595 exits0. Every registered score/contrast/raw difference retained,3figures visually reviewed, English analysis/statistical scope/catalog written. Successive raw RMS-change ratios4.917–5.171 are descriptive, not a global-order proof. K2max32to128coordinatechange.26366 vsSwi129.05417, so lowerFIDdoesnotproveuniformlowernumericalsensitivity.

- Cycle18 archive90796 exits0:116files seal0a6fb24d1aba029be90e95f7c74919bbbdead973c101819852f04cb55cc64108; private170067471bytes,curated1047106bytes. Parent392files/source7bindings/separatev2andformal sampler manifests/controller hashes/77links/protected scopes pass. Extra closing read checks recovery receipt hashes equal the v2and sampler binding manifests; no sealed file was modified.

- Cycle18 result milestone8a7c06b committed underZhizheZhang; clean local main fast-forwarded. Normal publisher again exits1 atHTTPSUsername, and independentremotehead remains800d09f. All phasePIDs absent or reused with different create-times; no live resolution job remains. Publication credentials question is pending; no remote-update claim or new method nomination.

## Cycle19 design boundary
- Previous goal turn completed and sealed cycle18, so it was progress. Gate-necessity20261021 compares15trained synthetic models; clock-collapse20261025 fixes vector-clock channels and measures15field/Jacobian panels. Neither gives actual imageFID or the literal scalar-clock case in current JiT. Their negative ordinary-control decisions remain intact.
- Select a read-only simultaneous scalar-clock collapse in the two image ODE models, not another trained fixed-gate rescue. Native parameters/time-conditioning/source banks and Heun128stayfixed. Globalclockmeans use nativefirst64validationinputs/time only, no clean target, label, oracle, feature score or optimizer. Finite intervention recomputes downstream states; it is not an anchored derivative patch.
- Fresh Lenovo97041exits255(timeout). New work remains on Mac mini. Two guessed gate/source paths were absent; rg filename/source imports located the existing gate findings and ffn_blocks.py. No files, data or processes were changed by those reads.

- Cycle19 prepare97366 exits0. Preflight58539 consumed exit1 at native hardcodedfloat32 timefeatures vs promotedfloat64MLP, before any formal intervention generation/metric. Originalroot/source/partialcalibration/log retained. V2adds only a time-MLP prehook in the separate double algebra witness; formal native_model/calibration/sampling remain unchangedfloat32, as do1e-9folding andallbitwise/metricgates.

- Cycle19 v2prepare66895 exits0; preflight15067 exits0. Both native64imagepredictions bitwise,12scalarconstants bound; datafreevalues/ADerrors<=2.22e-16, all10promotedwhole-networkfoldcomparisons exactlyequal. Thetime-featurelift is witness-only and absentfromformalmodels. Constants/preflightcurated and committed before interventionmetrics. Traininginstalledversionsmatchwarm spec; noinstallation.
