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
