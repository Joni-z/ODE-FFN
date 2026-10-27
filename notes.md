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
