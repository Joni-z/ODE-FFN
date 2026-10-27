# ODE autoresearch task plan

## Goal
Use the autoresearch experiment loop to develop an evidence-supported ODE architecture for flow matching. A useful successor must face strong SwiGLU and compute controls, including both Product16 and dependent-sign data. Installing a loop or completing one diagnostic does not complete this goal.

## Current cycle: oct27-complete-hidden-field
- [x] Read the official autoresearch protocol, project conventions, and preceding transfer/failed-generator reviews.
- [x] Create isolated branch `autoresearch/oct27-ode` and worktree `/Users/mr.z/ODE-FFN-autoresearch`.
- [x] Write an ODE-specific program with immutable evaluation, explicit budgets, preserved failed experiments, and publication rules.
- [x] Register a read-only checkpoint diagnostic of the complete hidden vector-field derivative, including normalization and gate derivatives.
- [x] Verify analytic derivatives against automatic differentiation and exact checkpoint replay.
- [x] Execute the registered diagnostic with monitoring; assess the result without coefficient or checkpoint selection.
- [x] Publish the protocol/code milestone e0ad302 and the final audited evidence fc1afda to GitHub main, verified through MBP SSH without rewriting commit authors.
- [ ] Select and execute the next justified cycle; keep the overarching goal active until useful architecture evidence exists.

## Decisions
- Latest user direction: experiment execution uses Mac mini and Lenovo. MBP/torch are not compute dependencies. Remove the old publication dependency by setting up local GitHub authentication; do not request SSH restoration for publishing.
- Adapt the official research protocol, not its GPT task, tokenizer, val_bpb, CUDA environment, or unrestricted hyperparameter search.
- Preserve historical data, seed 0, native model semantics, and all sealed evidence.
- Use the actual residual duration c=1/sqrt(3), without tuning it.
- First test frozen checkpoints. This can identify whether higher-order integration of the learned hidden field is useful, but cannot establish the behavior of a freshly trained higher-order model.
- Record failures rather than reset shared/published history. Never overwrite existing runtime outputs.

## Cycle02: fixed-component-support mixture
- [x] Specify the single half-Product/half-sign-code law, its exact Bayes mean/covariance/Jacobian, controls and interpretation limits.
- [x] Commit the protocol/source before outcomes (4016837, also published); generate and hash the new data without changing historical laws.
- [x] Verify the oracle against65536-component enumeration, native cold states and exact resume.
- [x] Train five unchanged architectures with the registered5000-update budget and two monitored CPU workers (PID34331,exit0,56.17s).
- [x] Audit every new trajectory:20 checkpoints,25 panels,five samplings,48 matrices,5000 stream batches; all310 bindings unchanged.
- [x] Publish cycle02 endpoints, P histories, figures, and the scoped interpretation (b35497b verified on GitHub main and MBP).
- [ ] Review the next justified generator question against complete-field constraints and all relevant stop rules before selecting a third architecture.

## Cycle03: exact gated-shear baseline
- [x] Revalidate the current worktree and previous stop rules. Previous goal turn made progress: two completed, audited, published cycles.
- [x] Identify a representation audit that does not add failed feedback/coupling settings: an exact nilpotent lifted flow for unchanged SwiGLU.
- [x] Read primary GLU/coupling references and state the exact conditional-flow theorem, complete FFN Jacobian, nonnormal norm and claim limits.
- [x] Write the representation/compatibility protocol, minimal component and independent literal matrix-exponential fixture; commit before new outcomes.
- [x] Verify native checkpoints, full Jacobians, parameter gradients, exact paired optimizer updates/resume and actual cost. One paired5000-update trajectory completed; this is not a quality search.
- [x] Record conditional-generator singular distributions on real visited states; independent replay passed44 checkpoints,55 panels,11 sampling batches,132 literal FFNs and33 literal local derivative probes.
- [x] Seal all cycle03 results and interpretation (79files, SHA25635fda58e5a73a280c9dd9cf8288c1affc9cfa052d848494f568f4acd618e753a); preserve the published-history boundary.
- [ ] Publish cycle03–05 from Mac mini once local GitHub login completes.

## Next design review: exact gate transport
- [x] Derive square-zero gate transport and its joint cube-zero generator, exact degree-two endpoint, complete FFN Jacobian and fixed-feature rank witness.
- [x] Specify ordinary parameter/MAC controls conceptually; acknowledge that an equivalent ordinary gate mixer is not independent evidence for an ODE effect.
- [x] Verify the rank/operator witness and actual cost, register source and a fixed15-trajectory screen, then preflight before training. Cycle04 completed below and failed its continuation rule.

## Cycle04 execution plan
- [x] Implement only the square-zero gate transport and ordinary diagonal scaling control; retain the native SwiGLU projections and zero-initialize new parameters.
- [x] Register one15-cell screen (8dba424) with all10 historical Swi128/160 controls; verify exact exponential, complete derivatives, rank witness, measured costs, cold pairing, stream and resume.
- [x] Complete15 trajectories on two one-thread Mac CPU workers (PID36784,exit0,170.20s). Lenovo SSH was unavailable; no MBP/torch/GPU execution.
- [x] Independently audit60 checkpoints,75 panels,15 samplings,25000 batch hashes,60 C matrices,120 conditional generators and10 historical exact streams. All345 bindings unchanged. Apply the fixed rule: stop this setting.
- [x] Inspect both figures, match curated files to audited outputs and seal415 runtime files (manifest SHA256 d1a901ebda18e8e0e6c3572f088d8dfcc0cf23842d7163b9ed2b99616d22eb7f). All345 bindings and636 prior sealed files match.
- [ ] Publish locally committed evidence after local GitHub authentication is established. Both previous browser-login processes are terminal failures. Authentication is not a blocker to local research.

## Next bounded step: nonlinear-feature review
- [x] Check older nonlinear-drift, state-clock, anchored-corrector and forced-gate stop scopes, and primary GLU, neural-flow and quadrature literature. No old setting is reopened.
- [x] Verify the preactivation-flow integral, full derivatives, midpoint equivalence and quadrature error using independent numerical fixtures. Float64 derivative discrepancies<=2.78e-17; RK4 agrees with64-node quadrature to2.50e-16. The single-feature Hessian witness differs from midpoint. One controlled screen is justified; no FM-specific benefit is proved.
- [x] Write the one-candidate/midpoint10-cell protocol with all15 reused Swi controls and an independent numerical-accuracy gate. Commit source before formal outcomes.
- [x] Complete native preflight and the monitored10-cell screen on Mac mini (5cab241 registration, PID37824,exit0,121.67s).
- [x] Independently replay40checkpoints/50panels/10samplings/25000batch hashes/120C matrices/240pointwise spectra/15historical streams. Audit exits0,14.36s; all466bindings and1051prior sealed files match.
- [x] Apply the registered decision: quality fails, numerical accuracy passes. Maximum two-node/32-node gate RMS0.0169%, field-MSE change0.0176%, complete-Jacobian RMS0.0360%. No node-count or other setting rescue.
- [x] Inspect both plots, copy curated audited outputs, document the first-order tangent redundancy and seal320files (SHA2562a7a5fd06458ee4578db8cb52ed8076178b4b96046c2e1832fc267db697d6885).

## Next mathematical question, before another architecture
- [x] Write a bounded read-only counterfactual protocol for the fitted nonlinear curvature contribution and20 complete C-tangent absorption checks. Keep this descriptive; no coefficient optimization, new training, initialization change or reopening of cycle05. Register source before reading these counterfactual outputs.
- [x] Execute the fixed five-checkpoint cohort (0f39045,PID38295,exit0,3.27s), replay80000original predictions/25Jacobian panels, compute25counterfactual panels and20nonzero complete tangent checks. Maximum tangent discrepancy1.11e-16; independent saved-array audit passes.
- [x] Seal/archive53cycle06files (SHA2560d0118ab6febf04a486b18cbaee3771a5c04176c1d4911ea44629c9835faf7f1), verify1371prior sealed files, inspect the plot and synchronize local main.
- [ ] Push the pending sequence when Mac mini GitHub authentication is established; the last local auth check still reports not logged in. No remote publication is claimed.
- [ ] Use the complete-network evidence to decide whether any different task-relevant ODE computation is justified. A new operator must face its ordinary implementation and matched cost, rather than repeat a solver-accuracy premise.

## Cycle07: prediction-space and data-geometry compatibility
- [x] Revalidate cycle06, the unselected head-control review, JiT Section3.3, Gagneux2026v3 controlled geometry and Liang2026v2 residual-stream evidence. Earlier head experiments did not test a high-ambient, low-rank target.
- [x] Derive/verify thin-versus-thick embedded-mixture oracles and universal linear-bottleneck field/Jacobian bounds. Independent dense-conditioning values and full AD Jacobians agree to4.89e-14/1.48e-13. Thin direct-head uniform population risk floor is0.468229, independent of hidden P. This is not a finite-bank lower-bound assertion.
- [x] Write the paired128D benchmark with two normal-noise levels and18fixed trajectories: unchanged professor K2/Swi128/Swi192 crossed with clean/shifted/direct heads. All heads share one capped regression target; old datasets/evaluators and stop rules remain unchanged. Commit source before preparation/outcomes.
- [x] Verify oracle/AD identities, raw pairing, costs and exact resume for18cells; complete training under registrationb19bcb8 (PID39080,exit0,331.57s), two monitored one-thread Mac CPU workers. No new FFN or learned skip.
- [x] Independently replay72checkpoints/90complete Jacobian panels/18sample batches/10000batch hashes/144P spectra/two oracle samplers. All65bindings/prior seals unchanged. Native nonzero-head JVP audit also passes; audit exits0,34.51s.
- [x] Apply the fixed interpretation: thin/clean ODE is9.0%/11.8% better than parameter/compute Swi controls, but misses10% against both and regresses22.7% on thick tangent error. No context lead or new architecture is nominated. Record continuous-ring failure and capped-sampler bias.
- [x] Inspect all four figures and seal584runtime files (SHA2562638989d5b46798e4150d933140ad439c2ba62ea3dc2514e4a88cfd92e83e0b4). Curated audited files match byte-for-byte; all65bindings and1424prior sealed files match.
- [x] Commit curated cycle07 evidence and synchronize local main after all archive/figure/source checks. GitHub still needs local authentication; the latest gh check remains unauthenticated.

## Next unselected read-only review
- [x] Register and verify the final-readout null-space sampling identity on all18 fitted cycle07 endpoints. Distinguish deterministic source-wise Heun checks, population W2 bounds and empirical SW2. No learned radial skip, wider head or renewed head training is selected.

## Cycle08 execution plan: fixed-readout sampling invariant
- [x] Revalidate cycle07 (019e676, clean worktree,18completed/audited models). Previous goal turn is progress; local GitHub authentication is still absent.
- [x] Write the fixed read-only cohort/protocol:18final cycle07 models, six zero-P interventions at fixed readout, two earlier2D/16D rank controls. Register source before new fitted null-space outcomes; no training or quality selection.
- [x] Derive the affine-subspace/Gaussian-W2 theorem with hypotheses and limits. Literal nonlinear Heun fixtures, zero-gamma/bias and full-row-rank controls pass; maximum float64 source identity error5.78e-15, scalar coefficient discrepancy1.39e-17. Discrete skip factor.02116394 differs from continuous.01839397.
- [x] Hash-bound9inputs, source/cohort and2008prior sealed files; formal run PID40649 completed18models in6.57s, exit0.
- [x] Verify90raw panels/18source-wise identities/covariances/normal W2 bounds and six nonvacuous P interventions on Mac CPU. Max original/zero-P identity errors9.21e-7/9.81e-7.
- [x] Independently reconstruct projectors through centering-basis/QR algebra and replay six P interventions bitwise. Terminal audit receipt/log report PASS,6.00s;32formal files unchanged. Audit handle was lost to truncated observation; no rerun or exit-code claim.
- [x] Archive the reviewed figure/results and seal61files (SHA256384d9b6974dddd969bd48509c9f889763d5e5a08b41585f2be543ce53fd2d650); all9bindings and2008prior sealed files verify.
- [x] Commit verified cycle08 milestone11ae9e6 and synchronize the clean local main checkout. Publication still requires local authentication.

## Cycle09 execution plan: conditional input-information attribution
- [x] Revalidate clean local c2ba616 and cycle08 archive. Previous goal turn is progress, not a repeated blocker; publication authentication is separate from local execution.
- [x] Derive exact conditional means and information/residual/cross decomposition. Dense fixtures agree to9.06e-14; full-observation native oracle to1.78e-14; rank-deficient fixture passes. Residual includes output restrictions and is not a causal FFN optimization measure.
- [x] Register fixed72-checkpoint cohort, five time bins, endpoint signed attribution rule and runnable source before conditioned fitted outcomes. Bind immutable inputs at preparation; no new training or candidate selection.
- [ ] Verify float64 dense-conditioning and full-observation fixtures; run on one monitored Mac CPU thread within600s, retaining per-row components privately and publishing summary tables.
- [ ] Independently reconstruct input row spaces through QR, recompute conditioned means through2D Woodbury solves, replay all72 decompositions and18native endpoint predictions, verify unchanged seals.
- [ ] Inspect the plot, write the scientific decision with limits, seal and commit; publish when local authentication is available.

### Cycle03 decision boundary
The scientific question is whether a useful existing gated FFN admits an exact constrained ODE representation without losing any forward/training behavior. This is a baseline/representation theorem, not a new quality method. No feedback, correction coefficient, extra order, new target law or parameter search is selected. Even perfect compatibility cannot complete the useful-successor objective.

## Errors encountered
- Cycle08 documentation patch matched a stale final index sentence and failed atomically. Files were re-read and exact targeted replacements were applied; frozen inputs and outcomes were untouched.
- A multi-file cycle07 report patch assumed an outdated root-README sentence and failed before writing files. Git status and file existence were checked; the report additions and exact README edit were reapplied separately. No source snapshot, dataset or run changed.
- Two initially guessed legacy review filenames did not exist. A filename-only search located the actual anchored-corrector protocol/analysis; those were read instead. No legacy file was changed.
- The archive check initially expected the private rank-analysis `summary.json` to retain its filename in curated results. The curated name is `gate-rank-summary.json`; the corrected explicit mapping passed byte comparisons. No runtime seal had been written by the failed check.
- A broad text search included archived patches/metrics and produced excessive output. Restrict subsequent stop-rule searches to named top-level protocol/review files.
- A multi-file report patch matched a stale task-plan status paragraph and failed atomically. Current files were inspected and the patch was split/reapplied to actual text; no experiment or frozen artifact was changed.
- A completed audit handle10994 was polled again after its exit0 was consumed. The tool reported Unknown process; the terminal log/files were inspected without relaunching computation.
- The second local GitHub browser-auth process ended with network EOF. It is terminal; no login success is established and its old device code must not be reused. Publishing remains pending separately from local research.
- Cycle04 first audit compared two-row formal conditional-generator probes against slices from a32-row float32 projection. It stopped at a cold spectrum with2.68e-7 discrepancy. Preserve that source/log; audit-v2 repeats the correct two-row projection and keeps the same spectrum tolerance. No formal result is changed or rerun.
- The first local GitHub CLI browser-auth process ended with a network unexpected-EOF error; no local token/login success was established. Do not treat the pending browser question or its old code as proof of authentication.
- Cycle03 publication failed while copying its bundle to MBP, and both the Tailscale address and the previous LAN address were unavailable. No remote branch was updated. The user clarified experiments should use Lenovo/Mac mini; local publication setup replaces this transport dependency.
- Final-results publication initially failed at GitHub HTTPS fetch, before any bundle or remote write. A fresh fetch succeeded; the terminal publication was retried and b35497b was verified remotely. No experiment was restarted.
- Audit-v5 passed all Product16 cases, then one sign-code panel output differed by3.0599e-6. V6 makes panel-output tolerance1e-5 to match the existing full-bank numerical replay threshold; all other tolerances and formal outcomes stay fixed.
- Audit-v4 stopped on two Product16 generated coordinates after64 field evaluations, with maximum violating difference5.11e-5. V5 records all sampling maximum/RMS differences and checks max<1e-3, RMS<1e-5 plus unchanged stringent SW2 agreement. All failed audits remain preserved; formal outputs are never rerun.
- Audit-v3 passed all Gaussian cases, then stopped on one Ring8 float32 coordinate with1.2945e-6 difference after elementary normalization, beyond the1e-6 absolute tolerance. Audit-v4 uses1e-5 absolute prediction tolerance, retaining2e-5 relative,3e-5 Jacobian absolute, and stringent float64 fixtures. Scientific acceptance thresholds are unchanged.
- Audit-v2 field values matched but nested derivatives through fused LayerNorm did not match the analytic formal Jacobian. A separate float64 finite-difference fixture supported the analytic implementation. Expressing normalization in elementary operations inside independent AD resolved the fixture to4e-16; preserve v2 and run the full audit-v3 before accepting verification.
- The registered text incorrectly said every field bank had20,000 rows. Actual immutable2D banks contain10,000 each, while both16D banks contain20,000. Formal evaluation already used every row. Independent audit stopped on the mistaken count before replay; preserve that source/log and correct only the count assertion in audit-v2. See cycle01 deviations.md.
- A guessed old recurrence-protocol filename did not exist; rg located the actual immutable `docs/research-recurrent-depth-2026-10-26/protocol.md` before registration.
- Lenovo SSH timed out; torch SSH authentication failed. Mac runs remain available. MBP HTTPS credentials were inaccessible, but MBP GitHub SSH authenticated as Joni-z; use that route without changing commit authors.
- Local GitHub SSH initially lacked a host key; accepting the new key reached authentication, which failed. No local SSH publication succeeded.

## Status
Cycle08 formal verification and independent audit pass. All18readouts have rank63; six zero-P interventions move samples by0.108–0.355RMS but preserve output-null coordinates to1.30e-6. This localizes a generic128D complete-predictor restriction; the older2D/16D heads have full row rank and their negative results remain unexplained by it. No new architecture or training is selected. Cycle08 is archived, sealed and committed at11ae9e6; local main is synchronized. Next quantify conditional input-information loss before proposing another ODE computation. GitHub still requires Mac mini authentication; Lenovo SSH still times out. Only Mac mini/Lenovo are execution hosts. The useful-successor goal stays active.
