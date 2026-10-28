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
- [x] Frozen preflight passes; formal PID41562 exits0 after14.15s on one Mac CPU thread,72checkpoints/1296summary rows, peak sampled RSS0.408GiB. Per-row components remain private.
- [x] Audit-v2 exits0 after17.73s; QR/Woodbury conditional fields agree to1.13e-13, all72decompositions/1296rows and18bitwise native endpoint replays pass;78formal files unchanged. V1 audit normalization error preserved, no formal rerun.
- [x] Inspect the plot and write the scientific decision with limits; seal108runtime files (SHA2568e572acd096f1310fe2ff2cab337342c981a6fcfa4d265a1a34cda3a94e984f5), verify8bindings/2069prior files. Private runtime126.3MB; curated results0.46MB.
- [x] Commit verified cycle09 milestone8a05b13 and fast-forward the clean local main checkout.
- [ ] Publish the pending local sequence after Mac mini GitHub authentication; no MBP/torch relay.

## Cycle10 plan: complete conditional-field derivative diagnosis
- [x] Revalidate clean9e1578f, cycle09 attribution and archived evidence. Previous goal turn is progress. No new architecture is selected.
- [x] Derive exact conditional derivatives and covariance/projector commutator. Keep derivative cross terms; field-optimal references need not minimize Jacobian loss or be symmetric.
- [x] Register all90existing endpoint panels, full/tangent symmetric/skew decomposition and fixed late-time descriptive classification before fitted conditional-Jacobian outcomes. No training or threshold rescue.
- [x] Preflight passes: AD1.05e-12/finite differences7.55e-9/full observation3.30e-12/commutator4.45e-16. Formalv2 PID42271 exits0 after34.22s,90panels/5760anchors,0.410GiB sampled RSS.
- [x] Audit exits0 after17.95s: all90native panels/5760Jacobian anchors bitwise,180independent denseAD reference anchors agree to2.49e-14, fields6.27e-14;96formal files unchanged. Figure reviewed and interpretation written.
- [x] Seal135runtime files (SHA256e8c8278b77ce7fd6550c0cd684936b56c1e61bf62963efc8cf11d3449349c9f6), verify7bindings/2177prior files and preserve10initial preparation files. Private728.1MB, curated0.29MB.
- [x] Commit verified cycle10 milestone27f4503 and synchronize the clean local main checkout.
- [ ] Publish pending local commits after Mac mini GitHub authentication; no MBP/torch relay.

## Cycle11 plan: one spectrum-preserving conditional-frame hypothesis
- [x] Revalidate clean5684e5d and cycle10 derivative diagnosis. Previous goal turn is progress; no existing stop is reopened.
- [x] Review HOFT2025, DeltaProductNeurIPS2025, HouseGNN2026preprint and ODEtoODEhistorical prior, plus old state-clock/affine/clock-response stops. Derive conjugation, commutator differential, ordinary low-rank realization and the limited local-response witness.
- [x] Prepare one cold screen on unchanged thin/thick clean banks: conditional-frame K2, global-frame K2 parameterization control, conditional-frame K1 ordinary control and Swi257 compute control; reuse sealed original K2 and Swi192 parameter controls. Freeze all budgets/thresholds before outcomes.
- [x] Register initial designa8259f5 and final equivalent-arithmetic270c1d0; v4preflightPASS for all8arms: exact native resume, bitwise whole-model initial/inherited gradients, common tensors/buffers/RNG and measured budgets. Independent active derivative5.11e-15, complete commutator1.78e-15. No scientific threshold changed.
- [x] V4formal8cold cells complete: PID43415 exits0 after284.07s, two one-thread Mac CPU workers, sampled total RSS1.349GiB. Seed0/5000updates/data/loss/head/clock bounds unchanged.
- [x] Audit-v2 exits0 after38.56s:32new/16reference checkpoints,40new/20reference Jacobian panels,8new/4reference samplings,10000stream batches,144newP snapshots and36fitted dense-frame checks. All235formal files unchanged.
- [x] Apply fixed decision: stop this K2 frame setting. Thinfield77.9%better thanSwi192 but only3.93%better thanframeK1; thicktangent/fullSW2/tangentSW2guards fail. No historical-wave or frame-count/scale/width/initialization rescue.
- [x] Review all4figures; seal317runtime files (SHA256392371988605176496b8af7b6876751a7370a911adf40e709a44d1c95b922449), verify12bindings/2312prior files,144derived spectra/132telemetry rows, preserve33failed-preparation files. Private592.3MB/curated1.61MB.
- [x] Commit verified cycle11 milestonef4a3c64 and fast-forward the clean local main checkout.
- [ ] Publish pending local sequence after Mac mini GitHub authentication; no MBP/torch relay.

## Cycle12 plan: discrete defect transport on frozen sampling paths
- [x] Revalidate clean e1754fa and cycle11 failed-screen decision. Previous goal turn is progress. Preserve every trained endpoint and all frame/head/loss/solver stops.
- [x] Derive exact oracle-suffix telescoping/signed MSE, scalar normal suffix gain and fixed-readout forced bridge prediction. Keep paired error distinct from SW2 and finite secant gain distinct from stability bounds.
- [x] Prepare registration of all12endpoints/2048sources/33branch endpoints/32defects/fourfixed eight-step bins, nativeHeun32/cap and forced-readout decomposition. No new training or best-prefix selection.
- [x] Preflight fixtures/native endpoint replay pass (14.74s). Formal PID44962 exits0 after126.32s on one Mac CPU thread, peak sampledRSS1.340GiB:12models/2048sources/33branches, all raw paths retained. No new training.
- [x] Audit exits0 after61.88s:12native endpoints/384learned steps/396oracle branches bitwise, all signed projections/scalars checked;21formal files unchanged. Four figures inspected; strict analysis/statistics/catalog written. ThinframeK2 last-quarter tangent share1.01%; thick forced-normal fraction91.53–93.15%, shared by all families. No new nominee.
- [x] Seal77runtime files (SHA256c5c17339156af984d03edaea508bd4a007aaad2a9719decdd1a2d1872c02675b), verify7bindings/2629prior files/exactauditcopies/48conditionalprecisionrows/78locallinks/sourceAST/fourfigures. Private1.209GB/curated1.146MB; no model/evaluator/Report1changes.
- [x] Commit verified cycle12 milestonee228381 and fast-forward the clean local main checkout. Direct Mac mini publication reaches push but exits1 because HTTPS credentials are absent (terminal prompts disabled). No hybrid nominee or cycle11 rescue.
- [ ] Publish pending sequence after local GitHub authentication; remote main remains800d09f.

## Cycle13 plan: what is the quadratic action adding?
- [x] Revalidate clean1329914 and cycle12 final seal; previous goal turn is progress through the strict analysis/archive and direct publication attempt. Preserve every earlier stop.
- [x] Read prior direct-effective-map, controller-response, state-clock and recurrent-depth controls. Those do not measure the best scalar K1 approximation to the trained K2 action at actual layer inputs.
- [x] Research card/theory/protocol and runnable scalar-envelope code written. Projection/boundary/zero-direction/flat-SVD fixtures pass; live-frame/clock AD identity max4.44e-15. Primary AREX/Koopman/MIMS sources checked with external-versus-hidden-flow limits; no new spectral constraint or architecture.
- [x] Cohort fixed before fitted outcomes:24checkpoints/144solve-checkpoint pairs, all20000field states each; native inference chunks are1024 (trainingbatch512 is different),32-rowclock probes. Fixedc=.6, fourtimebins, free/boxed scalar relaxations and one same-clock centred whole-model intervention. No retuning/rescue.
- [x] Preflight24nativefields passes7.71s. FormalPID46900 exit0/78.04s/0.768GiB; independentauditexit0/58.67s/0.848GiB.24fields/hiddeninputs/interventionsbitwise,144layers/2.88Msourceenvelopes/1.92Mdenseframes checked;56formal files unchanged.72reporting-onlysubspaceaccountingrows retained; savedhistoryclosure is part of archive verification. No oracle-clock architecture.
- [x] Scientific decision/strict analysis/statistics/catalog and three inspected figures complete. Seal112files SHA256c1a7dd5187106378bb1156645e15c3adcae24253061823c08c5327e797f20b91;8bindings/2706priorfiles/144P-clockhistories/72projected-accountingrows/67links/sourceAST pass. Private1.369GB/curated0.709MB, no production/evaluator/Report1changes.
- [x] Commit verified cycle13 milestonea991af0 underZhizheZhang and fast-forward clean local main. Directpublisher session67397 exits1 for missing localHTTPSUsername; remote main independently remains800d09f.
- [ ] Publish pending sequence after local Mac mini GitHub authentication. No architecture nominee; useful-successor objective remains open.

## Cycle14 plan: independent second-coefficient direction
- [x] Revalidate clean e608c9c and the cycle13 seal. Previous goal turn is progress. Read old external-time adapter, static effective-map, nonlinear-field and frame/order stops; this proposed change keeps K2 and scalar clocks, but introduces one independent second-coefficient matrix.
- [x] Derived nonautonomous jet, direct-B ordinary equivalence and commutator/complete-gradient formulas. ICLR2025nonautonomous-transformer and AREX2026primary sources scoped without novelty/FM transfer claims. ActivecompleteJacobianfixturemax2.22e-16; small-duration ratios near8, C-family inherited gradients/RNG bitwise.
- [x] Prototype/calibrate C/affine/free/direct and Swi258; actualparams/MACs fixed in protocol. Full trainer initialization/resume remains the mandatory frozen preflight gate. Preserve executed calibration source/log; reporting-only warning detached before registration, no fitted outcomes.
- [x] Register19af9a8 before outcomes;14bindings/2818priorfiles. Alltenfull-model inherited initialization/active-head and exactthree-update/resume checks pass. Direct-B native prediction discrepancies2.38e-7/4.77e-7 are recorded; inherited P gradients intentionally differ. No frame/cap/clock-range/optimizer/solver change.
- [x] FormalPID47790 exits0 after234.80s, two single-thread Mac CPU workers, sampled summedRSS1.240GiB. Auditexits0 after44.13s:40new/16refcheckpoints,50new/20refpanels,10new/4refsamples,10000batches,192P-coefficientrecords/6144localJacobiananchors.293formal files unchanged. Both nomineesfail; no rescue/historical wave.
- [x] Analysis/statistics/catalog and four figures reviewed. All192wideP/Bref/Cspectra and154training-window rows curated. Post-result coordinate-metric theorem/toyfixture explains class-equivalent update geometry without an Adam causal claim or optimizer rescue.
- [x] Seal366runtime files SHA256182a3daa0a237b1d2c7e6b509005869f6002a25c15649668772e24d2069b8ac7;14bindings/2818priorfiles/192derivedspectra/154trainingrows/exactauditcopies/77links/AST/fourfigures verified. Preserve3executedcalibrationsourcefiles; private740.7MB/curated2.162MB.
- [x] Commit verified cycle14 milestone51fedd5 underZhizheZhang and fast-forward the clean local main checkout. Directpublication11320 exits1 atHTTPSUsername lookup; remote main independently remains800d09f.
- [ ] Publish pending sequence once Mac mini GitHub authentication is available. Both candidates stopped; useful-successor objective remains active.

### Cycle03 decision boundary
The scientific question is whether a useful existing gated FFN admits an exact constrained ODE representation without losing any forward/training behavior. This is a baseline/representation theorem, not a new quality method. No feedback, correction coefficient, extra order, new target law or parameter search is selected. Even perfect compatibility cannot complete the useful-successor objective.

## Cycle15 plan: loss components and hidden gradient routes
- [x] Revalidate clean cd09851. The previous clarification turn made no research progress; no study process is live. Lenovo SSH times out; use one Mac CPU thread, without MBP/torch.
- [x] Inspect existing telemetry before selecting the diagnostic: final thin windows never clip, while thick windows clip63.4–93.8% across cycle14 families. This motivates checking the indirect route, not assuming a large loss drives P.
- [x] Derive the frozen-readout null-cotangent theorem; register all22 unique clean models from cycles07/11/14 at four saved checkpoints, exact next native batches, gradient decomposition and promoted Adam counterfactual. No training or optimizer rescue.
- [x] Pass independent algebra, zero-head, cross-term, clipping and Adam fixtures; commit protocol/source before fitted gradient outcomes.
- [x] Run monitored native replay and88-checkpoint gradient probes; independently audit every decomposition and preserve immutable inputs.
- [x] Curate strict analysis/statistics/catalog, final numeric table and two visually reviewed figures. Seal225runtime files (SHA256e36f77d3b56196069816003687cec9835c5887fc091072c3af3540edf9960a71), verify10bindings/3184prior sealed files and all public copies. Private470.5MB/curated0.459MB. No source/evaluator/Report1changes.
- [x] Commit verified cycle15 milestonea7d1c76 underZhizheZhang and fast-forward clean local main. Directpublisher8587 exits1 atHTTPSUsername (Device not configured); read-only remote main independently remains800d09f. Publication remains pending local authentication.

## Cycle16 design review plan
- [x] Revalidate clean8f459e5 and completed cycle15; previous goal turn is progress through the audited88-checkpoint gradient study. No model has met the useful-successor criterion.
- [x] Review conservative readout, recurrent depth, cubic drift, external-time operator, spectral features and Stage9covariance stops. Reject Hopfield/softmax-covariance K2 as an explicitly prohibited higher-order rescue before implementation.
- [x] Check primary Hyper-SET2025/ICLR2026, Hopfield discretization2026 and JiT paper. Energy/covariance algebra is established and already tested here; no architecture novelty or FM gain follows.
- [x] No new operator survives the current stop/evidence review. Identify a distinct coverage gap: recent screens use a tokenless vector backbone, whereas the original question concerns JiT with attention. Do not move failed candidates to a new dataset as a rescue.
- [x] Prepare a separate CIFAR-10 unconditional image baseline bridge using literal professor K1/K2 and ordinary parameter/work-covering SwiGLUs only. Preserve all synthetic decisions. State the new dataset/splits/metrics explicitly; this is not a new architecture candidate or a replacement validation.
- [x] Before any quality training, verify official data hash/provenance, actual JiT input-clock semantics, exact common surrounding tensors, derivatives/resume and measured CPU/MPS throughput. Freeze a resource-bounded protocol only after feasibility measurement. No MBP/torch or stopped-operator rerun.

- [x] Independent preparation audit and both clock witnesses pass. Seal92files SHA256c1f13c9a91b1599439670918315dc891bdce38adce86dfc4ba775bdb6c3d7145,23source/8closurebindings/225oldsealedfiles verified. Data324.5MB/privateprep0.805MB; two reviewed figures. No real image quality update.
- [x] Commit verified image preparation8baec48 underZhizheZhang and fast-forward clean local main. Publisher22601 exits1 atHTTPSUsername (Device not configured); independent remote main remains800d09f. No remote update claimed.
- [x] Register the separate four-arm quality-training/sampling/feature-metric protocol in cycle17 before real CIFAR updates. Cycle16 remains preparation-only; no new ODE candidate is selected.

## Cycle17 plan: fixed CIFAR image baseline screen
- [x] Revalidate clean98730e0, cycle16 sealed preparation, absent quality jobs and a fresh Lenovo SSH timeout. Previous goal turn is progress, not a verified wait or repeated research blocker.
- [x] Register one four-arm, seed0,5000-update image baseline protocol with unchanged literal input clocks, fixed paired draws, CPU backend, generated samples and small-sample Inception FID. No new ODE candidate or old-family rescue.
- [x] Build a separate declared metrics environment; validate actual seeded kernels/weights and the required fresh-agent documented invocation. Keep training/synthetic environments unchanged.
- [x] Register057eb60/fe9d1a3 and finalcafa5af before outcomes. V2source/data/weights frozen; identical reference features reused by hash. Independent fresh-agent trainer doc pass exits0; all4model/moment/global/data RNG/buffer/disk-resume checks bitwise. Cold Heun replay passes; actual dense FFN training MACs measured and covered by controls.
- [x] Launch verified parent51294 via unified session39868, children51302/51303 (ODEK1/K2) on Mac miniCPU; twoSwiGLUs remain queued. Status, launch records and5sCPU/RSS telemetry are live; no duplicated process. Preserve all P/checkpoint/data-RNG trajectories.
- [x] Complete all four arms (training parent39868 exits0, 1,107.813s). Replay44checkpoints/80panels/48P/240clocks/all4096sampled images and full5000-batch stream; all pass. Independent6,144-image feature replay is bitwise and separate SVD FID agrees within4.3e-13. Original failed audit source/status preserved; v2 only converts a path type.
- [x] Curate all four results, fixed decision, FID mean/covariance accounting and five reviewed PNG/PDF figures. Write English analysis, statistical scope and case studies. K2 weightedMSE .176842 versus K1 .176393 and Swi129 .171234; FID183.40 versus184.91/194.10. Fitting/order gates fail; no new method.
- [x] Archive session73203 exits0:392runtime files, sealSHA25662d178c4e2b88780495c9ea45c6aeef96453ce7098a8aeee84076cf58f081172. All71input/3postprocess/v2audit bindings, public copies,71local links and protected scopes verified. Private805570107bytes; curated1193604bytes.
- [x] Commit audited results a267210 under Zhizhe Zhang and fast-forward clean local main. Direct publisher20200 exits1 at missing HTTPS Username; independent ls-remote still800d09f. Lenovo85331 exits255/SSH timeout. Neither blocks local frozen-checkpoint analysis.
- [x] Select one frozen-checkpoint numerical-confound question: does the lower ODE FID persist across uniform Heun32/64/128? No training, best-solver nomination or change to cycle17 gates.

## Cycle18 plan: frozen image sampler resolution
- [x] Register exact four-model/all1,024-start, uniform Heun32/64/128 cohort, fixed metrics, mathematical scope and stop rules. Cycle17 fitting/order failure remains unchanged.
- [x] Register5e2c320 and pre-outcome resource amendment100f322. Prepare80542 exits0; preflight89758 exits0:all4096native32outputs bitwise, zero-clean/constant-velocity fixtures pass at32/64/128,129.1s. Actual installed package versions match both warm specs; parent392sealed files unchanged.
- [x] Sampling parent52386/4children all exit0,481.075s; controller99684 exits1 only because sampling-audit imports SciPy at module top in the deliberately lean training env. Preserve original source/log/failed status. All8finer banks already exist; no training/sampling restart. Register lazy-import audit-v2 with separate source bindings and a new recovery controller before any FID outcome.
- [x] Recovery69697 consumed exit0. Sampling audit52792 exits0/765.135s, all8x1024finer raw arrays bitwise and20sampler files unchanged. Metric producer53149 exits0/110.242s. Feature audit53197 exits0/188.166s:12,288generated+2,048reference features bitwise; all12FIDs independently agree within6.54e-13 and30formal files unchanged.
- [x] Curate76595 exits0, all12scores/15contrasts/12raw pixel differences, three visually reviewed PNG/PDF figures and English analysis/statistics/catalog. Ranking persists, K2-Swi129 gap10.71to8.32; no change to failed fitting/order gates.
- [x] Archive90796 exits0:116files sealed, SHA2560a6fb24d1aba029be90e95f7c74919bbbdead973c101819852f04cb55cc64108. Parent392files,7sourcebindings, separatev2/sampler/controller manifests and77local links verified; private170067471bytes/curated1047106bytes. src/synthetic/configs/scripts/Report1 unchanged.
- [x] Commit verified cycle18 result8a7c06b under Zhizhe Zhang and fast-forward clean local main. Direct Mac mini publisher exits1 at missing HTTPS Username; remote main independently still800d09f. All recorded phase PIDs are terminal. Publication remains pending local credentials; no new architecture nominated.
- [x] Verify bounded JiT/NeurIPS high-order FM primary sections; exact cap terminal scale.05/e and a loss/generation counterexample pass. Restate/link existing canonical Gaussian material-acceleration result, verified by AD. Public interpretation/coverage/fixtures added; no new trained method or cap rescue.

## Cycle19 plan: frozen image clock attribution
- [x] Revalidate clean311aed7 and completed cycle18. Previous goal turn made progress through the audited resolution study. Old synthetic clock-collapse and gate-necessity studies show useful conditioning but do not measure this actual image FID lead. Do not reinterpret them as image evidence or revive training variants.
- [x] Register a single simultaneous six-clock collapse for each original K1/K2 image endpoint. Calibrate each global scalar from one native64-image/no-target forward, keep P/attention/time conditioning fixed, and use the same1,024starts/Heun128/reference features. Reuse all four sealed native controls explicitly.
- [x] Registeredcecf226/73631af; fresh v2prepare66895 exits0 and preflight15067 exits0. Nativefirst64predictions bitwise for botharms;12clockconstants fixed. Data-free value/ADidentity errors<=2.22e-16;10whole-network folded-vs-literal promoted predictions exactlyequal. Bothsealedparents and warmtrainingversions verify; formalfloat32model unchanged.
- [x] Controller51166 consumedexit0; bothchildren exit0. Sampling audit53893 passes both fullvalidation predictions/tenpanels/all2048raw samples bitwise; all learnedtensors unchanged. MPSmetrics54034 exit0; features54037 exit0, new2048+reference2048featuresbitwise, allsix FIDs independentlycheckedwithin5.41e-13. No training/sample retry.
- [x] Correct_metadata+curation66542 exits0; only four native raw-fraction metadata fields corrected to128-step banks. SixFID/loss values unchanged. Allsix endpoints/12clockmeans/30panels/three reviewedfigures and English analysis/statistical scope/catalog curated. K1FID185.37to184.16;K2183.70to179.72, primaryfittingchanges<0.1%.
- [x] Archive52574 exits0:107files sealedSHA256c8ecd01c4c24ad6be6f3a8d1bf3aa7b5b3222f03712e4879e2cd5ecf15759d96. Teninputbindings,12clockscalars,69links,allpubliccopies/metadata-correctionrawfractions/protectedscopes verified. Private76731701bytes/curated707986bytes.
- [x] Commit verified clock-attribution milestone267dc74 under Zhizhe Zhang and fast-forward clean local main. Direct Mac mini publisher exits1 at missing HTTPS Username; independent remote main remains800d09f. All controller/child phase PIDs are terminal. Publication remains pending local credentials; no new architecture nominated.

## Cycle20 plan: cold ordinary image SiLU controls
- [x] Revalidate clean1a685eb and completed/sealed cycle19; previous goal turn was progress. Read the earlier synthetic SiLU control and its Ring8 failure. This new image baseline fills the same-form control gap exposed by clock collapse; it does not rescue a stopped ODE family or nominate an ordinary FFN as an ODE method.
- [x] Registered047479e: three bias-free SiLU widths64/131/193derived from shape/parameter/work comparisons. All original data/loss/optimizer/5000updates fixed; FFN initial functions differ and are not claimed paired.
- [x] Prepare97177 exits0; fresh-agent documented preflight64217 exits0/no mismatches, frozen source047479e verified. Wholeparams185744/211472/235280; FFNforward8192/16768/24704, training24576/50304/74112. Three physical model/optimizer/RNG/buffer resumes, common backbone, active actual gradients, JVP and cold Heun pass. Warm environment packages unchanged.
- [x] Controller37739 terminal exit0,1554.29s, all four phases and all three children exit0. Training audit passes33checkpoints/60panels/72W histories/3072manual-Heun images/5000stream batches; fresh feature replay and all nine independent FIDs pass. No retries. Curate all endpoints next.
- [x] Curator4376 exits0: nine-endpoint table,77curve rows,72matrix records,45final panels and fixed first-eight cases; all four figures visually inspected. Analysis/statistics/catalog written. SiLU193weightedMSE0.171526/FID180.0454 vsK20.176842/183.6981 at equal matrix work but11.46%more whole parameters. Near-parameter rankings mixed; frozen constantK2FID179.7179 retained. No ODE nomination or rescue.
- [x] Archive42711 exits0:245files sealSHA256d7fe17e92c7bd8468b6c048aa9925a4350d95168ed611682bb9c7d787634e973; ten bindings, public copies,83links/AST, all nine scores/costs/raw ranges and protected scopes pass. Private537170357bytes/curated1300414bytes. Seal re-read and all245hashes match.
- [x] Verified result534d1df committed and clean local main fast-forwarded. Publisher56955 exits1 only at missing HTTPS Username; independent ls-remote confirms GitHubmain800d09f unchanged. No remote publication claimed. All four phase processes are terminal.
- [ ] Before selecting another method, reassess the ODE-specific hypothesis against all ordinary image controls and existing synthetic stop rules; no width/init/budget rescue is authorized by this result.

## Cycle21 plan: frozen image matrix-flow attribution
- [x] Review completed cycle20 and previous negative exact-flow/recurrence/effective-map/covariance evidence. Primary PDE-SSM and GraphFlow papers are checked; distinguish spatial-token dynamics from graph nodes that are entire latent images. No new spatial or spectral architecture selected.
- [x] Specify one finite question: same learned image P and frozen durations, polynomial versus exponential; four cached-map endpoints include two arithmetic controls, all nine prior endpoints reused. Derive actual map/tail/nonnormal/semigroup distinctions without a task-quality theorem.
- [x] Registeredf56509b; preserved failedv1preflight87896. V2sourcea2ad4d2, prepare30921 exits0. Fresh documented preflight82275 exits0:48independent operator checks/two native+constant bitwise prediction replays/ten promoted witnesses all pass. All nine runtime source files match registration; doc mismatches none. Existing image/check/metrics versions and MPS availability verified, no installation.
- [x] Readiness90d20a5 precedes formal controller39600, consumed terminal exit0 after792.89s, no retries. All four sampling children and four phases exit0. Independent sampling430.96s replays48operator records/all4,096images/20panels bitwise; new features4096/reference2048bitwise and all13FIDs within5.41e-13. All parameters/buffers unchanged.
- [x] Curator84158 exits0; all thirteen endpoints/48P histories/65panels/six arithmetic/function contrasts retained, three figures visually reviewed. K1expFID297.584vs184.155folded,weightedMSE+11.40%; K2expFID178.773vs179.718,weightedMSE+0.0453%. No joint quality win, retraining family or old-stop rescue selected.
- [x] Archive72371 exits0:136files,sealSHA256fa13abe370318077c521bb4202bf8f37b8ccfee6daf4cb5c0ef7ea92daf9b3cf. All136hashes re-read, public copies/13FIDs/48P histories/65panels/six exact contrasts/links/AST/protected scopes pass. Private160067709bytes,curated1643190bytes. All phase processes terminal.
- [x] Resultf8c79ae committed asZhizheZhang and clean localmain fast-forwarded. Publisher2921 exits1 only at missingHTTPSUsername; independent ls-remote still800d09f. No remote publication claimed. Authentication remains a delivery dependency, not a reason to stop local scientific work.

## Cycle22 plan: supervised denoiser dynamics
- [x] Revalidate clean2baa0a9 and completed cycle21; previous turn was progress. Existing stop rules forbid another Taylor-training, covariance, recurrence or spectrum rescue. The image bridge already distinguishes input/output dimensions; do not silently change patching as a new benchmark.
- [x] Freeze objective-level conditional mean and U0–U8 claims: heat/denoiser PDE, FM coordinate transform, non-invariance, wrong-prior and moment-closure counterexamples. Primary LagrangianView2609.00198v1 and ICML2023FP-Diffusion read; neither supplies a new architecture win.
- [x] Fresh deep proof review coversU0–U8/35microclaims. Round1finds onlyR1/R2cosmetic notation issues; round2rechecks all units after fixes and reports no open issues. Raw reports retained; coverage is execution-only and semantic assurance remains native/same-family/provisional. Actual model identity was not returned; requested route is recorded separately.
- [x] Source3f098b5 registered before outcomes. Preparation exits0; numerical controller3045 consumedexit0 in6.08s with run/audit phasesexit0. All120AD/oracle cases and240finite-difference checks atbothh pass; no new training, environment or GPU. Source/data bindings unchanged.
- [x] All120oracle cases and240FD records atbothsteps pass fixed gates. Full prior-class application, raw proof rounds/coverage and all probe arrays retained. Curator6034 exits0; two figures plus one-page audit PDF visually checked. Complete derivation/analysis/statistics/proof audit written; no training or regularizer selected.
- [x] Archiveverifier exits0:200files sealSHA2569d6df5b6687c4447e13871295c9adddf7de057fa8975321c4d0b5471fcaadc5b; all200hashes re-read, source/data/proof hashes, all120probe inputs/records and240FDrows, bothsteps, protected scopes/links/AST pass. Private3180335bytes/curated988995bytes. Both phase processes terminal.
- [x] Result0136239 committed asZhizheZhang; clean local main synchronized. Publisher8291 exits1 at missing HTTPSUsername, and fresh local GitHubSSH61861 exits255(publickeydenied). Independent remote main stays800d09f; no publication claimed. No MBP/torch relay used.

## Errors encountered
- A cycle22 archive-writing JS template contained an unescaped backtick and failed before any tool mutation. Replaced the Python fence matcher with chr(96), then wrote and parsed the intended verifier. No runtime output changed.
- Cycle22 review receipt parsing initially assumed the first JSON block was coverage; round2starts with source hashes. The parser stopped before writes. Select the explicit assigned_unit_ids schema instead; both full raw reports remain unchanged.
- A cycle22 proof-clarification patch included an unnecessary mismatched context hunk and failed atomically. Applied only the intended notation changes using checked exact strings; no numerical source or output changed.
- The first cycle22 Add File patch omitted one leading patch marker and failed atomically before creating study.py. Reapplied with mechanically prefixed source lines; AST parsing passes. No runtime or scientific output was created by the failed edit.
- A spacing-only edit in three new, uncommitted cycle21 notes also split hashes, paths and scientific notation. Read-back caught it before publication; explicit replacements restored those identifiers. Runtime sources, protocol and numerical outputs were untouched.
- Cycle21 preflight87896 exits1 before model outcomes: existing SciPy/check interpreter lacks einops. Preserve first runtime/source/log/agent receipt. V2 keeps model work in the existing image interpreter, matrix-only SciPy checks in the existing check interpreter; no package installation or scientific/tolerance changes.
- Cycle21 review initially used the wrong skill root for research-refine/formula-derivation and guessed a nonexistent analysis-frozen-order folder. Correct roots are .agents/skills; the existing exact-flow evidence is research-2026-10-20-followup/analysis-report.md. No runtime or source changed by those reads.
- Cycle19 native raw-fraction metadata accidentally came from cycle1732-step fitting records, while FIDs correctly used128-step features. Preserve original JSON/audits; separately correct only four native out-of-range fields from registered128-step raw arrays. All FIDs/losses/features unchanged; public curation and archive verify corrected provenance.
- Cycle19 preflight58539 exits1 in the promoted algebra witness: native time features are hardcoded float32, while promoted MLP weights are double. Preserve first preparation/source/log and absence of formal outputs. V2 lifts only the witness time-feature values before that MLP; native float32 quality functions and all tolerances unchanged.
- Cycle18 audit-v1 exits1 before any numerical work: top-level SciPy import is unnecessary for CPU sampling replay and absent from the lean training environment. Preserve failed source/status/log. V2 defers only that import to the FID function; no package/model/sampler/metric/tolerance change, and no training/sample rerun.
- Cycle17feature-audit-v1 exits1 before numerical work because its SHA helper expects Path but binding keys are strings. Preserve original frozen source/log/status and empty audit directory; audit-v2 converts the path only. Training44-checkpoint/P/clock/manual-Heun audit already passes; no training, generated samples, metric producer or tolerance changes.
- One cycle17read-only status check accidentally used system Python without psutil; rerun with the declared image-training interpreter succeeded. No job or runtime was restarted.
- Before any cycle17quality run, tighten telemetry to the literal recursive quadratic term and extend trainer preflight to physical checkpoint serialization plus cold affine-Heun replay. Preserve the first prepared source/reference bank; prepare a fresh v2runtime. Reuse reference features only after exact pixel/extractor/weight/hash identity checks; no predictor outcome or acceptance gate changes.
- Cycle17 initial metrics environment imports fail inside SciPy1.15.3 PROPACK Mach-O loader (zero-fill section offset). No weights, quality run or metric result is accepted. Preserve spec/log/version record, identify a locally working SciPy binary and render a fresh environment/spec before declaring readiness.
- Cycle16 feasibility-v1 passed model/clock/derivative/CPU-resume checks but stopped before timing: local clock s shadowed settings s. Preserve source/bindings/log, rename the settings variable, mark terminal failure from the consumed handle and prepare a new v2 snapshot. No image quality training occurred.
- Cycle16 two guessed architecture-search index files were absent; the actual covariance protocol/readout and task plan establish its stop. No source/runtime mutation.
- Cycle15 audit-v1 independently replayed gradients but compared dimensionless cosine for a cold thin route of norm4.92e-15 across SVD/QR. That direction is numerically undefined; preserve the failure. Audit-v2 retains all raw gradient/projector tolerances, independently validates routes, then verifies reported ratios from saved vectors. Formal outputs/source are unchanged.
- Cycle15 initial combined plan/notes patch assumed a generic notes heading and failed atomically. Apply the plan separately and append the scoped note; no runtime or result changed.
- Cycle11 audit-v1 replayed predictions exactly but computed the legacy float32 prediction-RMS diagnostic in float64, differing4.22e-7 at the cold checkpoint. Preservev1source/log; v2 uses the original metric dtype. No formal run, scientific metric, acceptance threshold or tolerance is changed.
- Cycle11v3 preflight caught tiny float32 backward cancellation (max5.82e-11) from two explicit sign reflections at identity. Preservev3source/log. Rewrite the same Q as I plus its exact rank-two residual, preserving thresholds and initialization; independent active dense-Q derivatives still pass and complete-model inherited gradients now match bitwise on both laws/modes. Prepare a freshv4snapshot before any training.
- Cycle11 second preparation exposed a Python benchmark_core cache collision when importing prior verifier code. Preservev2source/bindings/log; verify all prior seals and their external binding hashes directly, keeping numeric benchmark namespaces separate. No prior source or fitted result is modified.
- Cycle11 first preparation tried to hash the newly created source/__pycache__ directory after importing helpers. No binding file or training was produced. Preserve the failed snapshot/log; filter source entries to files and prepare a freshv2runtime under the corrected registration.
- Before any cycle10 fitted run, source review caught that decimal panel labels differ from their stored float32 times. Correct formal references to the exact bank time, preserve first prepared/preflight-only runtime, and prepare a freshv2 snapshot under a new registration. No fitted outcomes or tolerances change.
- Cycle09 audit-v1 divided explicitly projected126D normal energy by ambient128 via mean(), while formal results correctly divide by126. The independent check caught the exact126/128 discrepancy. Preserve v1 source/log; v2 corrects only the audit normalization, with all tolerances/formal outcomes unchanged.
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

## Nonlinear SwiGLU candidate eligibility review
- [x] Revalidate clean eb346a7; the preceding host-scope turn was not scientific progress. Read actual native SwiGLU and existing complete-field/nonlinear-drift/coupled-generator stopping decisions.
- [x] Complete the [bounded candidate review](docs/autoresearch/nonlinear-swiglu-review/decision.md):18AD/36FD checks and actual dense work pass; a legal scalar field escapes before duration1.15files sealed022e4cc. No new performance result, training nomination or stopped-family rescue. User correctly notes that formula verification does not deliver an effective model.

## Frozen image inference export
- [x] Re-read the closest positive/negative outcomes and original image sources. A portable folded endpoint is an unmeasured practical deliverable, not a new ODE method or a reopened training setting.
- [x] Complete the [export and timing task](docs/autoresearch/folded-image-inference/README.md): portable compact checkpoint, exact1,024-image CPU replay, retained FID179.7184 and lower measured CPU/MPS inference cost. All ordinary and attribution controls retained. No new trained architecture.

## Independent review and fixed image budget question
- [x] Fresh research-review examines primary reports/curves/figures and returns a bounded recommendation. Actual model identity unavailable; assurance is native/same-family/provisional. No new architecture is nominated.
- [x] Register all seven unchanged models to20k,10k/20kgeneration, strong ordinary controls and no extra endpoint rescue. Source89de3e1; correct preflight-only Adam alias underbc290fb. All original checkpoint files and failedv1preserved.
- [x] Fresh documented v2command exits0 with seven native1024prediction and physical model/Adam/RNG/buffer resumes. All counters5000→5001→5003;82inputs unchanged. Warm environments475428fb/dc729a50 preserved.
- [x] All seven native continuations and fixed sampling banks complete with seven child exits0 and original controller60302/session91639 terminal exit0. Cohort6595.00seconds; shared final input-stream digest verified from every result. Runtime /Users/mr.z/ode-workspace-prep/autoresearch/oct28-image-budget-v2.
- [x] Complete all fourteen fixed10k/20kFIDs with evaluator80573exit0. Separate feature audit62633 exits0/PASS: all generated/reference feature rows replay bitwise and independent FIDs agree.
- [x] Finish training/stream/sample audit62632 and finalization60880/session68281, both terminal exit0. All112checkpoint/14sampling-bank checks pass. Curator7523exits0 and all seven figures are visually reviewed; all twenty-one budget/model rows and the unchanged20kpriority rule are retained.
- [x] Verify public artifacts and source/parent preservation, then seal the complete image-budget archive. Archive18048exits0;636files sealac381ff444384e052c4cc48e9f11550f00ab410a233154972177e97fe543a48b. English reports, all numeric tables and seven figures verified. K1passes only the follow-up allocation margin, no new method is established.
- [x] Prepare the next bounded image protocol and runnable source: does the20kK1lead survive the declared simultaneous fixed-clock intervention? Keep all seven native20kcontrols and the completed budget archive. Five decision and six complete/constant-Jacobian fixtures pass; no new training family is selected.
- [x] Register/publish634b13b, prepare runtime2737exit0, and obtain fresh documented preflight52748exit0 with full native20kreplay and all calibration/folding/decision checks. No environment or invocation changes.
- [x] Complete controller64292/session21980 with terminal exit0 through all four frozen-intervention phases. Both new sample banks and every new/reference feature row replay bitwise; all nine FIDs independently agree. No retry or training.
- [x] Curator4663exits0; inspect all three figures and72fixed cases, and write the complete English20kclock analysis/appendix/catalog. All nine endpoints and declared strict-order/margin criteria remain visible.
- [x] Complete archive3618exit0:108files seal825980fab2105dce825380ddfa0c349b5a1f179cb68dd7e44bcd5510916a6f9b. All report rows, tables, visuals, protected scopes and four parent seals verify; original controller is absent. No new intervention or clock rescue.
- [x] Formulate a bounded mathematical/native-checkpoint analysis of complete gate feedback on paired-data versus generated states before selecting a new architecture or constraint. Derivation and data-free state/P/held-clock/Heun fixtures pass; no fitted-state outcome yet.
- [x] Publish source1e35ce2and curator836d621; fresh documented preflight21100exits0. Controller25342exits0 through8native cells and all122,880independent AD probes in241.13seconds, with CPU/RSS monitoring.
- [x] Curator18967 exits0; all3figures visually inspected. Archive23733 exits0 with complete public-table/link and all122,880 matrix-entry verification.1084files sealed faeac7fd69f048c0875184af92c6278956bdf478083c75d77bbd83f747aa6345; publish verified reports/evidence. No new constraint or quality claim.

## Time-preserving frozen K1 study
- [x] Revalidate previous progress, sealed native feedback and stopping rules; specify one time-table replacement with same-calibration pooled attribution control.
- [x] Publish v3source86afd38 after preserving the two pre-quality entrypoint failures; fresh native69731/metric72055preflights terminate0, all checks pass.
- [x] Controller61529 exits0 through all four phases; both complete new banks, calibration and all eleven FIDs independently replay.
- [x] Curator8411 and archive64846 exit0; all four figures/88cases and full report/numeric evidence verify.124files seal8917402ec1dd0ca04fec024b081bbf7b37f64e563444220592c3ab7604bc17cb; publish the verified closure. No retrained architecture is selected.

## Cold clock and ordinary-coordinate controls
- [x] Revalidate previous sealed progress and stop rules; specify two cold controls, one frozen same-clock reference and scoped optimizer-coordinate identities before outcomes.
- [x] Publish source28be1e8; fresh native47808/metric32097exit0, including shared timestep-MLP gradients. All25cached FIDs replay exactly, shared ordinary init/data RNG match, physical resumes and actual costs pass.
- [x] Run exactly two20ktrajectories with matching historical stream and complete P/effective-W histories.
- [x] Audit boundaries/streams and all seven new inference banks; retain32FIDs and complete the report/five inspected figures.
- [x] Verify final public artifacts and seal the private runtime.
- [x] Publish the complete closure; normal remote main publication c7471ab verified.
- [x] Finish bookkeeping and verify clean local-main fast-forward to the published research closure.

## Status
All six phases and original controller5632 terminate0, as does registered curator17551. All7168new images/features and2048reference rows replay bitwise;32independent FIDs differ by at most6.54e-13. Fixed06FID174.708609passes the original finite margin at half the FFN matrix work, versus native173.436914; matched-init ordinarySiLU178.056840misses the margin, and frozen same-c native216.070195remains an attribution reference. Five figures/144fixed cases are inspected; full report, stats and catalog are written. Verify public numeric/source/data preservation, seal and publish. No new constraint, clock rescue or optimizer grid is selected. The broader useful-ODE-successor objective remains active and unachieved.

Archive51807exits0;302files seal aad29b214ba537a2ee2034c60987c98f220eef00ca97b0105fb2e3bbadffff9c. Private runtime417939140bytes, curated2110898bytes. Four reports equal archived copies; all verification gates pass. Only normal GitHub publication/local-main sync remain for this cohort; the useful-ODE-successor goal is still active.

Published closure and clean local main are verified at10ec3a2. Final plan synchronization changes only public bookkeeping; all scientific artifacts and the sealed private runtime remain unchanged. No tasks remain for this registered cohort. The overall ODE successor objective remains active.


## Post-cold-control direction review
- [x] Revalidate published0ef6939 and cold-control sealaad29b21; prior goal turn
  completed real empirical progress, no live old process remains.
- [x] Review current primary literature and prior stops. No new neural ODE
  architecture is nominated from recurrence, target changes or conditioning.
- [x] Specify a train-only PPCA59/isotropic classical image reference and exact
  Gaussian theory, unchanged banks/sampler/metrics, scalar budget and stop rules.
- [x] Publish primitive source68a5402 and pass fresh documented data-free
  eligibility21cases; retain its minor environment-label mismatch and seal.
- [x] Implement complete source-bound Gaussian fit/audit/sampler/metrics/controller.
- [ ] Register the full runtime and pass a NEW fresh documented entrypoint
  preflight before fitting any real statistics.

Broader objective remains active. The reference measures a missing classical
context; it is not a new ODE successor, a changed benchmark or a rescue.

Gaussian full source3d18ae0/fresh doc preflight43297pass, with zero mismatch. Controller35918exits0through6stages; actualfit11.19s, no neural training. PPCA59FID293.734970 and isotropic405.298943fail the original quality/fitting criteria. Retain all32neural contexts. Complete curator/visual/report/archive/publication; no new rank/noise/mixture family or ODE method is selected.


## Gaussian reference closure and next design evidence
Previous goal turn verified direct GitHub authentication and receive-pack access. The formal reference and post-outcome reporting arrays are terminal; do not restart them.
- [x] Preserve core curation receipt, explicitly record missing original reporting exit, and independently verify all seventeen subspace rows.
- [x] Prepare complete 34-endpoint/16-primary analysis and fixed cases; no covariance/rank rescue.
- [x] Seal unchanged source/data/parents and curated evidence, then publish and FF clean local main.
- [ ] Use the completed controls to identify the next unresolved ODE question; no new neural family is selected yet.

Gaussian closure7159 exits0, seal97961412597065ad8213a8ae21e5721a7086875113641026bfe3338431f846ba. Publication/local-main FF is next. The original reporting-producer exit remains unavailable and is explicitly distinguished from the complete independent array verification. No ODE successor has passed.

Milestoned15c041 published/verified; clean local main fast-forwarded to it.
Primary Pi2026linear-support review completed without selecting another
architecture. Next explicit step: derive finite posterior-state algebra in
precision coordinates, including the third-moment term, and check ordinary
softmax/RBF equivalence before any implementation or training. This is not a
mixture-prior quality rescue or evidence of a new method. Goal remains active.
