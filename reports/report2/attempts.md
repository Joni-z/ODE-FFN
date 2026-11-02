# ODEFFN attempt ledger

Zhizhe Zhang · 2 November 2026

This is the complete evidence map supporting the short Report2, not a leaderboard.
It covers every area in [attempt-inventory.json](attempt-inventory.json), additional
root-level architecture studies, and the legacy ImageNet configuration inventory.
Entries are questions/evidence families, not independent training runs; some reuse
identical controls or describe the same study at a different analytical level.

The mainline is the professor's two Taylor-flow maps around SiLU. Ordinary
SiLU/SwiGLU and free ISRU are controls or derivative leads. A successful ordinary
control is not credited as an ODE-specific result. Training, frozen interventions,
teacher regression, proof fixtures, proposals and engineering are labeled separately.

Earlier Report1 material is included only as context. Report1 remains final.
All small-image rankings retain the later baseline correction: no architecture
quality verdict follows before a competent ordinary baseline. Historical stop
decisions describe their exact screens, not impossibility of an entire direction.
The source links preserve exact settings, metrics, negative outcomes and limits.

Machine-readable versions: [JSON](attempt-ledger.json), [CSV](attempt-ledger.csv).
The [three-page report](report2.pdf) selects evidence that changes the scientific
argument. No training was launched to prepare this ledger.

## 1. Original P, clocks and polynomial order

| ID / evidence type | Purpose and change | Observation and limit | Primary evidence |
|---|---|---|---|
| E001 · training | Compare original K1/K2, spectral caps and ordinary SiLU. | No uniform order/cap advantage; field and SW2 rankings differ. Prior Report1 context. | [source](../../docs/analysis-2026-10-13/analysis-report.md) |
| E051 · training and theory | Add the missing SwiGLU baseline and inspect learned P and full derivatives. | Ordinary controls are required before crediting P constraints; earlier SiLU-only comparisons are insufficient. | [source](../../docs/research-2026-10-19/README.md) |
| E052 · training | Cross input/time clocks and K1/K2/K4/K8; add longer ordinary controls at matched matrix work. | Same-step gains change under work matching; K8 better approximates exp without uniformly better generation. Prior Report1 context. | [source](../../docs/research-2026-10-20/analysis-output/analysis-report.md) |
| E053 · frozen intervention | Replace learned polynomial order with higher orders/exp and vary outer sampler resolution. | K1/K2-to-exp worsens every initial target; K8 is near-exp. P eigenvalues/norm do not determine transient gain. Prior Report1 context. | [source](../../docs/research-2026-10-20-followup/analysis-report.md) |
| E054 · frozen derivative diagnostic | Hold gate values at each anchor and remove only their input derivative. | Clock sensitivity can improve full-field derivatives; this is not a trained gate-removal result. | [source](../../docs/research-2026-10-21-gate-geometry/findings.md) |
| E055 · training and case study | Compare input clocks, fixed duration, learned scalar, SiLU and SwiGLU. | Input clocks reduce Ring8 bridges versus constant clocks; they do not establish a general advantage over SwiGLU, and clock removal reduces capacity. | [source](../../docs/research-2026-10-21-gates/findings.md) |
| E058 · training and intervention | Remove only the centered quadratic clock term while preserving static and first-order response. | Retraining recovers much field fit; derivative/sample changes remain mixed. The special clock curvature is not an established advantage. | [source](../../docs/research-2026-10-24-lenovo/findings.md) |

## 2. Local operator and controller alternatives

| ID / evidence type | Purpose and change | Observation and limit | Primary evidence |
|---|---|---|---|
| E062 · study index | Collect staged static/rate/controller designs and their scientific controls. | Substudies and root-level readouts are enumerated below; this directory is not an additional independent experiment. | [source](../../docs/research-architecture-search/task_plan.md) |
| E063 · training | Condition h rather than Ph using additive instead of multiplicative rates. | Vector additive improves Product16 over its parent but remains behind ordinary controls and worsens Ring8. | [source](../../docs/research-architecture-search/analysis-additive-generator/analysis-report.md) |
| E064 · training | Add static versus clock-modulated projection bias. | Both remain close to the parent and fail to close the nonlinear-task gap. | [source](../../docs/research-architecture-search/analysis-affine-forcing/analysis-report.md) |
| E065 · training | Factor P and spend a matched budget on a larger hidden state. | Factorization hurts; expanding the state does not recover that loss at the tested budget. | [source](../../docs/research-architecture-search/analysis-augmented-state/analysis-report.md) |
| E066 · control training | Copy ODE surrounding FiLM tensors/buffers into ordinary baselines. | Baseline ordering changes, but the better ordinary reference remains stronger. Earlier unpaired contrasts cannot isolate the FFN alone. | [source](../../docs/research-architecture-search/analysis-backbone-pairing/analysis-report.md) |
| E067 · frozen intervention | Replace trained vector clocks with their channel means. | All saved panels worsen; clock activity matters at these weights, but a cold constant model was not tested here. | [source](../../docs/research-architecture-search/analysis-clock-collapse/analysis-report.md) |
| E068 · training | Rescale clock coordinates while preserving initial function and function class. | Ring8/Product16 improve by 11.1%/33.6% versus parent, without catching stronger SwiGLU controls. | [source](../../docs/research-architecture-search/analysis-clock-coordinates/analysis-report.md) |
| E069 · frozen diagnostic | Project the complete denoiser Jacobian onto symmetric PSD geometry. | Only 1.24% of Product16 parent Jacobian error is structural distance; symmetry alone does not explain most error. | [source](../../docs/research-architecture-search/analysis-complete-field/README.md) |
| E070 · frozen diagnostic | Test pointwise gate directions and shared-parameter tangent fits. | Pointwise span is sufficient in vector models; tiny-bank tangent improvements do not transfer reliably. | [source](../../docs/research-architecture-search/analysis-controller-tangent/analysis-report.md) |
| E071 · training | Use a tied covariance-shaped field, untied decoder and static tangent control. | Internal energy behavior holds, but ordinary controls fit better; untying improves Ring8. | [source](../../docs/research-architecture-search/analysis-covariance/analysis-report.md) |
| E072 · training | Add bounded diagonal K1/K2/exact flows around an expanded SiLU core. | Numerical identities hold; adaptive K2/exp do not improve the practical ordinary baseline. | [source](../../docs/research-architecture-search/analysis-diagonal-flow/analysis-report.md) |
| E073 · training and evaluation | Continue scalar/vector K1 and controls; test rotated Ring8 in 16D. | Vector K1 improves Ring8 field fit but not derivative/sample frontier, and the field advantage does not transfer. | [source](../../docs/research-architecture-search/analysis-followup/analysis-report.md) |
| E074 · training | Add an exact or Taylor forced-response correction to SwiGLU's gate. | The learned correction supplies no useful improvement over the same-width ordinary control. | [source](../../docs/research-architecture-search/analysis-forced-gate/analysis-report.md) |
| E075 · frozen diagnostic | Measure the target error forced by the saved output range. | Range floors explain zero Ring8 and only 0.0992% of Product16 parent field error. | [source](../../docs/research-architecture-search/analysis-head-envelope/README.md) |
| E076 · training | Cross map/increment returns in both solves with paired initial functions. | Identity location changes behavior but no placement closes the strong-baseline gap. | [source](../../docs/research-architecture-search/analysis-identity-location/analysis-report.md) |
| E077 · training and diagnostic | Test scalar/vector rates on rotated independent bimodal Product16. | At 20k the vector model remains about 4.45 times the full-SwiGLU test field error. | [source](../../docs/research-architecture-search/analysis-operator/analysis-report.md) |
| E078 · training | Use identical population inputs with pair versus exact conditional-mean labels. | Noise removal helps but does not close the ODE gap; this is not an expressivity lower bound. | [source](../../docs/research-architecture-search/analysis-population-noise/analysis-report.md) |
| E079 · frozen diagnostic | Measure feature anisotropy, gradients and scale-corrected response geometry. | Both model families are anisotropic; the results do not select P-only whitening. | [source](../../docs/research-architecture-search/analysis-projection-geometry/analysis-report.md) |
| E080 · training and diagnostic | Relax positive clock range or use an affine clock. | No nonlinear-task rescue at 5k; frozen response-space fits are separate diagnostics. | [source](../../docs/research-architecture-search/analysis-range/analysis-report.md) |
| E081 · training | Move vector rates before P rather than after it. | PD worsens Ring8 and Product16 versus the DP parent and remains behind SwiGLU. | [source](../../docs/research-architecture-search/analysis-rate-order/analysis-report.md) |
| E082 · control training | Test ordinary expanded SiLU at the parent's parameter/matrix-work budget. | Product16 improves over vector ODE while Ring8 worsens; SwiGLU remains stronger on both nonlinear laws. | [source](../../docs/research-architecture-search/analysis-reference-silu/analysis-report.md) |
| E083 · training | Share P with controller features, versus independent wider features. | Sharing worsens both nonlinear tasks; wider independent clocks do not recover the ordinary frontier. | [source](../../docs/research-architecture-search/analysis-shared-features/analysis-report.md) |
| E084 · control training | Replace SwiGLU's SiLU gate with sigmoid at the same tensors/budget. | Gaussian improves slightly; Ring8/Product16 field errors worsen substantially. | [source](../../docs/research-architecture-search/analysis-sigmoid-glu/analysis-report.md) |
| E085 · training | Reevaluate the clock at a midpoint instead of freezing it during the solve. | The extra state-clock derivative is active, but midpoint loses the tested quality/compute comparison. | [source](../../docs/research-architecture-search/analysis-state-clock/analysis-report.md) |
| E086 · training | Return only the increment from both original solves. | Product16 improves, including a paired-init control, but Gaussian/Ring8 worsen and SwiGLU remains stronger. | [source](../../docs/research-architecture-search/analysis-zero-order/analysis-report.md) |
| E117 · training | Compare fixed K2 with function-matched free effective matrices. | Ring8 field MSE .076519 to .028901 with fewer MACs; free maps still lose to SwiGLU. | [source](../../docs/research-architecture-search/evidence/static-summary.md) |
| E118 · training | Cross scalar/vector clock granularity and K1/K2 with paired non-gate tensors. | Vector K1 improves local screens; longer and Product16 follow-ups do not establish a broad frontier. Controller width changes with granularity. | [source](../../docs/research-architecture-search/protocol-rates.md) |
| E119 · training | Remove only the middle SiLU, leaving nonlinear clocks. | Ring8 error worsens for scalar and vector models; a cleaner zero-generator limit is not a demonstrated benefit. | [source](../../docs/research-architecture-search/composition-readout.md) |
| E120 · training | Share controller weights or reuse one duration across both solves. | Reuse saves parameters/MACs but fails Ring8 quality relative to budget-matched SwiGLU. | [source](../../docs/research-architecture-search/clock-reuse-readout.md) |
| E121 · training | Replace split linear maps by a nonlinear Euler drift; compare map and increment output. | Increment helps Product16 versus the parent but remains behind ordinary controls. | [source](../../docs/research-architecture-search/nonlinear-drift-readout.md) |
| E122 · frozen oracle diagnostic | Compare the exact cached-target conditional mean with the population oracle. | Late-time finite-bank concentration makes the two targets substantially different; this motivated the fresh-population noise control. | [source](../../docs/research-architecture-search/empirical-target-readout.md) |
| E123 · frozen derivative diagnostic | Remove clock input sensitivities at selected first/second solves while retaining anchor predictions. | Derivative contributions depend on task and solve location and are not additive; this is not gate-removal training. | [source](../../docs/research-architecture-search/gate-placement-readout.md) |
| E124 · frozen diagnostic | Partition Product16 derivative errors by posterior ambiguity and latent geometry. | Conditional expansion/contraction regimes locate derivative error; they do not select a trained correction. | [source](../../docs/research-architecture-search/posterior-regions-readout.md) |
| E125 · frozen diagnostic | Partition velocity-error excess by posterior entropy. | About78% of excess lies in confident regions; ambiguous component assignment alone is insufficient. | [source](../../docs/research-architecture-search/posterior-velocity-regions-readout.md) |
| E126 · proposal review | Review coupled generators, state clocks, diagonal flows, covariance, head and feature-sharing alternatives before training. | Root design/protocol files link to the executed families above; proposals without a result remain proposals. | [source](../../docs/research-architecture-search/state-clock-design-review.md) |

## 3. Whole-network dynamics, adjoints and local fitting

| ID / evidence type | Purpose and change | Observation and limit | Primary evidence |
|---|---|---|---|
| E059 · frozen intervention | Decompose the complete gradient readout into cross-block reverse paths. | Removing the path through all three residual derivatives raises ODE Product16 MSE .03732 to .20266; SwiGLU also depends on it. | [source](../../docs/research-adjoint-composition-2026-10-26/results/analysis-report.md) |
| E060 · frozen diagnostic | Separate raw-affine and nonlinear field-error components. | Only about 3.9% of the direct ODE Product16 excess is affine; a physical-coordinate linear repair is insufficient. | [source](../../docs/research-affine-error-2026-10-26/analysis-report.md) |
| E061 · training | Compare shared accumulating updates with corrections anchored to the original state. | Anchoring worsens Product16 for ODE and ordinary controls; no practical candidate is selected. | [source](../../docs/research-anchored-corrector-2026-10-26/results/analysis-report.md) |
| E087 · teacher regression | Place one matched bias before the activation or at the output. | Biases become active but do not recover the local SwiGLU-teacher fitting gap. | [source](../../docs/research-bias-placement-2026-10-26/analysis-report.md) |
| E088 · training | Predict the full input gradient of one-half x dot G instead of G. | Original ODE Product16 MSE falls .325527 to .037320; gradient Swi192 reaches .033826 and Gaussian generation worsens. | [source](../../docs/research-conservative-readout-2026-10-26/results/analysis-report.md) |
| E089 · frozen diagnostic | Separate unwanted cross-coordinate dependence from own-coordinate response error. | Unrelated-coordinate dependence explains at most a small fraction of the Product16 gap. | [source](../../docs/research-coordinate-dependence-2026-10-26/analysis-report.md) |
| E090 · training | Use tied linear/cubic/linear subflows with SiLU and free-ISRU controls. | Tied cubic improves Product16 over original ODE but loses to Swi160; free ISRU wins this task without retaining generator coupling. | [source](../../docs/research-cubic-split-2026-10-26/analysis-report.md) |
| E091 · frozen teacher diagnostic | Measure clocks and curvature on self-teacher versus SwiGLU-teacher fits. | Clock dependence is learnable on own teachers; collapsing nearly saturated clocks on harder targets leaves the main gap. | [source](../../docs/research-data-curvature-2026-10-26/analysis-report.md) |
| E092 · frozen diagnostic | Check gradient cancellation and omitted depth-dependent P directions under sharing. | Sharing restricts directions, but the registered severity criterion fails and the effect is not ODE-specific. | [source](../../docs/research-depth-gradient-2026-10-26/analysis-report.md) |
| E093 · training | Compare radial J_G x with the transposed J_G^T x readout. | Radial sensitivity does not recover ODE's complete-gradient benefit; the same construction behaves differently with SwiGLU. | [source](../../docs/research-derivative-orientation-2026-10-26/results/analysis-report.md) |
| E094 · teacher regression | Learn effective W matrices directly instead of their P polynomial coordinates. | Free matrices do not close the local SwiGLU-target gap or consistently beat P on realizable targets. | [source](../../docs/research-effective-map-2026-10-26/analysis-report.md) |
| E095 · identity and intervention | Implement the complete primal/adjoint calculation and omit clock sensitivity at frozen weights. | Explicit calculations reproduce the full derivatives; omitted clock sensitivity changes the learned field, without a new trained method. | [source](../../docs/research-explicit-adjoint-2026-10-26/results/analysis-report.md) |
| E096 · frozen regression probe | Refit an affine head on frozen final features. | Only about 1% of the ODE Product16 gap closes; a final linear readout fix is insufficient. | [source](../../docs/research-frozen-features-2026-10-26/analysis-report.md) |
| E097 · training | Cold-train full versus gate-path-omitted explicit adjoints in all families. | Omitting sensitivity hurts all backbones; ordinary controls suffer larger relative penalties, weakening ODE-specific attribution. | [source](../../docs/research-gate-adjoint-2026-10-26/results/analysis-report.md) |
| E098 · frozen diagnostic | Test cancellation between linear and quadratic P-gradient routes. | Severe cancellation is absent in the examined trained scalar-K2 model; this is not a global conditioning theorem. | [source](../../docs/research-generator-tangent-2026-10-26/analysis-report.md) |
| E099 · frozen regression probe | Read out controller, middle-activation and residual features with affine probes. | No probed site exposes a large recoverable linear correction of the remaining Product16 error. | [source](../../docs/research-internal-features-2026-10-26/analysis-report.md) |
| E100 · training | Retest unchanged models with dependent component signs and matched low-order marginals. | Free ISRU field ranking reverses; original ODE remains behind ordinary controls. Sample SW2 has a different ordering. | [source](../../docs/research-joint-dependence-2026-10-26/analysis-report.md) |
| E101 · training | Reuse local reverse sensitivity within an ODE-inspired FFN. | Original ODE improves to .190290 Product16 MSE with fewer parameters, but tied ordinary and SwiGLU controls remain better. | [source](../../docs/research-local-adjoint-2026-10-26/analysis-report.md) |
| E102 · teacher regression | Fit one frozen FFN on common hidden inputs without whole-network alignment. | The ODE/SwiGLU fitting gap already appears locally; finite optimization does not prove impossibility. | [source](../../docs/research-local-ffn-teacher-2026-10-26/analysis-report.md) |
| E103 · frozen intervention | Remove the K2 path that leaves and returns from the uniform-channel direction. | Removal worsens all three field tasks; whole-branch shift invariance does not extend to internal P products. | [source](../../docs/research-mean-return-2026-10-26/results/analysis-report.md) |
| E104 · frozen regression probe | Fit known conditional-response bases to the original model's residual. | Linear plus posterior nonlinear response explains most residual; an oracle-assisted correction is not a learned architecture gain. | [source](../../docs/research-mode-response-2026-10-26/analysis-report.md) |
| E106 · frozen intervention | Remove radial, tangent or uniform-shift components of hidden branches. | Uniform shifts can be exactly unobservable, while deleting radial/tangent components worsens predictors. | [source](../../docs/research-normalized-geometry-2026-10-26/results/analysis-report.md) |
| E107 · frozen diagnostic | Decompose even, linear-odd and nonlinear-odd prediction error. | The nonlinear deficit is real; learned SiLU states violate the small-response simplification proposed to explain it. | [source](../../docs/research-parity-2026-10-26/analysis-report.md) |
| E108 · training | Cross direct velocity and clean-data prediction under a common protocol. | No original scalar K1/K2 arm meets the registered continuation criteria; this is not a general rejection of clean prediction. | [source](../../docs/research-prediction-bridge-2026-10-25/notes.md) |
| E109 · teacher regression | Cold-fit an own-architecture teacher and a SwiGLU teacher. | ODE fits its own teacher much better; the comparison separates neither global expressivity nor optimization completely. | [source](../../docs/research-realizable-teacher-2026-10-26/analysis-report.md) |
| E110 · training | Cross shared/cloned/independent blocks and final/intermediate supervision. | No original-ODE quality/parameter frontier passes; sharing and deep supervision affect ordinary controls too. | [source](../../docs/research-recurrent-depth-2026-10-26/results/analysis-report.md) |
| E111 · training | Combine shared hidden dynamics with the complete potential-gradient readout. | Shared ODE MSE .233525 versus cloned independent .034229; the useful potential fit does not survive this sharing. | [source](../../docs/research-recurrent-potential-2026-10-26/results/analysis-report.md) |
| E112 · training | Add a matched static or time-dependent operator adapter. | Time-dependent P does not establish a practical lead over ordinary controls in the tested screen. | [source](../../docs/research-time-generator-2026-10-26/results/analysis-report.md) |
| E113 · frozen diagnostic | Compare shared versus time-dependent tangent directions at a fixed budget. | Temporal local directions help across families, but the tested finite displacement overshoots and is inconclusive. | [source](../../docs/research-time-operator-2026-10-26/results/analysis-report.md) |

## 4. Exact flows, manifold geometry and objective-level checks

| ID / evidence type | Purpose and change | Observation and limit | Primary evidence |
|---|---|---|---|
| E003 · frozen intervention | Add the complete nonlinear Taylor correction, midpoint or half steps to fitted residual blocks. | All-law post-training upgrades worsen error; this does not compare cold-trained solver families. | [source](../../docs/autoresearch/cycle01-complete-field/analysis.md) |
| E004 · training | Mix independent/dependent sign laws while retaining every original component. | ISRU loses to stronger SwiGLUs despite full component support; literal component count alone is insufficient. | [source](../../docs/autoresearch/cycle02-full-support/analysis.md) |
| E005 · identity control | Realize ordinary SwiGLU as the exact endpoint of an augmented square-zero ODE. | Values, gradients and a paired training trajectory match; exact ODE form alone adds no computation or quality benefit. | [source](../../docs/autoresearch/cycle03-gated-shear/analysis.md) |
| E006 · training | Add exact nilpotent gate transport and ordinary capacity controls. | Extra gate interactions do not beat matched ordinary capacity. | [source](../../docs/autoresearch/cycle04-gate-transport/analysis.md) |
| E007 · training | Integrate nonlinear activated features along an exact affine hidden path. | Accurate two-node integration loses to ordinary capacity; the midpoint control is algebraically absorbable. | [source](../../docs/autoresearch/cycle05-feature-flow/analysis.md) |
| E008 · frozen intervention | Replace the fitted two-node response by its midpoint at unchanged weights. | Curvature changes full-model risk only slightly and with mixed signs. | [source](../../docs/autoresearch/cycle06-curvature-attribution/analysis.md) |
| E009 · training | Embed Ring8 in 128D; cross thin/thick data and clean/shifted/velocity heads. | Original ODE becomes competitive under thin clean prediction, but the registered margin and thick-data guard fail. | [source](../../docs/autoresearch/cycle07-manifold-heads/analysis.md) |
| E010 · proof and replay | Derive directions the final width64 normalized readout cannot change. | At least 65 ambient directions obey an affine sampling law shared by ODE and SwiGLU. | [source](../../docs/autoresearch/cycle08-readout-invariant/analysis.md) |
| E011 · frozen diagnostic | Compute the analytic optimum given each learned input projection. | Only 3-5% of the thin comparison gap is input-information loss; most remains after conditioning. | [source](../../docs/autoresearch/cycle09-input-attribution/analysis.md) |
| E012 · frozen diagnostic | Decompose complete conditional-field derivative residuals. | Late tangent error is overwhelmingly symmetric; a skew-only explanation is unsupported. | [source](../../docs/autoresearch/cycle10-field-derivatives/analysis.md) |
| E013 · training | Rotate the original P polynomial in an input-dependent orthogonal frame. | Thin modes recover and fitting improves; K1 nearly matches, while thick sampling fails the practical comparison. | [source](../../docs/autoresearch/cycle11-conditional-frame/analysis.md) |
| E014 · frozen diagnostic | Telescope native sampling paths against remaining oracle solver suffixes. | Thin discrepancy is mostly before the last quarter; much thick normal error is shared readout restriction. | [source](../../docs/autoresearch/cycle12-path-defects/analysis-report.md) |
| E015 · frozen intervention | Approximate fitted K2 action with scalar clocks or a fixed absorbed matrix. | The quadratic action is partly absorbable but not uniformly redundant; this does not establish a retrained K2 benefit. | [source](../../docs/autoresearch/cycle13-quadratic-action/analysis-report.md) |
| E016 · training | Untie an added second Taylor coefficient C with affine/free-gate controls. | Thin fitting improves modestly, but thick and mechanism guards fail; active C is not evidence of useful ODE coupling. | [source](../../docs/autoresearch/cycle14-independent-coefficient/analysis-report.md) |
| E017 · frozen diagnostic | Trace normal/tangent loss gradients into P and global clipping. | A readout-forced component has zero direct hidden gradient; indirect clipping effects are state-dependent. | [source](../../docs/autoresearch/cycle15-gradient-routes/analysis-report.md) |
| E024 · proof and numerical fixtures | Distinguish clean conditional means, endpoint maps and posterior precision dynamics. | An ideal denoiser need not be trajectory-invariant; covariance-only evolution omits a third-moment term. | [source](../../docs/autoresearch/cycle22-denoiser-dynamics/analysis-report.md) |
| E035 · training | Use an exact state-dependent norm-preserving hidden rotation. | Field errors remain far above near-cost SwiGLU; exact amplitude preservation does not imply a task advantage. | [source](../../docs/autoresearch/invariant-clock-flow/results/analysis-report.md) |
| E038 · eligibility review | Check the full SwiGLU Taylor term, matrix cost and finite escape. | Derivatives are valid but no task-specific premise justifies another training run; no model was trained. | [source](../../docs/autoresearch/nonlinear-swiglu-review/decision.md) |
| E040 · research review | Reassess whether image clock observations justify a posterior-state mechanism. | Proposal and transfer-assumption review only; not an additional quality experiment. | [source](../../docs/autoresearch/post-cold-control-review/research-question-card.md) |
| E041 · eligibility review | Check finite posterior-state ODE closures and exact endpoint realizations. | The derived endpoints admit ordinary realizations; no trained architecture advantage was tested. | [source](../../docs/autoresearch/posterior-state-eligibility/README.md) |

## 5. Image bridge and controls: quality gate unresolved

| ID / evidence type | Purpose and change | Observation and limit | Primary evidence |
|---|---|---|---|
| E002 · control training | Pair cold feature-conditioned and external-time-conditioned K1 clocks. | The small recorded FID gain misses the historical margin; absolute quality prevents effectiveness claims. | [source](../../docs/autoresearch/cold-clock-source/results/analysis-report.md) |
| E018 · implementation | Install literal input-clock professor blocks in a tiny attention backbone. | Feasibility checks pass; this stage itself supplies no quality result. | [source](../../docs/autoresearch/cycle16-image-bridge/research-question-card.md) |
| E019 · training | Compare original K1/K2 and two SwiGLUs in tiny CIFAR JiT. | Historical poor-FID ordering is retained, but cannot establish architecture effectiveness or rejection. | [source](../../docs/autoresearch/cycle17-cifar-baselines/analysis-report.md) |
| E020 · frozen sampling | Increase outer sampling resolution for the tiny image endpoints. | Ranking survives tested meshes with smaller gaps; inadequate absolute quality still blocks architecture claims. | [source](../../docs/autoresearch/cycle18-image-resolution/analysis-report.md) |
| E021 · frozen intervention | Collapse feature clocks at the 5k image checkpoint. | Recorded FID improves after collapse; this does not show cold fixed-clock training equivalence or image competence. | [source](../../docs/autoresearch/cycle19-image-clocks/analysis-report.md) |
| E022 · control training | Add ordinary SiLU at near-parameter and equal matrix-work budgets. | Ordinary work-matched SiLU removes the historical K2 lead; all image scores remain inadequate. | [source](../../docs/autoresearch/cycle20-image-silu/analysis-report.md) |
| E023 · frozen intervention | Replace fixed-clock polynomials by matrix exponentials. | K1 worsens and K2 has mixed fitting/generation changes; no exact-flow advantage is established. | [source](../../docs/autoresearch/cycle21-image-exponential/analysis-report.md) |
| E025 · deployment | Fold fixed-clock FFNs into two effective matrices. | Exact sample replay and faster inference are engineering results, not a new trained ODE method. | [source](../../docs/autoresearch/folded-image-inference/README.md) |
| E026 · training continuation | Continue all seven tiny image models to 20k with original optimizer/RNG state. | K1 has the lowest recorded endpoint FID, but every FID worsens from10k to20k and quality is inadequate. | [source](../../docs/autoresearch/image-budget-continuation/analysis-report.md) |
| E027 · control training | Train fixed clocks and ordinary matched-initialization SiLU from cold states. | Cold fixed models differ from frozen collapse; ordinary coordinates/initialization must be controlled. FIDs remain inadequate. | [source](../../docs/autoresearch/image-clock-training/analysis-report.md) |
| E028 · frozen intervention | Repeat clock collapse after longer image training. | The 5k conclusion does not transfer: collapse now worsens scores; cold training remains a distinct question. | [source](../../docs/autoresearch/image-clocks20k/analysis-report.md) |
| E029 · frozen derivative diagnostic | Measure complete clock feedback and feature/time variation. | Gate sensitivity grows but typical norm effects are small; P-only penalties are not justified by these probes. | [source](../../docs/autoresearch/image-gate-feedback/analysis-report.md) |
| E030 · classical reference | Fit train-only isotropic/PPCA conditional means. | These two classical references fail the recorded comparison; no ODE mechanism follows. | [source](../../docs/autoresearch/image-gaussian-reference/analysis-report.md) |
| E031 · frozen diagnostic | Check necessary response properties in common observed directions. | The common-direction mechanism premise fails; own-space violations are not a causal FID explanation. | [source](../../docs/autoresearch/image-response-geometry/results/analysis-report.md) |
| E032 · frozen diagnostic | Compare models against an exact finite-bank mean on shared sampler states. | No registered reversal; the bank reference fails its held-out-use guard. | [source](../../docs/autoresearch/image-state-risk/results/analysis-report.md) |
| E033 · control training | Match the initial function of SwiGLU to the ordinary SiLU control. | Recorded FID improves but remains poor; initialization affects the screen without validating an ODE mechanism. | [source](../../docs/autoresearch/image-swiglu-initialization/results/analysis-report.md) |
| E034 · frozen intervention | Use a time profile instead of feature clocks or a matched pooled constant. | Time profile improves over the pooled control but not the native practical margin; no trained quality claim. | [source](../../docs/autoresearch/image-time-clocks/analysis-report.md) |
| E036 · training snapshot | Transfer free ISRU to a larger bespoke JiT with matched and wider SwiGLU. | The preserved 25k snapshot has FIDs252.96/262.93/268.74; no image advantage is established. Not a live status claim. | [source](../../docs/autoresearch/isru-cifar10/README.md) |
| E037 · evidence review | Recover the exact winning ISRU operator and all target-law reversals. | A Product16 lead remains; it is a free-matrix configuration rather than proof of original-ODE superiority. | [source](../../docs/autoresearch/isru-followup/README.md) |
| E039 · optimizer identity | Transport effective matrices and real Adam moments between P and W coordinates. | Next updates match numerically; this is a coordinate identity, not an identified cause of quality differences. | [source](../../docs/autoresearch/p-coordinate-steps/analysis.md) |

## 6. Mathematical reviews, audits and implementation support

| ID / evidence type | Purpose and change | Observation and limit | Primary evidence |
|---|---|---|---|
| E042 · archived proposals | Retain frequency-split, soft-Lipschitz and architectural design notes. | Design documents are not counted as completed experiments without a linked result. | [source](../../docs/designs/frequency_split_branch.md) |
| E043 · training and performance audit | Run author-derived SwiGLU and audit the full B-model compute cost. | Four epochs preserved before pause; measured throughput is not convergence or an FFN comparison. | [source](../../docs/lenovo-baseline/README.md) |
| E044 · literature review | Review ode architectures for the ODE architecture question. | Source reading and design motivation only; not a trained-model result or a newly verified literature review. | [source](../../docs/literature-ode-architectures-2026-10-25/README.md) |
| E045 · literature review | Review ode backbone for the ODE architecture question. | Source reading and design motivation only; not a trained-model result or a newly verified literature review. | [source](../../docs/literature-ode-backbone-2026-10-26/review.md) |
| E046 · literature review | Review ode design for the ODE architecture question. | Source reading and design motivation only; not a trained-model result or a newly verified literature review. | [source](../../docs/literature-ode-design-2026-10-26/review.md) |
| E047 · literature review | Review ode intuition for the ODE architecture question. | Source reading and design motivation only; not a trained-model result or a newly verified literature review. | [source](../../docs/literature-ode-intuition-2026-10-26/review.md) |
| E048 · literature review | Review ode model for the ODE architecture question. | Source reading and design motivation only; not a trained-model result or a newly verified literature review. | [source](../../docs/literature-ode-model-2026-10-26/review.md) |
| E049 · literature review | Review ode practical for the ODE architecture question. | Source reading and design motivation only; not a trained-model result or a newly verified literature review. | [source](../../docs/literature-ode-practical-2026-10-26/review.md) |
| E050 · literature review | Review ode synthesis for the ODE architecture question. | Source reading and design motivation only; not a trained-model result or a newly verified literature review. | [source](../../docs/literature-ode-synthesis-2026-10-26/review.md) |
| E056 · research review | Reassess original-layer mathematics, constraints and experimental evidence. | Derivations and literature synthesis; no new training result. | [source](../../docs/research-2026-10-21-reassessment/README.md) |
| E057 · research review | Audit clock coordinates, objective parameterization and next hypotheses. | Working mechanisms are distinguished from performance evidence; no new training result. | [source](../../docs/research-2026-10-23-direction/README.md) |
| E105 · research review | Reassess why local-flow designs have not established a useful successor. | Synthesis of prior results and conditional research questions, not another experiment. | [source](../../docs/research-next-design-review-2026-10-28/briefing.md) |
| E114 · evidence audit | Recover actual synthetic model identities and replay checkpoint predictions. | The fixed-budget ISRU field gain is reproducible; original-ODE, convergence and image-transfer claims remain unsupported. | [source](../../docs/synthetic-audit-2026-11-01/EXPERIMENT_AUDIT.md) |
| E115 · theory fixtures | Relate Gaussian denoising/flow equations to the original polynomial layer. | A fixed-time physical-coordinate correspondence does not identify an arbitrary hidden P with the target dynamics. | [source](../../docs/theory-gaussian-ode-layer/README.md) |
| E116 · implementation verification | Replace the bespoke active trainer with pinned author JiT and an FFN hook. | Compatibility checks establish copied behavior, not a validated CIFAR training recipe. | [source](../../docs/upstream-baseline/validation.md) |
| E127 · legacy configuration inventory | Preserve the early ImageNet JiT configs: base, beta95, beta99, batch256, EMA9996 and full projection. | Six archived configurations exist. This review does not recover a validated completed-run/result binding for them, so they are not claimed as six completed experiments. | [source](https://github.com/Joni-z/ODE-FFN/tree/research-archive-20261101/archive/imagenet/configs) |

## What is not claimed

The ledger does not turn a configuration into a completed run, an audit PASS into
model effectiveness, or a fixed-budget failure into an approximation theorem.
The six early ImageNet configurations have no newly recovered completed-run
binding here. Existing protocols/results are preserved; primary-source hashes
are recorded. Selected main-report values are checked against their tables in
this reporting pass. The November1 synthetic checkpoint replay is earlier,
separate evidence. Other entries index the preserved analyses without claiming
that every experiment was independently rerun for this report.

## Decision

Keep the research question centered on whether a specific ODE constraint earns
its cost against an ordinary realization. Input clocks can help mode separation,
complete adjoints can improve the original predictor, and conditional operator
directions can help thin data. None alone establishes a general advantage of the
original Taylor coupling. Future hypotheses require a prospective controlled
comparison, not reinterpretation of an old failed criterion.
