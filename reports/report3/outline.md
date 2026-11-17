# Progress Report 3: What Does the ODE Constraint Contribute?

The complete [report draft](report3.pdf) and [LaTeX source](report3.tex) now
implement this plan. This file preserves the earlier planning outline.

Author: Zhizhe Zhang. Status: working outline, 17 November 2026.
This is preparation for a three-page report, not a finalized report or a new
training authorization. Reports 1 and 2 remain unchanged.

## Project description and main progress

We study whether ODE-inspired feature transformations provide a useful
architectural constraint for flow-matching models. Since Report 2, the main
progress is a direct test of that constraint: distinguish feature response,
operator sharing and numerical solution accuracy, and measure their effects
on the learned field and generated distribution.

Working finding: shared nonlinear flows have a Ring8 generation advantage in
the retained comparisons, but greater hidden ODE accuracy does not improve
generation and Product16/image controls remain stronger. An unfitted
ODE-derived correction predicts the whole-field effect of changing the hidden
solver. These are separate structural, numerical and practical results.

## Page 1: answer the professor's questions with direct evidence

| Question / purpose | Observation | Primary source |
|---|---|---|
| Does Muon preserve effective weight rank, beyond spectral norm? | Product16 Free ISRU update stable rank rises 4.567 to 32.776, while weight entropy rank falls 55.547 to 43.789 and condition number rises 109.267 to 690.632. Individual update rank and accumulated weight geometry differ. | [Weight/update rank analysis](../../docs/research-assessments/2026-11-15-professor-execution/spectrum-findings.md) |
| Does Ring8 symmetry explain distribution mismatch and mode imbalance? | A frozen C8 projection improves ISRU-D pooled SW2 0.133 to 0.053 and sector L1 0.055 to 0.006; ordinary models improve too. Between 15k and 20k, asymmetric error rises while total MSE falls. | [Symmetry decomposition, samples and trajectories](../../docs/research-assessments/2026-11-15-professor-execution/findings.md) |
| Is the ODE-derived ISRU response necessary? | Scaled tanh is competitive; the eight-response K1/K8 screen produces no confirmed cross-task winner. | [Matched response comparison](../../docs/research-assessments/2026-11-15-professor-execution/findings.md), [complete response screen](../../docs/research-assessments/2026-11-15-response-screen/current-comparison.md) |

Use one Ring8 case study: the 15k-to-20k checkpoints, symmetric/asymmetric
error, and generated mode populations. Keep extended rank curves, all response
tables and trajectory examples as linked supporting material. C8 uses eight
field calls and is a diagnostic intervention.

The earlier Muon/PD request is also addressed: scalar/DP/PD/sandwich placements,
controller variants and response families were tested. Summarize in one sentence
and link the [completed map](../../docs/autoresearch/professor-20261103/research-map.md)
rather than repeating its full grid.

## Page 2: separate sharing from faithful ODE solution

1. **Sharing control.** Purpose: test whether a stationary shared P contributes
   beyond independent stage operators. Shared/stage-specific pooled SW2 is
   0.091/0.111 on Ring8 and 0.064/0.073 on Product16 at equal matrix work.
   The mean-preserving coordinate control recovers part of the Ring8 gap;
   exact ODE coefficients are not established as its cause.
   [Sharing](../../docs/research-assessments/2026-11-16-ode-mechanism/sharing-results/analysis-report.md),
   [coordinate control](../../docs/research-assessments/2026-11-16-ode-mechanism/coordinate-control/analysis-report.md).
2. **Solver-fidelity control.** Purpose: test the prediction that better hidden
   ODE solves improve generation. At identical parameter count and eight P
   calls per solve, RK4 reduces hidden error approximately 46x/22x without a
   confirmation sampling improvement.
   [Equal-work training](../../docs/research-assessments/2026-11-16-ode-mechanism/rk4-results/analysis-report.md).
3. **Analytic field response.** Purpose: predict how the hidden numerical bias
   reaches the actual network output. The unfitted correction predicts
   whole-field changes with scores 0.957–0.983; both Product16 RK4 streams
   score 0.962. Define the score once and distinguish it from generation.
   [Derivation and completed field test](../../docs/research-assessments/2026-11-16-ode-mechanism/learned-defect-field-results/analysis-report.md).

Use a single diagram: hidden solve -> feature change -> complete field ->
generated distribution. The next fine-mesh analytic target-loss forecast was
canceled before a complete cohort; it is not a completed result. The finished
Heun128 diagnostic may be mentioned separately if space permits.

## Page 3: practical result and the next decision

The previously queued ordinary controls have now completed and passed their
independent comparison check. On the current **common** confirmation banks,
using Heun2048 and native inference for each row:

| Model | Ring8 pooled SW2 | Product16 pooled SW2 |
|---|---:|---:|
| State flow / midpoint training and inference | 0.068 | 0.047 |
| Free ISRU / AdamW | 0.091 | 0.051 |
| SwiGLU64 / Muon on Ring8; SwiGLU192 / AdamW on Product16 | 0.105 | 0.034 |

This compact table is a native-solver slice of the completed comparison, not a
post-outcome choice of best solver. Retain both streams and all four trained/
inference solver combinations in the linked data. Ring8's pooled ordering
favors ODE throughout, while most adjusted paired-bank intervals include zero.
Product16 retains a SwiGLU advantage.
[Full comparison, figures and independent verification](../../docs/research-assessments/2026-11-16-ode-mechanism/ordinary-population-results/analysis-report.md).

State flow uses 227,648/229,440 dense forward MACs versus Free ISRU
55,616/57,408 on Ring8/Product16. Equal updates are not equal computation.
The Ring8 benefit therefore motivates a compute-matched control, rather than
an immediate image transfer.

Image anchor: parameter-matched CelebA128 at epoch160 gives FID50k 24.304
(native), 27.789 (frozen-D K1), 28.083 (frozen-D K8). Make the distinction
between those frozen-diagonal image blocks and the refreshed state flow explicit.
[Completed image comparison](../../docs/celeba128-wide-20261114/continuation-160/matched-results.md).

## Hypothesis for the next two weeks

**Primary hypothesis:** the shared state-flow model's Ring8 generation gain
survives matching cumulative training computation with ordinary Free ISRU and
SwiGLU, rather than only matching update count.

Proposed experiment, not launched: reuse the fixed 20k state-flow trajectory
and continue the two existing ordinary Ring8 controls at seed0, with unchanged
architectures and optimizer recipes. Determine the required continuation
length from measured/explicitly accounted forward, backward and optimizer
work before launch; roughly four times the ordinary update budget is an
initial matrix-work estimate, not a claim about total FLOPs. Bind checkpoints,
common sample banks, generation settings and mode-balance measurements first.

If ordinary models catch up at matched computation, the current advantage is
an allocation-of-compute result. If state flow retains a quality/cost advantage
against both, retain the shared-flow architecture and investigate which target
geometry predicts that benefit. A Ring8 win alone does not justify claiming
general superiority or selecting a new CelebA architecture.

The report can be drafted from the completed evidence now; this future test
belongs at its end and is not a prerequisite for writing the report.
