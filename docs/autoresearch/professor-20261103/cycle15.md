# Cycle15: explain the saved fitting/sampling mismatch

Registered4November2026, before new trajectory/profile/intervention outcomes.
Question: what differs between ISRU-D and ordinary models when field fitting
improves from15k to20k but sampled discrepancy increases? The project tests an
ODE-inspired feature bias in flow matching. Internal feature duration and
global FM time are distinct variables.

## Scope and hypothesis

Use only existing seed0 Ring8 checkpoints: ISRU-D/AdamW3e-4,
Free ISRU/AdamW3e-4, and selected Swi64/Muon.02, at15k/20k. Six frozen cases.
The native model, cached law, loss, checkpoint selection and training are
unchanged. The cohort uses the already saved cycle14 bank0:4,096 common Gaussian
sources, targets and256 projection directions. No new evaluation stream.

H1: the changed error is concentrated by transport time or radial/angular
direction, even as average field MSE improves. H2: amplification of the actual
state discrepancy differs from the injected field/step error. Both ordinary
models remain in every diagnostic; compare within-model15k/20k first.

## Common states and exact discrete identity

Construct an oracle transport path by running the unchanged native Heun sampler
on the analytic population velocity u, starting from the same noise. This is
an ODE marginal-flow path, not the straight interpolation of a sampled pair.
Use128steps for common-state/model comparisons; also save oracle256/512
paths/endpoints and their numerical differences. They measure reference solver
sensitivity. Save every native model128path, with exact endpoint replay against
cycle14 bank0 outputs for the same checkpoint and4096-point batch.

Let Psi_v,i be one native learned-field Heun step, Psi_u,i its oracle version,
z_i the common oracle state and y_i a learned state. With e_i=y_i-z_i,

```
d_i = Psi_v,i(z_i) - z_(i+1)
m_i = Psi_v,i(y_i) - Psi_v,i(z_i)
e_(i+1) = m_i + d_i.
```

This is an exact discrete decomposition. Retain the actual arrays and test the
identity in double arithmetic; it is not a linearization. A helper copies the
native step algebra verbatim and must reproduce a full native rollout exactly.
Measure the secant gain ||m_i||/||e_i|| where ||e_i||>1e-8, retaining counts and
unclipped extrema. It describes the actual error direction at those states.

## Profiles and interventions

At33 common times0,1/32,...,1, save full4096-point model/oracle velocity errors.
Decompose error into radial and angular components relative to the Ring8
origin; report signed means and squared energy. Verify their summed energy is
the vector squared norm (twice coordinate-mean MSE). Record any radius<=1e-8
separately; these are geometric directions, not a continuous-support manifold
assumption. Compute full2x2 velocity Jacobians on the first256 common states,
plus analytic-oracle Jacobians; report norm, symmetric growth and discrepancy.
Native model tensors/buffers stay fixed. Analytic Jacobians require central-
difference validation; model derivatives require nonzero fixtures and checks
that samples do not interact.

At cuts32/64/96, run both learned-prefix/oracle-suffix and oracle-prefix/learned-
suffix trajectories with the same128native physical steps. A policy chooses the
field once per Heun step and uses it for both stages, so switch boundaries use
the appropriate one-sided interval value. All-learned/all-oracle policies must
match native anchors exactly. Save all36 switched endpoints and their SW2,
plus common-source endpoint errors to the oracle path. Oracle replacement is
a diagnostic intervention on the frozen field, not a proposed sampler.

## Analysis and decisions

Always show all six cases,33times,128local-step decompositions, three cuts and
both switch directions. Retain unchanged cycle14 distances as reference.
Compare20k-minus15k within each model for fitting, common-state profiles,
endpoint error and each intervention. Show paired sample-bank evidence already
completed rather than call the repeated states independent training samples.

- A common late-error/switch effect across all three models selects a shared
  field/transport question; it does not select another ODE layer.
- A candidate-specific local-defect or directional-amplification difference
  selects that measured mechanism for a subsequent matched control.
- If oracle-resolution or source-subset behavior dominates the comparison,
  quantify it before choosing an architectural explanation.
- Similar Free-ISRU and ISRU-D behavior directs work toward the shared response;
  a distinct response/propagation pattern motivates an initialization/placement
  control. No new architecture, coefficient sweep or image transfer follows
  automatically from a profile correlation.

q(h)=h/sqrt(1+1.2h^2) is the exact scalar flow of du/dr=-u^3 at internal
r=.6. Current ISRU-D uses F2(SiLU(F1(h))), Fi(h)=h+Pi q(h); copied Free ISRU
uses A q(Bh). Initialization, residual placement and number of nonlinear maps
therefore differ. This exact source audit motivates the next control; matching
parameter counts alone does not isolate these choices.

## Execution and progress

Warm Lenovo synthetic environment, separate source/output folder, original
CUDA device and one-thread processes. Use native monitored queue, two workers,
source/file hashes and fresh documented model/oracle checks before full runs.
All raw arrays remain outside Git. The completed Report2 PDF/TeX stay fixed.

- [x] Implement native trace/step/profile/oracle-switch wrapper and CPU fixtures.
- [x] Publish/deploy source and pass fresh documented original-checkpoint CUDA checks.
- [x] Run/replay the finite cohort and write figures, exact data and next decision.

Completed from frozen source `6fe9e7f` on Lenovo. See the
[execution record](cycle15-execution.md), [analysis](cycle15-results/analysis-report.md),
and [mode-mass derivation](mode-mass-derivation.md). The registrations below
retain the order in which saved-array follow-ups were chosen.

## Post-outcome precision check selected before recomputation

The preselected bank0 reverses the mean-of-eight checkpoint trend: all three
models improve SW2 from15k to20k there. It is therefore a counterexample to
assuming a bank-specific switch explains the average trend. Before more model
calls or architecture, recompute the same six checkpoints with all saved eight
banks concatenated (32,768 samples), plus all leave-one-bank-out pools. Publish
all24within-model bank differences and all54pooled/omission scores. Keep the
original mean-of-banks endpoint and the preselected bank0 diagnostics intact.
No sampling or training is rerun and no bank is selected to restore a result.

Also use the known Ring8 mixture means/weights to report posterior-component
frequencies and within-component residual energy for every saved bank0 switch
endpoint and the pooled native endpoints. Compare frequencies to both the
empirical target and the analytic weights, with assignment confidence. These
are descriptive diagnostics, not new primary performance metrics. This checks
whether a low finite-bank distance reflects target-bank mass matching. The
common-source oracle endpoint MSE measures transport-map discrepancy; it is
not the minimum distributional Wasserstein coupling.

## Follow-up selected after the pooled checkpoint check

Pooled15k/20k SW2 retains a deterioration for ISRU-D and selected Swi64, while
Free ISRU changes little. Component counts move farther from the known equal
weights in all three models as center-residual energy moves closer to the
analytic variance. Before new field/model evaluations, use saved paths to
count angular-sector changes (gross crossings and net mass), and split common-
source endpoint error by agreement with the oracle sector. Validate native
oracle reflection symmetry numerically on sector boundaries. Record the exact
sector-invariance derivation and distinguish necessary mass balance from a
particular particle-to-mode coupling. Also separate component centroid bias
and covariance error from the previously reported residual energy. All are
post-outcome diagnostics on existing arrays; no new primary metric or sampler.

Final saved-array decomposition: evaluate the unchanged finite-projection SW2 formula after mapping all pooled samples to their assigned component means, and separately after subtracting those means. The first depends only on component counts and center geometry; the second probes centered residuals. Retain all six cases and empirical targets; these diagnostic distances are not an additive decomposition or replacement for native SW2. No model calls.
