# Current project status

## The next test isolates continuous-solver accuracy during training

The midpoint hidden bias is now quantitatively predictable from its learned
field: all 2,880 states qualify, and its parameter-free trained 4-step prediction
fractionB=.831--.971 across six models. Total/normal adjusted intervals resolve
positivity in all cases. This predicts a numerical effect; its task value is
still an open question.
[Derivation](docs/research-assessments/2026-11-16-ode-mechanism/solver-bias-derivation.md)
and [complete results](docs/research-assessments/2026-11-16-ode-mechanism/solver-bias-results/analysis-report.md).

Registered next: two new seed 0,20k-update Ring8/Product16 trajectories. Keep
f=PISRU(y), parameter count, initialization, optimizer and 8 P applications per
local solve; replace midpoint4 with classical RK4 two steps. Author algebra,
gradient/order/continuation/evaluator checks pass. Publication, source freeze
and the fresh documented pass are complete (actual exit 0, no divergences).
[Protocol](docs/research-assessments/2026-11-16-ode-mechanism/rk4-control-protocol.md).
Both formal 20k trajectories and both fixed evaluation splits are terminal:
all six native commands and supervisor session 40324 exit 0. Complete checkpoints,
standard final tests, update spectra and adapter markers are retained. Independent
metric reconstruction and trained hidden-fidelity measurements are next.
[Completion receipts](docs/research-assessments/2026-11-16-ode-mechanism/rk4-training-completion.json).

## Completed controls and their interpretation

All six dimension-matched tangent-minus-null contrasts are negative, including
adjusted intervals. Ring8 normal-specific effects resolve; Product16 does not.
The earlier 63D-versus-1D dominance cannot establish special ODE sensitivity.
[Equal-dimension control](docs/research-assessments/2026-11-16-ode-mechanism/hidden-null-results/analysis-report.md).

The uniform 512/1024 fixed-first-layer sampling extension completes numerical
qualification in 12/12 model/grid comparisons. It reproduces the direction of
refinement-induced distribution changes, but target SW2 changes are small and
mixed. Original256 pilot failures remain unchanged. This does not select an FFN.
[Complete extension](docs/research-assessments/2026-11-16-ode-mechanism/hidden-sampling-extension-results/analysis-report.md).

## Preserved benchmark and execution boundaries

CelebA128 at 160 epochs: native24.304/K1 27.789/K8 28.083 FID50k.
[Matched table](docs/celeba128-wide-20261114/continuation-160/matched-results.md).
Final Reports1/2 and native image code stay unchanged. Current experiment code
is isolated under studies/muon_pd_20261102/.

Mac mini uses the unchanged one-thread-per-workerCPU environment04f11f75.
145 physical 4-7are occupied;0-3prohibited. No job is stopped or shared.
All earlier null/precision/bias native/verifier handles have actual terminal0.
Raw root: /Users/mr.z/ode-workspace-prep/coefficient-20261116/.
