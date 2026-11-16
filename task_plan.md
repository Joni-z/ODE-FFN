# ODE contribution investigation

Goal remains active: determine what ODE structure contributes to flow-matching
models through mathematical predictions and falsifiable controls. Preserve
final Reports1/2 and native image code; training seed 0 only. Publish verified
work promptly to GitHub main as Zhizhe Zhang. Mac mini freely usable;145 physical
0-3 prohibited,4-7 only when<500 MiB and no compute PID. Never stop/share a job.

## Completed

- [x] Fixed/free coefficients, state feedback, sharing/coordinate/order controls,
  symmetry/density and qualified error-transport diagnostics; retain all failures.
- [x] Complete equal-dimension null rejects special tangential sensitivity.
- [x] Parameter-free modified-equation prediction explains trained hidden bias.
- [x] Matched midpoint/RK4 training and full common-bank evaluation: hidden
  fidelity improves strongly without native generation gain.
- [x] Full 4,096-input crossed-field diagnosis, all six local solves and two
  complete continuous references; independent every-point checks and statistics.
- [x] Eight crossed-sampling evaluations, both fixed splits and strong ordinary
  controls. All original samples retained. Full matched-split verification and
  independent exact paired statistics complete; four new scientific figures inspected.

## P0 next: predict the target effect of the hidden ODE bias

1. [ ] Implement the exact tangent of the native Heun sampler around the ACTUAL
   learned field, forcing it with the same-weight crossed-solver perturbation.
   Kernel/JVP/endpoint replay checks first; no new training or response search.
2. [ ] Differentiate the SAME empirical squared SW2 functional via its sorted
   projected endpoints. Track ties and the right directional derivative.
3. [ ] Bind fresh common evaluation banks before new outcomes. Primary case:
   Product16 RK4-trained model's observed midpoint benefit; retain other cases.
   Calibrate lambda=.125/.250/.500 and assess lambda=1 separately, with no fitted gain.
4. [ ] If that transport stage qualifies, derive and independently verify the
   RK4 fifth-degree local defect, then propagate the ODE-predicted hidden bias
   through readout and sampling. Do not silently ignore the RK4 two-step defect.
5. [ ] Decide from complete evidence whether a useful ODE-specific design follows.
   Ordinary controls remain required. Do not promote a within-family gain alone.

Canonical derivation: docs/research-assessments/2026-11-16-ode-mechanism/target-aware-bias-transport.md.

## Current turn and retained errors

This goal turn is progress: completed the full field and crossed-sampling
experiments, repaired a verification input-binding bug without rerunning samples,
and independently checked the complete cohort. Both new mechanisms are informative:
coarse numerical bias can help, but average field-risk direction is insufficient.
The target-aware transport link and architectural novelty remain unresolved.
No study worker is currently live. Source freezes and exact receipts remain
outside Git; the preceding detailed journal is archived under
docs/history/2026-11-17-after-crossed-solver/.

The original generation queue 32346 exited 1after all eight native evaluations
exited 0 because confirmation predictions were compared to selection-only FP64
inputs. Its terminal/logs and failed output are preserved. New references bind
each actual split/checkpoint/solver/input hash; the original 3e-5 precision gate
and all 4,096 inputs remain. A wrong-split negative test rejects mismatched bindings.
Fresh cached validation 15857 and all 16 children exit 0. No sample was replaced.
