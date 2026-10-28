# Gaussian reference implementation plan

## Goal
Measure a fixed train-only linear Gaussian reference under the unchanged image
screen, as context for selecting a genuinely useful ODELayer computation.
This control alone cannot complete the broader research objective.

## Steps
- [x] Review primary literature and prior stop scopes after the published
  cold-control study. Nominate no new neural architecture.
- [x] Fix the question, two reference models, rank-budget rule, predictions,
  data/sampler/metric identity, interpretations and stop rules before fitting.
- [x] Derive conditioning, state/time derivatives, exact-flow versus capped
  sampling distinction; implement Gaussian/MLE primitives.
- [x] Commit primitive source and obtain fresh documented data-free eligibility
  pass; preserve its one minor environment-label mismatch and passed source.
- [x] Implement source-bound fit/sampler/audit/metric/controller entrypoints.
- [x] Validate full documented preflight and finite resource estimate with a
  NEW fresh agent before fitting any actual statistics.
- [x] Fit exactly two registered references and complete all six audit phases.
- [ ] Curate all34endpoints, interpret all fixed cases, verify/report/archive
  and publish. No rank, noise or mixture rescue.

## Current status
Only data-free eligibility is selected now. No formal real-data fit or new
quality result is available. Warm metrics spec dc729a50 is unchanged; CPU/MPS
capability and 4.63GiB available RAM/54.81GiB disk were checked. Keep every old
image/synthetic result and Report1 unchanged. No neural checkpoint is edited.

Primitive source68a5402/doc invocation7b2db9terminate0. All21state/time cases pass, maximum1.78e-15; formal image fitting remains unstarted. Source-bound main entrypoints now exist; full runtime/source registration and new documented preflight are next.

Full source3d18ae0published; fresh documented preflight43297terminates0 with no mismatch. Actual-shape fake fit4.31s, native sampler/restore bitwise, independent sampler difference0, measured362496MACs, historical32features/FIDs pass. Formal reference controller may start once. The public theory note corrects a stray prose plus sign only; all frozen runtime source/docs remain unchanged.

Controller35918consumedterminal0; all six phases complete. PPCA59FID293.734970268549/isotropic405.298942736832fail both fitting and generative gates. This specified Gaussian reference does not reproduce the neural quality. No new ODE model, neural mechanism or universal Gaussian failure is claimed. The original metadata helper had a final presentation-only Path/dict name collision after successful readiness assertions/writes; the independently guarded controller ran once without source/output changes or restart. Curation code is next.
