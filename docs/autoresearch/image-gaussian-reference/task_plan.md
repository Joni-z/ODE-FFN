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
- [ ] Validate full documented preflight and finite resource estimate with a
  NEW fresh agent before fitting any actual statistics.
- [ ] Fit exactly the two registered references, audit all states/samples/
  features/FIDs, then curate, interpret, seal and publish.

## Current status
Only data-free eligibility is selected now. No formal real-data fit or new
quality result is available. Warm metrics spec dc729a50 is unchanged; CPU/MPS
capability and 4.63GiB available RAM/54.81GiB disk were checked. Keep every old
image/synthetic result and Report1 unchanged. No neural checkpoint is edited.

Primitive source68a5402/doc invocation7b2db9terminate0. All21state/time cases pass, maximum1.78e-15; formal image fitting remains unstarted. Source-bound main entrypoints now exist; full runtime/source registration and new documented preflight are next.
