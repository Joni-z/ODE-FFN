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
- [ ] Commit source and obtain fresh documented data-free eligibility pass.
- [ ] Implement source-bound fit/sampler/audit/metric/controller entrypoints;
  validate full documented preflight and finite resource estimate.
- [ ] Fit exactly the two registered references, audit all states/samples/
  features/FIDs, then curate, interpret, seal and publish.

## Current status
Only data-free eligibility is selected now. No formal real-data fit or new
quality result is available. Warm metrics spec dc729a50 is unchanged; CPU/MPS
capability and 4.63GiB available RAM/54.81GiB disk were checked. Keep every old
image/synthetic result and Report1 unchanged. No neural checkpoint is edited.
