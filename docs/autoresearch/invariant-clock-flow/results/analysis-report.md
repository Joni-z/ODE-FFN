# Invariant-clock fixed screen

The registered conjunction fails. All comparisons below use the fixed 5,000-update endpoint.

| Law | Rotation MSE | Swi129 MSE | Best practical MSE | Rotation / best | Rotation SW2 | Swi129 SW2 |
|---|---:|---:|---:|---:|---:|---:|
| rotated_product_bimodal_16d | 0.2947854 | 0.07729042 | 0.03715921 (free_isru) | 7.9330 | 0.09790239 | 0.1027162 |
| sign_code | 0.2550427 | 0.04096724 | 0.03979021 (swiglu160) | 6.4097 | 0.190724 | 0.1216025 |
| half_mixture | 0.5100554 | 0.1192287 | 0.1132542 (swiglu160) | 4.5036 | 0.1347579 | 0.0855002 |

All 36 endpoints and all criteria are in the CSV/JSON files; the table does not select checkpoints or omit old strong controls. The operator is an exact amplitude-preserving hidden flow, but that property does not imply a contractive complete derivative or improved FM quality.

## Claim candidates

- Claim: the tested invariant-clock constraint does not meet the registered old-law quality/cost conjunction.
  - Source evidence: summary.json, all 36 audited endpoints, fixed protocol.md.
  - Allowed wording: this particular parameterization and fixed budget do not justify task/image promotion.
  - Forbidden stronger wording: all ODE architectures succeed/fail, exact flows inherently improve generation, or feature conservation guarantees full-network stability.
  - Uncertainty: these designed synthetic laws are distinct tasks, not independent repeated trials from a task population.
  - Next check: stop this setting; use existing failure evidence before choosing another design.
  - Decision: discard

## Verification

Actual controller and both phase exits are zero. The automatic independent audit replays all checkpoint fields, native Jacobian panels and sampled outputs, checks alternative explicit-chain derivatives and raw matrices, and reconstructs the training-stream boundaries. It does not rerun complete optimizer trajectories. No held-out test bank was opened.
