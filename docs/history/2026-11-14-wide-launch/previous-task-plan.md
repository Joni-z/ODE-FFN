# Current task: audit the next ODE-FFN decision

Request: audit what to do after the completed CelebA128 comparison. No new
training is authorized by this audit. Keep finalized reports and frozen runs.

- [x] Obtain fresh integrity review of source, checkpoint, metric and data bindings.
- [x] Trace the image candidate back to synthetic settings; check order, gate,
  optimizer, normalization, width, initialization and parameter/compute fairness.
- [x] Quantify convergence and per-term cost from existing records; derive the
  minimum diagnostics and experimental controls needed to identify the failure.
- [x] Obtain an independent evidence-to-decision review; write a ranked,
  bounded plan with outcomes that determine continuation or rejection.
- [x] Publish audit/evidence/decision to GitHub and sync active Torch.

Existing results: docs/celeba128-20261112/results.md.
Output: docs/celeba128-20261112/next-step-audit.md plus reviewer receipts.

Decision: recommend frozen K8 diagnostics, then proposed K1/narrow-SwiGLU
controls. No experiment submitted. Eight documentation/evidence files changed;
model/training/metric source and finalized reports preserved. Verification:
81 local links, 15 audited input hashes and 11 Report2 hashes passed.
