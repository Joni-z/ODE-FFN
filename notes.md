# 15 November follow-up: completed

145 physicalGPUs 4-7 only. Five registered20k trajectories: Ring8/Product16
Tanh-D plus three Ring8 reference replays. All frozen15k/20k evaluations pass.
Main wave5m41s; an additional2m57s original-entry SwiGLU audit verifies exact
20k model/optimizer/RNG identity. All queues ended without failure.

[Canonical findings](docs/research-assessments/2026-11-15-professor-execution/findings.md).
- Product16: high-rank Muon updates coexist with lower-rank Free ISRU weights.
- Ring8: C8 projection reduces pooled SW2 to0.052-0.053 across all four families;
  oracle same-bank reference0.05528. ISRU-D asymmetric orbit error grows15k->20k
  while symmetric error falls; it accounts for 66.7% of error at 20k.
- Matched scaled tanh has no consistent disadvantage to ISRU across two tasks.
- Ring8 ISRU-D/Free replay nearly match history. SwiGLU/Muon differs from
  Lenovo history; old/new entry agrees exactly on 145, source of cross-host
  discrepancy remains unidentified. Never substitute the newweights for old evidence.

Frozen training source da8b8a4; CPU interpretation14c65a7, separate worktree.
Raw runtime /data2/zz5070/ODE_FFN/professor-followup-20261115.
Local raw analysis mirror /Users/mr.z/ode-workspace-prep/professor-followup-20261115.
Original Product16 data/weights preserved; Ring8 cache matches all registered hashes.
Report1 and every protected Report2 file unchanged.

Next bounded proposal: pairedC8training augmentation as ordinary control, then
an equivariantPD generator versus an equally equivariant ordinary FFN. No new
image run or further order/spectral constraint sweep selected.
