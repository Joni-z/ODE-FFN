# Research question

- Gap: cycle08 proves a fixed-readout restriction; cycle12 locates much thick
  sampling error there. Neither proves this error directly corrupts P training.
- Existing evidence: cycle14 thin late windows never clip, thick windows clip
  63.4–93.8%. This is a reason to measure gradient routes, not an explanation.
- Theory: a fixed readout-null cotangent has zero upstream pullback; a global
  norm clip can still couple it to hidden updates through readout gradients.
- Controlled observation: all22clean models, four saved checkpoints, next
  unchanged native batch, no fitted step or stochastic label replacement.
- Deliverable: verified route theorem, audited numerical decomposition and
  explicit limits on what it explains. A useful successor remains unachieved.
- Stop: no head/loss/optimizer/width/order/rank/clock rescue from this diagnostic.
