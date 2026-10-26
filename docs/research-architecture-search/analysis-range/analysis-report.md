# Clock transfer and response geometry

Question: with controller tensors, K1 composition, optimization and linear MACs paired, does relaxing only the final duration range close the nonlinear-task gap to SwiGLU?

The answer is negative at5000 updates. Exact endpoint values and the stopping decision are in [the readout](../clock-range-readout.md). Neither relaxed arm beats either SwiGLU budget on Ring8 or product16. The positive Gaussian comparison therefore does not establish the required frontier. This stops the registered range branch.

The supporting response-space figure answers a different question at a frozen20000-update scalar checkpoint. Allowing any rank-six correction gives a7–12% floor relative to the actual Jacobian error; restricting the correction to the measured clock-response directions gives51–85%. These are conditional least-squares relaxations, not two measured training interventions. Whole-model retraining can move both the base and its response space.

## Claim candidates

- Claim: relaxing the original duration range did not rescue the tested model.
  - Evidence: both Stage8 frozen-source audits and all15 endpoint rows.
  - Allowed wording: wider positive and affine vector clocks fail the registered5000-update screen.
  - Forbidden stronger wording: saturation never matters, or range changes cannot help after longer training.
  - Uncertainty: this screen does not directly test the maturity of the20000-update saturation finding.
  - Next check: stop these branches; require a different operator hypothesis before further training.
  - Decision: keep.
- Claim: clock count alone is a loose frozen-model error explanation.
  - Evidence: exact response projections and the conditional rank/response floors.
  - Allowed wording: the trained scalar response orientation imposes a much larger conditional floor than the rank-six relaxation at these anchors.
  - Forbidden stronger wording: P alone causes the quality gap, or the whole scalar architecture has this irreducible error.
  - Uncertainty: changing weights or clock values changes the response frame; the vector response can already span the output.
  - Next check: derive a precisely defined operator geometry and its unconstrained rival.
  - Decision: keep with the stated conditioning.
