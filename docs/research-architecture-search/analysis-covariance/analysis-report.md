# Covariance-field screen

The primary question is whether a covariance-shaped internal generator improves the quality/compute frontier, with exactly initialized untied and static-tangent controls. The full endpoint table and interpretation are in [covariance-readout.md](../covariance-readout.md).

The nonlinear arms fail the frozen continuation rule. Untying improves Ring8 versus tying, so the observations contradict crediting the gradient/covariance constraint for the nonlinear arm's gain over its static tangent. Both practical SwiGLU references remain better on both nonlinear tasks. The internal energy-descent diagnostic is satisfied but does not establish an FM benefit.

## Claim candidates

- Claim: tied hidden covariance geometry is not a useful lead under this screen.
  - Evidence: all18 endpoints, both remote audits, recorded initial-function pairing and budgets.
  - Allowed wording: neither nonlinear arm passes the registered5000-update baseline-sensitive rule.
  - Forbidden stronger wording: no covariance-informed architecture can help FM.
  - Uncertainty: only this fixed hidden parameterization, temperature and budget were tested.
  - Next check: stop the family and identify fitting/noise restrictions before another design.
  - Decision: keep.
- Claim: the intended internal energy property holds on the recorded probes despite poor task quality.
  - Evidence: the derived inequality, its numerical identity/failure-boundary tests, and all36 final tied/tangent layer records in energy-summary.json.
  - Allowed wording: sufficient margins are positive and measured E changes are negative at these probes.
  - Forbidden stronger wording: the complete model is globally stable, optimizes one common energy, or therefore generates better samples.
  - Uncertainty: E is unbounded below in some allowed cases and changes with learned weights/layer.
  - Next check: retain the distinction between an internal certificate and objective fit.
  - Decision: keep with scope.
