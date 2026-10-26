# Figures and interpretation

- rate-order-endpoints: primary oracle MSE divided by the better SwiGLU on
  each task. The nonlinear-task failures decide to stop the candidate.
- rate-order-learning: all retained500-step oracle readouts, showing that
  the fixed endpoint follows the recorded fitting trajectory. This does not
  predict the result of an unrun longer experiment.
- rate-order-operators: mean P spectral norm and actual-anchor rate
  commutator magnitudes over the six internal layers. These are descriptive
  layer averages, not independent runs or a complete-Jacobian certificate.
  Smaller P alone does not track improvement across the two populations.

All plots are provided as PDF and PNG. They use native-audited raw JSON/JSONL;
plot-provenance.json records their input digests and plotting-source digest.
Numerical endpoints and the registered stop decision are retained separately.
