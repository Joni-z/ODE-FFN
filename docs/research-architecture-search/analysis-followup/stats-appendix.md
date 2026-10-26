# Comparison units and calculations

This analysis reports deterministic descriptive contrasts under the fixed training protocol (seed0). It does not assign independent-replicate status to checkpoints, projection directions or correlated time panels.

Each model has512 common source/target pairs evaluated at five fixed times (.1,.3,.5,.7,.9). The same pair is reused across times and architectures. Field MSE averages squared error across points, times and coordinates. Jacobian error averages the squared Frobenius norm across points and times, without dividing by the number of matrix entries. Error units therefore differ between these two metrics and between dimensionalities.

Sampling uses4096 identical source vectors per model and a common4096 target reference, with128 directions. Heun32 and Heun128 are solver checks on the same models and sources. The target-versus-target comparison uses two disjoint4096-target subsets. It is a reference for finite empirical discrepancy, not a lower bound or a calibrated acceptance threshold.

Relative contrasts are100*(candidate/reference-1). No p-values, confidence intervals or standard deviations over model training are estimated from these trajectories. Repeated times and directions would not provide the needed independent training units. No assertion about instability is inferred from the fixed protocol.

For each point with symmetric target J*, define S=(J+J^T)/2 and A=(J-J^T)/2. Orthogonality under the Frobenius inner product gives ||J-J*||_F^2=||S-J*||_F^2+||A||_F^2. geometry-breakdown.json verifies this identity numerically and records means, medians,95th percentiles and the maximum-error case for each model. A95th percentile of pointwise error is not a confidence interval.

Validation drove selection of the vectorK1 lead; the two original test splits were opened only after its architecture and20000-step endpoint were fixed. They are now exposed. Subsequent architecture confirmation requires an independent newly generated evaluation bank whose sampling procedure is fixed before readout.
