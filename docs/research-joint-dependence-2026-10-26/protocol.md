# A joint-dependence transfer control for the ordinary ISRU observation

Registered before new training outcomes, 26 October2026. The previous cubic-field study is sealed at440371b3b09658da8bebfefcc592b6052dff8a4f332f87ed0c34ad1e7f1f5b7d. Its ODE candidate failed and remains stopped. This study changes the synthetic target law while keeping five already specified architectures unchanged; it does not nominate or rescue an ODE method.

## Question

The ordinary free-ISRU control achieved Product16 field MSE0.037159 versus0.078178 for SwiGLU160, but did not outperform strong SwiGLU on Gaussian2D or Ring8. Does the Product16 observation transfer when independent component signs are replaced by dependent signs, while preserving substantial low-order and radial properties?

## Exact target law

Use the existing Product16 latent amplitudes `mu=geomspace(.75,2.5,16)`, noise variance `.04` and the exact existing orthogonal rotation R. Let H be the deterministic Sylvester16x16 Hadamard matrix, with entries+1/-1 and `H^T H=16I`. Define32 equally weighted latent means as the rows of `[H; -H] diag(mu)`. Add independent isotropic Gaussian noise with standard deviation.2, then apply R. This is a deliberately constructed Gaussian-mixture mechanism control, not a claimed standard public benchmark.

The original target uses all65536 independent sign vectors; this law uses32 dependent sign vectors. Both have zero mean, the same covariance, identical single-coordinate latent marginal laws and the same full radial law. Pairwise and three-coordinate sign marginals also match. For every interpolation time, their radial laws and covariances remain equal under the same independent standard-Gaussian source. Their higher-order joint dependence and number of components differ. These changes can affect approximation, posterior uncertainty and optimization; they are not isolated causal factors.

Generate100000 training targets,20000 validation targets and2048 probe tuples with one deterministic data-generation stream initialized0. Use the existing native fixed-evaluation convention: validation source/time stream7761, validation sampling source/directions14427 and five512-example time panels with source stream4427. These are preserved stream identifiers, not extra model training seeds. No test split is used for model selection. Generate labels only as independent pair velocities during training; component labels and analytic oracles are excluded from training inputs/targets.

## Target-geometry derivation

In latent coordinates q, write `d=(1-t)^2+.04t^2`, `alpha=(.04t-(1-t))/d` and `beta=(1-t)/d`. With M denoting the random component mean,

```text
v(q,t) = alpha*q + beta*E[M | q,t]
J_v(q,t) = alpha*I + t*(1-t)/d^2 * Cov[M | q,t].
```

The posterior weights are a stable softmax of `t*q dot M/d - t^2*||M||^2/(2d)`. All means have equal norm. For original Product16 this factors coordinatewise and every Jacobian is diagonal in the fixed R basis, so Jacobians commute at fixed t. In the sign-code mixture, posterior covariance need not be diagonal. At q=0 it equals diag(mu^2), with distinct eigenvalues; a fixed nonzero q with an off-diagonal posterior covariance gives a direct noncommuting-Jacobian witness.

Record exact Jacobian off-diagonal energy in the known latent frame and normalized pairwise commutators on the five actual-data panels, including the Product16 reference panels. Check the closed form against the existing GaussianMixture velocity and native Torch autodiff. A change in oracle geometry does not prove a limitation of either complete trained architecture: even a free two-matrix elementwise-activation FFN can produce noncommuting Jacobians. This is a property of the target law, not an ODE-specific impossibility theorem.

## Runs and unchanged settings

Five new5000-update fits on the dependent-sign law: original ODE K2, ordinary free ISRU, and SwiGLU64/128/160. The free-ISRU initialization is exactly the previous `H_.6(P1)` initialization and its response remains `z/sqrt(1+1.2z^2)`. Keep native width64, depth3, time conditioning, readout and matched surrounding tensors. Training seed0, batch512, AdamW3e-4/default moments, weight decay0, clip1, uniform time and generator stream100000 are unchanged. There is no new architecture or hyperparameter search.

Explicitly reuse the five audited Product16 trajectories, including their immutable input/checkpoint provenance. Verify all initial parameters/buffers/optimizer states against them and all five new three-update uninterrupted/resumed histories. The reused trajectories are historical references, not new runs or replications.

## Measurements and interpretation

Primary: all20000 validation-input oracle MSE for every architecture. Report the within-law free-ISRU/best-SwiGLU risk ratio and how the ordering changes between laws. Secondary: all five time panels, mean squared full input-Jacobian error, common-source2048-sample32-step Heun SW2 with128 projections, all four snapshots, clipping, P trajectories and actual costs.

A practical ordinary-ISRU transfer pattern requires new-law field risk within1.05 times the best SwiGLU risk, mean derivative error within1.10 and finite sampling SW2 within1.10. This is a descriptive practical criterion. Failure weakens generalization of the Product16 observation; passing supports only this additional constructed law, not a new ODE mechanism or image-generation result. Do not use a small conditional evaluation difference as a claim about training variability.

No outcome selects a cubic-candidate rescue, LR/clip change, further width, time coordinate or new gate. The goal is evidence for the next ODE design question. Every arm and exception remains visible.

## Execution and audit

Freeze source/settings, generated data, oracle banks, protocol and all historical dependencies before one launch. Use two one-thread CPU workers with five-second CPU/RSS monitoring and100-update status. Save actual model/optimizer/RNG/buffers and input-stream hashes at0/1000/2500/5000. Independently reconstruct the complete stream, replay every native checkpoint/panel/sample, verify P spectra and compare the analytic data oracle/Jacobian with independent formulations before interpreting results.
