# Prediction-space bridge: a fixed comparative screen

Registered 2026-10-25, before new training outcomes. This is a transfer/calibration experiment motivated by a concrete protocol gap: previous architecture screens directly predicted velocity, whereas JiT directly predicts clean data. It does not introduce a new ODE architecture or revive a rejected structural modification.

## Question and fixed factors

Does the unchanged scalar ODEFFN become practically competitive with SwiGLU under clean-data prediction, when both architectures receive the same prediction/loss/time protocol?

Cross four backbones with two direct prediction spaces, for eight arms:

- Professor scalar ODE, K=1, gate hidden64, original random controller initialization.
- Professor scalar ODE, K=2, gate hidden64, original random controller initialization.
- SwiGLU nominal hidden128 (approximately parameter-matched).
- SwiGLU nominal hidden192 (approximately K2 matrix-compute-matched).
- Each backbone directly outputs either clean data x or the effective velocity v.

All use width64, depth3, the existing affine time conditioning and output normalization, no change to the original two ODE solves and intervening SiLU, and the native zero raw-output head. Within a backbone, initial raw tensors/buffers are identical across prediction spaces. Across backbones, copy all non-FFN parameters and actual buffers from one seed0 ODE-K2 reference before training. A zero x-prediction head and a zero v-prediction head induce different initial velocity fields; this is explicitly not a same-initial-function optimization experiment.

## Shared training law

Use the three existing immutable cached datasets: Gaussian anisotropic2D, Ring8, and rotated Product16. Keep batch512, AdamW3e-4 with default betas, weight decay0, gradient clip1, seed0 and5000updates. All arms consume exactly the same target-index/source/time stream per dataset. No component labels or analytic teacher enter training.

For this bridge use the JiT paper's time sampler `t=sigmoid(-.8+.8*N(0,1))` and denominator `d=max(1-t,.05)` in **both** prediction arms. These are a shared recipe, not an ODE-only intervention, and comparisons against old uniform-time results are not causal estimates. Other settings remain the existing toy trainer's settings; this is not a full JiT reproduction.

Let z=t*x+(1-t)*e. Both arms regress the same effective pair-velocity target `(x-z)/d`:

```text
x-pred: v_hat = (raw_net(z,t)-z)/d
v-pred: v_hat = raw_net(z,t)
loss = mean((v_hat-(x-z)/d)^2)
```

This handles denominator clipping consistently. The x-prediction loss equals `mean((raw_net-x)^2/d^2)`. The common clipped oracle is `((1-t)/d)*v_star`. A direct velocity arm trained against unscaled x-e in the clipped region would change the target and is excluded.

## Evidence and primary decision

Primary field error uses20000 common validation pairs with independent uniform times in [.05,.95], where no clipping changes the physical teacher. Target rows are sampled with replacement from the immutable validation bank (10000 rows for the2D tasks,20000 for Product16), then paired with fresh independent noise/time. Also report native logit-normal-time oracle error on20000 common validation pairs, and full [.0,1.0] uniform-time error against the clipped oracle. Retain all three; do not select the favorable time measure after observing results.

Save initial,1000,2500,5000 checkpoints including optimizer, actual runtime buffers, training data-generator state and actual input hash chain. Save training losses/gradient norms/clip fractions every500updates and internal P singular spectra at every checkpoint. Save final prediction arrays and five fixed-time512-anchor full-network Jacobians, oracle arrays, generated samples and targets. At t=.1,.3,.5,.7,.9 conversion is unclipped.

Sampling uses the same learned effective field and Heun32,2048 common Gaussian starts and validation targets,128 fixed slice directions. Record SW2 and the target-vs-target reference floor. The clipping bias is shared; physical endpoint correctness is not assumed. No sample-quality gain is inferred from field error alone.

All generic controls in both prediction spaces define the practical frontier. An ODE arm qualifies for continuation only if:

1. Its primary field MSE is at least10% below the best generic value on Ring8 or Product16.
2. Its primary field MSE is no more than5% worse than the best generic value on either remaining task.
3. On both nonlinear tasks, Jacobian error and SW2 are no more than10% worse than the best generic values.
4. All states/metrics are finite, common input streams match, and the model is no more than5% more expensive in dense forward MACs than the winning generic comparator for an equal-cost claim. A higher-cost gain remains a quality/cost observation, not a matched-cost success.

The same ODE arm must satisfy all tasks; do not choose K or prediction space separately by dataset. These are practical engineering thresholds, not statistical significance claims. Parameters, dense MACs and training wall intervals are all retained; concurrent wall intervals are not isolated latency measurements.

## Continuation only after a qualifying screen

Continue each qualified ODE arm and all four generic controls from5000 to20000updates on all three tasks, without new tuning. Then freeze the selection, use previously untouched test targets and fresh fixed source/pair banks,8192-sample Heun32/128 generation, and isolated one-thread throughput measurements. An accepted useful result must retain the field gain, transfer and generation/cost guards on these new checks. No positive screen alone is reported as a validated architecture.

If no ODE arm qualifies, stop this prediction-space setting. Do not launch temperature, schedule, width, endpoint-clamp, optimizer or initialization rescue sweeps. Report the negative result briefly and preserve the arrays.

## Source

[Li and He, JiT, Section4.3 and AppendixA](https://arxiv.org/html/2511.13720): clean-data prediction, velocity-space loss, logit-normal time parameters and denominator clipping. The toy architecture, AdamW settings, dataset construction and acceptance rule are our declared experimental choices, not claims imported from the paper. No JiT code is consulted.
