# K2 generator-gradient route cancellation

Registered before new trained-P gradient-route measurements. This is a bounded frozen-checkpoint diagnosis, not a new training recipe or a reparameterized model. The earlier scalar stationarity/Sylvester derivation is acknowledged; the new question is whether its cancellation mechanism is visible in the current trained original K2 network's nonlinear-error gradients.

## Model, data and exact derivative

Use only the final original scalar K2 direct predictor from conservative_readout_20261026/output, with its recorded buffers and immutable source. Reproduce all five old float32 panels. Use every 4096 A population row at each of times .1,.3,.5,.7,.9 from coordinate_dependence_20261026/output-v2, together with its exact negative, the odd analytic oracle and the completed parity study's fixed raw-linear coefficients. The current strong SwiGLU field results remain context; this study cannot nominate an ODE architecture or claim a generic optimization advantage without such controls.

For one solve, x is its actual input, a is its actual once-computed scalar gate, P is its own matrix, and u is the full downstream output adjoint. Its P derivative is independent of the controller derivative with respect to x: P does not occur in that same controller, and all later recomputation remains in u.

```text
y = x + a*P*x + a²*P²*x/2
g1 = a * outer(u,x)
g2 = a²/2 * [outer(P^T*u,x) + outer(u,P*x)]
gP = g1 + g2.
```

Sum over every actual input and all five times; native complete-network autograd must agree with the sum. The two components are derivative routes of the current K2 computation. g1 is not the gradient of a separately evaluated or trained K1 model.

## Objectives and units

Measure two fixed functionals:

1. T: the mean paired squared field error over both signs, all outputs and all five times.
2. N: the parity study's nonlinear odd error, after raw-linear projection without an intercept. At each time let e=O-v-X*B_error using the saved original-panel coefficient. The envelope/orthogonality identity gives output adjoints e/(n*d*5) on f(x), and -e/(n*d*5) on f(-x). This is the exact gradient at the frozen predictor; no independent nonlinear basis or teacher model is fitted.

Record every solve/time/objective combination, its native gradient and reconstructed g1/g2, plus aggregation over time for all six solves. Do not choose a time, P, checkpoint or objective after inspecting results. The primary question concerns N across all six solves; T is a separate control, not an alternate selection endpoint.

For each row compute the squared Frobenius norms and cross product of g1 and g2, then sum energies before taking a ratio. Also form the coherent matrix sums before taking the corresponding ratio:

```text
local_ratio = sum_i ||g1_i+g2_i||² / sum_i (||g1_i||²+||g2_i||²)
coherent_ratio = ||sum_i(g1_i+g2_i)||² / (||sum_i g1_i||²+||sum_i g2_i||²).
```

Both ratios lie in [0,2] when defined. Zero denominator is reported as no response, not cancellation. A ratio at most .10 denotes at least90% energy cancellation relative to the sum of the two route energies. This is a descriptive severity threshold, not a significance test or optimizer-convergence theorem. Report counts and every value separately for local and coherent measurements. If no primary solve crosses .10, the severe measured-route account is not supported. A crossing is a reason for further diagnosis, not permission to train a new parameterization. No inference of well-conditioning follows from a large ratio in the observed gradient direction.

## Spectral context with explicit limits

At each solve/time record gate minimum/mean/maximum and 5th/50th/95th percentiles. For each of those three durations, record the minimum and maximum absolute factors 1+a*(lambda_i+lambda_j)/2 from all P eigenvalue pairs. These characterize eigenvalues of the normalized Sylvester differential; they are not its singular values for nonnormal P.

Also record lambda_min((P+P^T)/2), ||P||2 and the sufficient lower response bound max(0,1+a*lambda_min(sym(P))). A positive bound certifies that normalized differential's minimum singular value is at least that value. A zero bound is inconclusive. Raw eigenvalues, singular values and P hashes remain archived. These descriptive quantities cannot replace the actual gradient-route measurement or complete-network sensitivity.

## Preflight, independent reconstruction and resources

Before measurement verify the exact P derivative, complete local input/controller derivatives, a nonzero stationarity witness P=-I/a, linear/quadratic route gradients against independent autograd, and per-row norm identities against explicit outer products. Include a constructive case so the witness does not imply universal cancellation. Actual positive-hook extraction must retain old native predictions.

Freeze the implementation and protocol before new outcomes. Main extraction must compare analytic matrix sums with complete-network autograd for every panel/objective. An independent replay explicitly forms per-row outer-product matrices and computes their norms/cross products; it must reproduce every aggregate and decision, while native outputs and the original parity fields agree. Check checkpoints, runtime buffers, sources, stored fields/projection coefficients, and all input hashes before and after.

Run one Torch/BLAS CPU thread on Mac mini with actual process/time/memory telemetry. No optimizer, numerical learning-rate search, new checkpoint, finite trained update, fresh holdout or GPU reservation is part of this diagnosis. Lenovo SSH currently times out. Preserve all preceding evidence and Report1, including failed runtime attempts; do not restart on an observation timeout. No commit or push.
