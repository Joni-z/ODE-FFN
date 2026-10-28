# Before training a nonlinear SwiGLU flow

Registered before the bounded fixtures. This is a candidate eligibility review, not a quality experiment, new autoresearch cycle, or architecture nomination. The accountable researcher is Zhizhe Zhang. Existing local public-model code is the only input; no data, checkpoints, test set or personal information is read. No training, transfer, package installation or GPU reservation is authorized by this document.

## Candidate and rival

For column-vector notation, let

```text
a = Wa h + ba; b = Wb h + bb
f(h) = Wo [SiLU(a) * b] + bo
q = f(h)
T2(h) = f(h) + .5 J_f(h) f(h).
```

This returns the second-order approximation to the increment of `y'=f(y)` over internal duration1, starting from h. It is not the complete normalized/FiLM residual field in cycle01, and is not the professor's two separate linear solves. The unchanged outer residual would still act on the returned increment. For duration tau the local increment is `tau*f + tau^2/2*J_f*f`. A fixed tau=1 is not a small-step error guarantee.

The strong ordinary K1 rival is exactly f. The elementary recomputed-gate rival is midpoint `f(h+.5*f(h))`. A wider ordinary SwiGLU with twice the *actual* hidden width also uses twice the native forward dense work. Their functions, unique parameters and optimization geometry differ; no pairing is claimed from a common zero output head. These are comparison requirements, not selected training arms.

## Complete derivatives and bias handling

For any direction u,

```text
J_f u = Wo [SiLU'(a)*(Wa u)*b + SiLU(a)*(Wb u)]
H_f[u,v] = Wo [SiLU''(a)*(Wa u)*(Wa v)*b
                  + SiLU'(a)*((Wa u)*(Wb v)+(Wa v)*(Wb u))]
J_T2 u = J_f u + .5*(H_f[u,f] + J_f*(J_f u)).
```

Directional projections have no bias; f itself retains all three biases. Backpropagation includes the dependence of q on the same parameters. The frozen-gate approximation omits the first term in `J_f u`; it is a different candidate already reviewed and stopped. The script will retain a witness of the omitted gate term rather than assume it is negligible.

## Fixed checks

Run CPU float64 with torch seed0, one thread, dimensions/actual hidden widths (1,1),(4,5),(64,85), both bias choices, and input scales .1,1,4:18 cases. These fixtures are not a benchmark or evidence of on-distribution behavior. Verify analytic JVP, Hessian-vector contraction, full T2 input tangent and every full parameter gradient against torch.func; verify both fixed central-difference steps1e-5 and5e-6 for a joint parameter direction. Fixed limits: analytic/AD normalized max error<=1e-10; both finite-difference relative errors<=1e-5. Compare the native width64/nominal128 constructor against the literal fixture, including complete tensors.

Record actual dense matrix calls in a batch of8 with all parameter and input gradients active, for native, T2, frozen-gate, midpoint and twice-actual-width SwiGLU. Forward counts exclude biases and pointwise operations; backward counts include every executed matrix operation. These counts are not latency, throughput or equal total FLOPs.

For the legal scalar SwiGLU `f(y)=y*SiLU(y)=y^2*sigmoid(y)`, starting from y0>0, separate variables:

```text
T_escape = integral_y0^infinity (1+exp(-y))/y^2 dy
         = 1/y0 + exp(-y0)/y0 - E1(y0).
1/y0 < T_escape < (1+exp(-y0))/y0.
```

Hence y0=2 already escapes before duration1. This refutes a universal global-flow claim for unrestricted SwiGLU fields, not their use as finite feed-forward functions or all possible trained fields. Verify y0=1,2,4 via two quadrature coordinates and the exponential-integral identity, absolute error<=1e-10, retaining the quadrature error estimates. No simulation is needed to assert blow-up.

## Decision boundary

Successful fixtures certify the formulas and cost accounting only. They cannot satisfy the missing task-specific premise, establish novelty, explain historical losses or override the cycle01 prohibition on Taylor-training rescue. Any eventual training proposal must identify a measured need for this self-interaction, compare recomputed-gate ordinary work and strong SwiGLU on both dependence laws, and preserve the earlier stops. No coefficient, scale, gate, norm, order or width sweep follows this review.

Runtime: `/Users/mr.z/ode-workspace-prep/autoresearch/oct28-nonlinear-swiglu-review`. Existing check environment only. Wall-time ceiling120seconds. Preserve the registered source, execution command/exit status, raw JSON and hashes outside Git; curate small numerical records and the eligibility decision. Submitted Report1 and src/synthetic/configs/scripts remain unchanged.
