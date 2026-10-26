# Next read-only question: conditioning of the features seen by P

This is an unselected design question, not another launched architecture. Stage11's paired clock-coordinate change improves product16 by about33.6% at the same initial function and feasible function class, but remains behind both SwiGLU budgets and stops under its registered rule. No temperature sweep or continuation of that scale is planned.

Together with the matched-noise result, this motivates measuring parameter-response geometry before adding another constraint. A singular-value cap on P does not directly correct the conditioning of the features used to train it.

## Exact local gradient, qualified curvature model

At K1 let y=h+D(h)P*h. If q is the loss derivative with respect to y, the controller is independent of P at that layer, so

```text
gradient_P = (D*q) h^T.
```

For the complete velocity network, let W_i be the derivative of its output with respect to P*h at anchor i. The P-parameter response is the Kronecker-structured map h_i^T tensor W_i. Its same-anchor output tangent kernel is ||h_i||² W_i W_i^T. Cross-anchor kernels also depend on h_i^T h_j. A frozen-block Gauss–Newton factorization therefore involves both input second moments and downstream response; input covariance alone is not a full nonlinear convergence theorem.

Because P has no bias, inspect the **uncentered second moment** G=E[h h^T], not only covariance around the mean. Initial output heads are exactly zero, so initial downstream P responses vanish. Measure input moments at initialization, then both moments and downstream responses at1000/final checkpoints before interpreting the optimizer geometry.

The first read-only measurement should use common inputs at each fixed FM time and a separate bank-derived uniform-time calibration stream, separately for every first/second ODE solve and the SwiGLU inputs. Record positive-eigenvalue quantiles, participation, numerical rank and how much of the P-gradient energy lies in low-second-moment directions. Preserve the known middle-SiLU and task-dependent second-gate importance; poor conditioning is not grounds to delete a solve or activation again. Existing archived initial/1000/final checkpoints and runtime buffers support this analysis without new training.

### Structural null directions are not a diagnosis of failed optimization

The first FFN input at a fixed time has an exact linear-span bound in this backbone. The state embedding is A*x+b. Centering in LayerNorm gives A_c*x+b_c, and division by its scalar normalization keeps each sample in span(A_c,b_c). Learned LayerNorm scales and the time-dependent diagonal scale are fixed linear maps at this time; their biases/time shift add one constant vector. Therefore the uncentered first-input moment has rank at most input_dim+2, even at a trained checkpoint. The bound applies to both ODE and SwiGLU and survives nonzero LayerNorm epsilon.

For our width64 backbones this yields at most4 active first-input directions on a fixed-time2D panel and18 on a fixed-time16D panel. Time modulation can move these spans, so the uniform-time pooled moment need not share that bound. Later nonlinear features can also have full linear covariance rank despite low intrinsic input dimension. The diagnostic must report these distinctions and verify the bound numerically; an infinite condition number due only to the predicted null directions is not evidence that whitening would help. Use the positive spectrum and downstream response overlap, and match the actual training-time distribution when considering a fixed coordinate transform.

## A possible full-class control if the measurement warrants it

For an invertible fixed feature matrix C, writing P=R*C and initializing R=P0*C^{-1} preserves the initial map and all feasible P. The local gradient uses C*h, so its second moment becomes C*G*C^T. Taking C=(G+lambda*I)^(-1/2) gives eigenvalues g/(g+lambda) in the transformed input factor. Lambda regularizes the coordinate inverse, not the set of learned P matrices.

This could become a function-preserving **parameterization** of the original layer, with a matched initialization and a generic SwiGLU comparison. The gate's first projection sees the same hidden input and may need the same coordinate treatment if the hypothesis concerns shared features. Whether to transform P alone or the common input projections must be decided from measured responses and registered before outcomes; do not silently combine both. Finite-precision reconstruction, extra matrix-assembly cost, null feature directions and stale frozen statistics are explicit failure concerns.

No such whitening is implemented or selected here. If input moments do not show a relevant bottleneck, discard this proposal. If a later implementation is warranted, it must retain ordinary pair-target training, preserve the original full task/baseline scope, avoid combining the stopped clock scale, and distinguish an optimization effect from a new function class or a hidden-to-data-space geometry claim.
