# Identify the training restriction before another P constraint

Stage8 and Stage9 are stopped. No further architecture or optimizer trick is selected. The next problem is to distinguish representable fitting, conditional target noise and finite target-bank effects. These are different mechanisms.

## The exact conditional-noise identity

For either a population target law or its empirical-bank replacement, let v* denote that law's own conditional mean under X=(1-t)Z+tY with independent standard Gaussian Z. For0<t<1 and a=1-t,

```text
pair target = Y-Z = (Y-X)/a,
Sigma_target(X,t) = Cov(pair target | X,t) = Cov(Y | X,t)/a²,
J_v*(X,t) = -I/a + t*Cov(Y | X,t)/a³,
Sigma_target(X,t) = [I+a*J_v*(X,t)]/t.
```

The same posterior covariance determines both the ideal velocity curvature and conditional label noise. At a fixed parameter vector and fixed input, write A=partial v_theta/partial theta and r=v_theta-v*. For squared loss without the dimension-normalization constant,

```text
E[g_pair | X,t] = 2 A^T r,
Cov(g_pair | X,t) = 4 A^T Sigma_target A,
trace Cov(g_pair | X,t) = 4 trace(Sigma_target K), K=A A^T.
```

This is an exact conditional-gradient statement. It is not an AdamW convergence theorem. Adam's parameter-dependent preconditioning, correlations across updates, input-sampling variance and the current residual all matter. A parameterization-dependent NTK diagnostic can measure signal/noise at frozen weights; it cannot alone prove a representational lower bound or identify the causal reason training is slower.

## The teacher must match the training law

Training draws Y uniformly from the cached target bank and draws Z fresh. Its unrestricted risk minimizer is the **empirical-bank** posterior mean v_emp, not necessarily the known analytic-population mean v_pop used for scientific evaluation. Replacing pair labels by v_emp is a conditional-mean variance reduction of that training objective. Replacing them by v_pop at the same cached-bank inputs generally changes the minimizer. A successful v_pop-supervised fit would show achievable function fitting on those inputs, not a clean noise-only ablation.

If a later training comparison aims to isolate population conditional noise exactly, both pair and mean-label arms need the same fresh population input law and paired random draws. That changes the data protocol relative to the earlier cached-bank screen and must be declared for all references. Alternatively use the exact empirical teacher, whose posterior must sum over all bank atoms rather than a selected neighbor subset. No teacher-trained network should be promoted as a deployable FM architecture merely because an analytic oracle is available on synthetic data.

## First read-only diagnostic, specified before its outcomes

Use all100000 product16 training target atoms. For the first128 anchors in the already exposed common validation panel at each of t=.1,.3,.5,.7,.9, compute the exact finite-sum posterior, its velocity, covariance, Jacobian, effective sample size and largest weight. Compare these with the saved and previously verified population oracle. Include all atoms; use stable normalized exponentials and double precision, with chunking only over anchors. Record any covariance roundoff explicitly rather than clamping it silently.

The purpose is to decide whether population-versus-bank teacher mismatch must be controlled in the next noise/fitting ablation. These128 repeated validation anchors are not a new confirmation set or independent training replications. A large mismatch would not by itself explain why ODE loses to SwiGLU: both were trained on the same bank, and a neural model can generalize beyond the finite empirical posterior. The routine's analytic derivative is checked against finite differences and its single-atom limiting case before examining the real bank. No new training is launched by this diagnostic.
