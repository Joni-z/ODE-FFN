# Cycle02: dependence with fixed component locations and count

Registered before training. This addresses a confound in the prior Product16/sign-code transfer study; it is not a new architecture. That study changed the Gaussian component count from65536 to32 as well as their joint sign law. No conclusion from this control alone will nominate an ODE successor.

## One fixed target law

Use the same16 amplitudes, physical rotation and isotropic component variance0.04. Let p0 be the original independent-sign Product16 law and p1 the32-component symmetric Hadamard sign-code law. Train only on the one new law `p_half=.5*p0+.5*p1`. No mixture-weight sweep is allowed. Every one of the original65536 Gaussian components has positive weight;32 distinguished components have an additional probability mass. Their locations and variances are unchanged.

The new law retains both endpoint laws' mean, covariance, latent sign marginals through order3, component mean norm and full radial law, including at every independent-Gaussian FM interpolation time. Its probabilities and component-label entropy change; keeping the component count fixed does not control effective diversity or label uncertainty. All Gaussian targets have full Euclidean support. Here “full support” refers specifically to positive mass on every original mixture component, not previously impossible Euclidean samples.

The analytic oracle uses Bayes mixture responsibilities. With conditional endpoint means m0,m1, covariances C0,C1 and posterior probability w of the sign-code branch,

```text
m = (1-w)*m0 + w*m1
C = (1-w)*C0 + w*C1 + w*(1-w)*(m1-m0)*(m1-m0)^T
v = alpha(t)*q + beta(t)*m
J = alpha(t)*I + t*(1-t)/d(t)^2*C
d(t)=(1-t)^2+.04*t^2
```

The two branch likelihoods share their Gaussian factor because every component mean has the same norm. Their partition functions are `prod_i cosh(kappa*q_i*mu_i)` and the mean of32 exponentials, with kappa=t/d. Evaluate all normalizers in log space. Verify against independent enumeration of all65536 components and an AD derivative before training. No branch/component labels enter training.

## Models, data and budget

Train the five unchanged models: original ODE-K2, ordinary free ISRU, SwiGLU nominal64/128/160. Copy the exact historical cold states, surrounding parameters and nonpersistent buffers. Keep model seed0;5000 updates, batch512, AdamW3e-4, weight decay0, global clip1. Use100000 target training points and20000 validation points generated with data initialization0, independent standard-Gaussian source and uniform-time pair-velocity supervision. Preserve stream identifier100000 and existing evaluation streams7761,14427,4427. These stream identifiers are not additional model seeds.

Run at most two one-thread CPU workers on Mac; monitor memory/CPU every5seconds. Maximum formal cell wall time600seconds. Save cold/1000/2500/5000 checkpoints with optimizer, stream states and actual runtime buffers, plus all P singular/eigenvalue trajectories and visit statistics. Stop failed cells without changing the budget or settings. Do not reserve a GPU for these tiny models.

Evaluate all20000 fixed validation pairs, five512-row time panels and full network Jacobians,2048 common-source generated points with Heun32 and128 slicing directions. Use existing native metric/sampling implementations. Keep all ten endpoint-law outcomes as explicitly reused historical references, not newly trained replications. No held-out test is opened.

## Interpretation registered before outcomes

The question is whether the observed ISRU reversal persists when all65536 component locations retain positive probability. If its mixed-law field error is more than5% above the best SwiGLU or its Jacobian error more than10% above the best SwiGLU, the literal component-count change is insufficient to explain the reversal. That is not proof that coordinate coupling alone caused it: probabilities, posterior uncertainty and effective diversity still change.

If ISRU instead lowers field and Jacobian errors by at least10% against every SwiGLU while keeping SW2 within10% of the best SwiGLU, its old saturation advantage survives a nonzero dependent-sign contribution. This would reject a claim that nonzero coupled fourth-order interactions always eliminate that advantage. It would not explain the pure sign-code endpoint by itself. Other outcomes are mixed/inconclusive. Original ODE results and all ordinary controls remain visible regardless of these criteria.

Do not select weights, new activations, widths, epochs or solver variants to rescue the result. Any new architecture needs its own derivation and registered experiment. Audit all snapshots, oracle banks, consumed streams, generated metrics, P spectra and immutable hashes before drawing conclusions.
