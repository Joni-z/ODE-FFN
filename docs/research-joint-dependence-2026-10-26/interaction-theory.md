# Matched linearizations can hide different nonlinear interactions

This analysis was added after the primary training outcomes were inspected. It explains an exact property of the constructed target laws and checks narrowly defined feature maps. It does not change the registered decision or prove a capacity bound for the complete trained networks.

## The first difference is a mixed cubic response

Both component-mean ensembles are centrally symmetric, have covariance `Sigma=diag(mu^2)`, and place every mean on the same sphere. Consequently their posterior at latent state q=0 is uniform at every fixed FM time. With `kappa=t/d`, their posterior mean has expansion

```text
m(q) = kappa*Sigma*q
       + (kappa^3/6) E[M*(q dot M)^3]
       - (kappa^3/2) (Sigma*q)*(q^T Sigma*q) + O(||q||^5).
v(q,t) = alpha*q + beta*m(q).
```

The quadratic term vanishes by symmetry. Thus both target fields have the same value, Jacobian and second derivative at the origin. In particular, their entire origin-Jacobian spectrum is identical. The third derivative is `beta*kappa^3` times the fourth cumulant tensor of M. Gaussian component noise contributes no fourth cumulant.

For independent signs the fourth cumulant contains only entries `K_iiii=-2 mu_i^4`. For the Hadamard sign code, those same diagonal entries remain, while an additional entry with four distinct indices is `mu_i mu_j mu_k mu_l` when `i XOR j XOR k XOR l=0`. Other mixed entries vanish. There are140 such unordered quadruples. For example,

```text
d^3 v_0 / (dq_1 dq_2 dq_3) at q=0
    = beta*kappa^3*mu_0*mu_1*mu_2*mu_3     # sign code
    = 0                                    # independent signs.
```

At t=.3, the nonzero value is0.1630638724467017. A separate native Torch differentiation verifies it. This describes local behavior at the origin, which need not be a typical data point. The separately audited data-panel Jacobians show that the orientation difference also occurs on actual interpolated samples, especially around t=.3.

## An exact tensor-rank contrast

Flatten the third derivative of the velocity as a matrix with row index `(output i, input j)` and column index `(input k, input l)`. Positive mu factors and the nonzero time scalar do not change its rank for0<t<1. For independent signs the rank is16. For the sign code it is136.

To obtain the second result exactly, remove the positive mu factors and group ordered index pairs by `i XOR j`. Entries between different groups vanish. Group0 is `-2 I_16`, of rank16. Each of the15 nonzero groups has matrix `ones(16,16) - block_diag(ones(2,2),...,ones(2,2))`, up to pair ordering. Its eigenvalues are14 once, -2 seven times and0 eight times, giving rank8. The total is16+15*8=136. Integer finite-field elimination independently verifies the rank lower bound; the displayed block structure supplies the matching rational upper bound. Orthogonal rotation preserves this matricization rank.

For a single ordinary elementwise FFN `W2 phi(W1 q+b1)+b2` with hidden width h, its third-derivative matrix is a sum of h rank-one terms, so its rank is at most h at every differentiable evaluation point. A single width64 ISRU FFN therefore cannot exactly match the sign-code target's third derivative at the origin when used directly as this16D predictor.

For one SwiGLU unit, differentiating `SiLU(a dot q+a0)*(b dot q+b0)` gives at most two rank-one terms in this matricization after the output projection. A width-h SwiGLU FFN has rank at most2h. The native actual widths42/85/106 therefore give upper bounds84/170/212. Exceeding136 is a necessary possibility condition, not proof that a particular SwiGLU realizes the target tensor or learns it efficiently. Small independent fixtures verify the h and2h bounds numerically.

The measured predictors contain three blocks, nonlinear normalization, conditioning, input/output maps and residual paths. The single-FFN bound does **not** apply to those complete networks. Their empirical ordering is consistent with studying mixed interactions, but it is not a causal validation of this local rank explanation.

## A scalar clock can already generate high cubic rank

It would be incorrect to conclude that the professor's scalar clock is necessarily low rank at cubic order. Take its original K2 ODELayer with P=I. Give its controller paired coordinate rows `+e_j,-e_j`, zero biases and output weights1. Since `SiLU(u)+SiLU(-u)=u*tanh(u/2)`, the original duration becomes

```text
a(h) = .6 + .01*||h||^2 + O(||h||^4).
L(h) = (1+a(h)+a(h)^2/2)*h
     = 1.78*h + .016*||h||^2*h + O(||h||^5).
```

The cubic derivative is `.032*(delta_ij delta_kl + delta_ik delta_jl + delta_il delta_jk)`. Its matricization maps a symmetric matrix A to `.032*(tr(A)I+2A)` and annihilates antisymmetric matrices. Its rank is `dimension*(dimension+1)/2`, hence136 in16D. The native four-dimensional fixture has rank10 and agrees with the formula; the existing64-unit controller has enough rows for the16D construction. This is an existence example, not a trained initialization or a rescue experiment.

Therefore neither a common origin spectrum nor scalar-clock count establishes the useful nonlinear function class or explains the original ODE's measured deficit. A next design must address actual interaction structure and learning, with complete-model evidence. No new ODE method is selected by this local theorem alone.

Verification: [check_interaction_jet.py](check_interaction_jet.py), with immutable outputs in the external run's interaction/ directory. All formal training remains unchanged.
