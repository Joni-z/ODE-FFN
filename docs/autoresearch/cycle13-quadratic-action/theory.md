# Quadratic action, scalar clocks and spectral constraints

All statements in the first sections are our elementary derivations. They apply to a single frozen conditional solve; the whole-model qualification is explicit below.

## Exact fixed-direction envelopes

For a column input z in the layer's conditional frame, write

```text
u=Pz, w=P^2 z, d=s*u+.5*s^2*w.
```

The K2 output is z+d. For any nonzero direction v, the scalar coefficient minimizing `||d-r*v||^2` over real r is `r_free=<v,d>/||v||^2`. Over the original closed duration interval [.2,1], it is `r_box=clip(r_free,.2,1)`. For v=0 the residual is d independently of r; choose r_free=0, r_box=.2. The open sigmoid interval has the same infimum, even when a finite logit cannot attain the boundary. Per-source projection optimality is not a training procedure.

For v=u the unconstrained residual is exactly

```text
.5*s^2 * [w - u*<u,w>/||u||^2].
```

Thus quadratic-term magnitude and new direction are different measurements. A large term parallel to u can be a duration change, although the allowed duration range may block it. No claim follows about retraining P, the controller or the backbone.

## A fixed reparameterization absorbs the centre

At the native centre c=.6 define A=P+.5*c*P^2. It is a fixed matrix per fitted layer, not a fit to the evaluated states. Then

```text
d=s*A*z+.5*s*(s-c)*P^2*z.
```

The same-clock centred approximation has an explicit remainder. A second pair of scalar envelopes uses v=A*z. The free/box error is no larger than the same-clock error because s is in the original box. These bounds concern the layer's increment in its native conditional frame. They neither prove that a small controller can represent the relaxed coefficient nor predict a learned K1 trajectory.

For frame models, z=Q(x)^T*x and the final increment is Q(x)*d. Orthogonality preserves all the displayed residual norms. The frame and original clock remain live functions of x. Replacing the layer by `x+s(x)*Q(x)*A*Q(x)^T*x` therefore gives the same algebraic identity at that layer. Its complete derivative includes both frame and clock derivatives. Local measurements use native, unmodified inputs; the separate all-layer substitution visits different states and must be evaluated directly.

## What flattening a spectrum would and would not do

If `P^2=lambda*P`, the K2 increment is exactly `(s+.5*lambda*s^2)*Pz`: there is no new scalar-clock-free direction. More generally, the residual against that coefficient is `.5*s^2*(P^2-lambda*P)z`.

For a symmetric P whose eigenvalues lie within epsilon of lambda, commutation and the operator norm give

```text
||(P^2-lambda*P)z|| = ||(P-lambda*I)Pz|| <= epsilon*||Pz||.
```

A narrow symmetric eigenvalue spectrum can therefore suppress the geometric contribution of the quadratic term. It is a tradeoff, not an argument that such a spectrum improves FM. No fitted spectral constraint is applied in this study.

Equal singular values have a different meaning. Both `P=lambda*I` and `P=lambda*[[0,-1],[1,0]]` have identical flat singular values. For z=e1, the first has w parallel to u; in the second, u and w are orthogonal. Singular-value flatness alone does not remove this extra direction. Equal eigenvalues without normality also do not imply P is near a scalar matrix: a Jordan block has equal eigenvalues and a nonzero nilpotent part. This blocks treating eigenvalue and singular-value flattening as interchangeable proposals.

## Literature: transfer boundaries

- [Al-Mohy and Higham, SIAM SISC 2011](https://eprints.maths.manchester.ac.uk/1591/) studies computing the action exp(tA)B through matrix products and backward-error control. It supports distinguishing matrix action from a complete matrix representation. It does not predict an advantage of a trainable K2 FFN; the present scalar projection is not that algorithm. This is a classical numerical reference, not a frontier performance recommendation.
- [AREX, October 2026 preprint](https://arxiv.org/html/2610.03483v1), Sections2–3, derives a moment-matched affine backbone for the *external* FM sampler and integrates its residual numerically. It motivates asking which part has an actual data-space interpretation. Its training-free sampler evidence does not validate exact integration of our freely learned hidden P, and we do not change heads, moments, caps or inference here.
- [Unfolding Generative Flows with Koopman Operators, v2](https://arxiv.org/html/2506.22304v2), Sections4–5 and AppendixC, learns a teacher-related lift and consistency objective. Its trajectory-fidelity ablation differs from endpoint-quality comparisons. A generic hidden projection is not such a lift, and its results do not justify adding a consistency loss to our stopped recurrence family.

The latter two are distinct external-flow methods. This bounded source check identifies no theorem that upgrading a free hidden polynomial's order must improve the FM regression objective. It is not an exhaustive novelty review or a claim that no such architecture can work.
