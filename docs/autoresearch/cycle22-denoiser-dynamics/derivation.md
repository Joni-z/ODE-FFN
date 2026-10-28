# The dynamics of the supervised denoiser

## Target and status

This package derives properties of the clean conditional mean supervised by
our independent Gaussian flow-matching objective. Its invariant object is
that conditional mean, not an arbitrary hidden flow or an endpoint predictor
trained by distillation. No architecture improvement is claimed. Semantic
proof review and numerical verification are recorded separately.

## U0: Assumptions and typed notation

Let d be a fixed positive integer and X in R^d have a finite mixture of
Gaussian distributions, allowing positive-semidefinite component covariances
and hence point masses. Mixture weights are positive and sum to one. This
class covers the registered analytic benchmark laws used below and the finite
support counterexamples. It is not an assumption that natural images have a
known finite Gaussian-mixture population law.

Let e be an independent standard Gaussian, t in (0,1), Z_t=tX+(1-t)e,
and D(z,t)=E[X|Z_t=z]. Let delta(t)>0 depend only on t, with finite conditional
loss. For the Gaussian-observation coordinates, r>0 is noise variance,
Y_r=X+sqrt(r)e, p_r its density, and m(y,r)=E[X|Y_r=y]. Define s=grad_y log p_r,
J_m the d-by-d output/input Jacobian, and Delta m the componentwise trace of
its spatial Hessian. Gradients are columns. C is the conditional covariance.
For precision lambda=1/r, use m_lambda(y), C_lambda(y), and
kappa_lambda=E[(X-m_lambda)||X-m_lambda||^2|y], a vector, not the full tensor.

All equalities below hold at fixed interior t or r. No endpoint-uniform error
or t=0/1 differentiability is asserted. For a compact y set and r in a compact
subset of (0,infinity), Gaussian likelihood derivatives are bounded by a
constant times a polynomial in X. Every mixture component has all moments,
including degenerate components. Dominated differentiation of numerator and
denominator integrals is therefore valid. The denominator p_r is positive;
derivatives of its quotient and logarithm are well defined. Finite sums over
components commute with differentiation. These facts justify the interchanges
used in U2–U7; constants may depend on the fixed compact sets and prior.

## U1: The regression optimum

For every fixed t, the unique squared-loss minimizer up to p_t-almost-everywhere
equality of E[||f(Z_t,t)-X||^2/delta(t)^2|t] is D. Indeed, writing f-X as
(f-D)+(D-X), the conditional expectation of the cross term given Z_t is zero.
The excess risk is E[||f-D||^2|t]/delta(t)^2. Thus the denominator cap changes
time weighting but not this ideal conditional optimum. This statement concerns
unrestricted population prediction, not finite-width optimization.

## U2: Heat and posterior identities

Convolution with a Gaussian of covariance rI gives

```text
partial_r p_r = .5 Delta_y p_r
m(y,r) = y + r s(y,r)
J_m(y,r) = C(y,r)/r.
```

The heat identity follows by differentiating the Gaussian kernel. Its spatial
gradient is -(y-X)/r times that kernel, giving the second equality after
division by p_r. For the third, differentiation of the normalized posterior
gives partial_{y_j} E[X_i|y]=Cov(X_i,X_j|y)/r. Hence J_m is symmetric positive
semidefinite. These are properties of the complete conditional mean, not
constraints proved for a hidden P or one local FFN.

## U3: A necessary denoiser PDE

For every prior in U0 and every y,r>0,

```text
partial_r m = .5 Delta_y m + J_m (m-y)/r.
```

Proof: set l=log p_r. The heat equation implies
partial_r l=.5(Delta l+||grad l||^2). Differentiate spatially to obtain
partial_r s=.5 Delta s+J_s s; J_s is symmetric because s=grad l.
Differentiate m=y+rs:

```text
partial_r m = s + r*(.5 Delta s + J_s s)
            = .5 Delta m + (I+r J_s)s
            = .5 Delta m + J_m(m-y)/r.
```

Smoothness and derivative commutation are justified in U0. This is a necessary
identity of a prior-induced denoiser. No converse, uniqueness from this PDE,
correct-prior identification or performance implication is asserted.

## U4: The identity in our FM coordinates

For y=z/t and r=((1-t)/t)^2, the likelihoods differ only by a factor independent
of X, so D(z,t)=m(z/t,r). Consequently J_m=t J_D, Delta_y m=t^2 Delta_z D,
partial_t(z/t)=-z/t^2, and r'=-2(1-t)/t^3. The chain rule in U3 gives

```text
partial_t D = -(1-t)/t * Delta_z D
              -2/(1-t) * J_D D
              +(1+t)/(t*(1-t)) * J_D z.
```

For the uncapped marginal velocity u=(D-z)/(1-t), another chain rule yields

```text
partial_t D + J_D u = -(1-t)/t * Delta_z D
                      + J_D(z/t-D)/(1-t).
```

The right side need not vanish. This is not a statement about a capped sampler
for t>.95: the supervised D has the same conditional optimum, but replacing
1-t by delta changes the external velocity and its material derivative.

## U5: Target invariance is an extra condition

Take d=1 and X,e independently N(0,1). Put V=t^2+(1-t)^2. Then

```text
D(z,t) = (t/V)*z
u(z,t) = ((2t-1)/V)*z
(partial_t+u partial_z)D = ((1-t)/V^2)*z.
```

The last equality follows by differentiating t/V and adding its product with
(2t-1)/V. At t=.5 it is 2z, not zero for z!=0. At that time the marginal
velocity is zero, but the clean conditional estimate is changing. A
conditional straight interpolation does not make the marginal denoiser constant.

For the same marginal flow, the endpoint map F(z,t)=z/sqrt(V) instead satisfies
(partial_t+u partial_z)F=0 and F(z,1)=z. It transports each particle's final
identity, while D is a posterior mean under the independent training coupling.
At t=.5, F=sqrt(2)z and D=z. Under the actual supervised objective with
delta=.5, F has excess risk 2*(sqrt(2)-1)^2 per coordinate. Thus invariance is
not a necessary property of the current regression optimum. This does not
reject invariant endpoint predictors trained with a different target.

## U6: PDE consistency does not identify the data prior

Keep the true data law N(0,1), but define a predictor using a hypothetical
N(0,v) prior, v>0. It is D_v(z,t)=t v z/(t^2 v+(1-t)^2). Every D_v satisfies
the denoiser PDE because it arises from a valid prior in U0. At t=.5 and v=2,
D_2=4z/3 whereas the true D_1=z. Since Var(Z_.5)=.5, its excess supervised
risk is (1/3)^2*.5/(.5)^2=2/9. A zero PDE residual alone is not a sufficient
architecture selection or quality criterion.

## U7: A precision homotopy needs a third posterior moment

Hold the observation y fixed and write the posterior weight as
exp(-lambda||X-y||^2/2) times the prior. Differentiating a normalized
expectation gives

```text
d m_lambda/dlambda = -.5 Cov_lambda(X,||X-y||^2)
                   = C_lambda(y-m_lambda) - .5 kappa_lambda.
```

For the second equality, write X=m+xi and expand ||X-y||^2. E[xi]=0 leaves
E[xi||xi||^2]+2C(m-y). Gaussian conditional distributions have kappa=0;
general mixture posteriors do not. A covariance-only update omits a known
term unless an appropriate additional closure assumption holds. This is an
identity of the complete posterior, not a causal explanation of an earlier
hidden-covariance architecture failure.

## U8: Same mean/covariance need not give the same homotopy drift

At y=0,lambda=1, prescribe posterior q_A on {-1,1} with weights(.5,.5),
and q_B on {-.5,2} with weights(.8,.2). Both have mean0 and variance1.
Their third central moments are0 and1.5. Define each fixed prior's atom
weights proportional to q_i*exp(x_i^2/2). Conditioning that prior at the
specified y,lambda recovers q exactly. U7 gives drifts0 and-.75.

This refutes a prior-independent closure depending only on (y,lambda,m,C)
across U0. It does not refute a model carrying additional prior information,
third moments or other sufficient latent state. It does not prescribe a
particular moment architecture or validate an ODELayer successor.

## Implication and non-claims

The supervised object has exact, task-specific differential identities.
None says that two linear hidden solves, a smaller P spectrum or a more
accurate local exponential improves prediction. A proposed internal evolution
must specify how its state represents this object and which closure it makes.
The PDE and moment identities are established mathematical relations, not
novelty claims. No loss, target, data law, checkpoint or training setting is
changed by deriving them. Earlier stopped families remain stopped.
