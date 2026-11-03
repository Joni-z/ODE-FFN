# Derivation package: mode mass, boundary flux and ODE-FFN updates

## Target

Explain the measured Ring8 separation between field fitting, local residual
shape and mode allocation. Derive a first-order expression for sector-mass
error and connect it to a K1 ISRU-D matrix update. This is a construction of a
measurable theory line, not a new architecture or a new performance result.

## Status

COHERENT AS STATED under the regularity assumptions below. Oracle symmetry,
sector invariance and continuity-equation identities are exact statements.
The small-error response is a conditional first-order approximation. Its
accuracy at the trained checkpoints is the next numerical question.

## Invariant object

M_k(t): probability mass in a fixed angular sector around target mode k.
The analytic target and Gaussian source give each sector mass1/8. The hidden
matrix spectrum and common-source endpoint MSE are different objects.

## Assumptions

1. The target is the actual Ring8 law: eight equally weighted means on a regular
   octagon, common isotropic variance sigma^2=.04. Source is N(0,I). Sector
   boundaries are angular bisectors. The saved manifest fixes these parameters.
2. Flows are unique on t in[0,1]. The oracle has smooth, globally Lipschitz
   spatial drift: its mixture covariance stays positive, and its finite-mean
   posterior derivatives are bounded. For perturbed flows, assume sufficient
   smoothness, growth bounds and finite moments for a classical continuity
   equation and integrable boundary traces; flux at spatial infinity vanishes.
3. For the first-order statement, density q_epsilon and boundary flux are
   differentiable at epsilon=0, with integrable dominating derivatives so that
   differentiation commutes with the time/sector/boundary integrals. This is an
   explicit hypothesis, not supplied by the finite numerical Jacobian probes.
4. Parameters are finite. For the P-update differential, one K1 matrix is varied
   with all other parameters fixed. Its entrance feature does not depend on
   that same matrix. The downstream network is differentiated normally.

## Notation

- t: global flow-matching time. r: internal feature-flow duration; they are
  separate variables.
- mu_k: target mode means; sigma^2 I: component covariance; S_k: angular sector.
- p_t, u_t: analytic FM marginal density and velocity.
- v_t: a fixed trained model; delta_t=v_t-u_t.
- v_epsilon=u+epsilon delta; q_epsilon,t: its transported density.
- eta_t = derivative of q_epsilon,t at epsilon=0.
- B_k: first-order endpoint mass change in sector k.

## Derivation strategy and map

Analytic mixture velocity -> reflection symmetry -> invariant sector boundaries
-> exact boundary mass flux -> first-order mass response -> matrix-update
sensitivity. Approximation enters only when relating the epsilon derivative to
a finite trained-field error. The invariant quantity remains sector mass.

## Main derivation

### 1. Identity: the actual marginal velocity

For X_t=(1-t)X_0+tX_1, conditional component variance is

```
c(t) = (1-t)^2 + t^2 sigma^2
w_k(x,t) = softmax_k[-||x-t mu_k||^2/(2c(t))]
a(t) = [t sigma^2-(1-t)]/c(t)
b(t) = (1-t)/c(t)
u(x,t) = a(t)x + b(t) sum_k w_k(x,t)mu_k.
```

This follows by Gaussian conditioning within each mixture component and then
averaging its velocity. In particular,1-a(t)t=(1-t)/c(t). Differentiating the
posterior yields the exact complete-field Jacobian

```
J_u(x,t) = a(t)I + t(1-t)/c(t)^2 * Cov_w(mu).
```

Positive posterior covariance can produce directional expansion around mode
boundaries. This is a property of the complete velocity field, not of a hidden P.

### 2. Proposition: the oracle preserves angular sectors

Each reflection Q through a sector boundary permutes the eight means while
preserving covariance and weights. Substituting Qx in the posterior therefore
permutes w_k and gives u(Qx,t)=Q u(x,t). On the reflection-fixed boundary,
Qx=x, so u=Q u and u has no component normal to that line.

A trajectory starting on the boundary stays on it. Uniqueness also prevents a
trajectory starting off the boundary from crossing it: at a first crossing,
the backward solution would have to be the boundary solution. Thus each sector
is invariant under the exact oracle flow. Gaussian source symmetry gives
M_k(t)=1/8 for every t, as does the target's rotational symmetry.

The native128step reference had zero sector crossings in4,096particles.
Oracle normal velocities on tested boundary rays are below2.61e-13. These are
implementation checks supporting the exact symmetry calculation above.

### 3. Identity: sector-mass change of the learned field

Let q_t be the density transported by v. Integrate its continuity equation
across S_k and apply the divergence theorem (with the stated tail condition):

```
d/dt integral_(S_k) q_t dx = - integral_(boundary S_k) q_t v_t dot n_k dS.
```

Since u dot n_k=0 on that same boundary, v dot n_k=delta dot n_k there.
Net normal flux changes mode-sector probabilities. Gross particle crossings
can cancel; they are not themselves evidence of incorrect sector mass.

For a boundary ray x=r e_beta in2D, its line element is dr, not r dr. Let
F_j(t)=integral_0^infinity q_t(r e_beta_j) v_theta(r e_beta_j,t)dr,
with the angular normal pointing counterclockwise. For the sector between
boundaries j-1 and j, M'_k=F_(j-1)-F_j. Adjacent fluxes cancel in the sum.

### 4. Conditional first-order response around the oracle

Differentiate the perturbed continuity equation at epsilon=0:

```
partial_t eta + div(u eta) = -div(p delta),    eta_0=0.
```

Integrating over S_k removes the eta*u boundary term because u dot n_k=0.
The differentiability/domination assumptions permit interchange with the
integrals and give

```
B_k = - integral_0^1 integral_(boundary S_k) p_t delta_t dot n_k dS dt
M_k^epsilon(1) = 1/8 + epsilon B_k + o(epsilon).
```

Equivalently, use the p-weighted angular flux on each ray and take adjacent
flux differences. Sum_k B_k=0. Cauchy--Schwarz bounds |B_k| by the product of
sqrt(integral p_t over its time-boundary surface) and
sqrt(integral p_t |delta dot n_k|^2 over that surface).

This response is a signed boundary functional. Oracle MSE is a quadratic
full-domain average. Their scalar rankings need not agree. This does not
remove the transport error bounds in[transport-bound.md](transport-bound.md);
it identifies a more specific quantity for the observed multimodal task.

### 5. Exact differential of the response functional with respect to P

A current K1 ISRU-D map is F(h)=h+P q(h), q_i=h_i/sqrt(1+1.2h_i^2).
For a matrix change DeltaP at a fixed entrance h,

```
DeltaF = DeltaP q(h)
D_P v[DeltaP] = J_downstream(x,t) DeltaP q(h(x,t)).
```

The latter is the parameter directional derivative of the complete predictor.
For B_k, p_t is the fixed oracle density, so differentiating the boundary
functional gives

```
D_P B_k[DeltaP] = <G_k,DeltaP>_F
G_k = - integral_0^1 integral_(boundary S_k)
            p_t [J_downstream^T n_k] q(h)^T dS dt.
```

Consequently, the squared predicted mass change is the positive-semidefinite
quadratic form sum_k <G_k,DeltaP>_F^2. Since sum_k G_k=0, its rank on a single
matrix's parameter space is at most7 for Ring8. It depends on features,
downstream derivatives and boundary geometry. P singular values alone do not
specify it. The differential of B is exact; B's approximation to actual finite
mass drift still has the first-order scope in step4.

## Special cases and interpretation

- delta=0 gives B=0. A perturbation tangent to every sector boundary also gives
  B=0.
- Zero normal velocity is sufficient, not necessary. A rotational perturbation
  delta=omega(t) R_90 x has equal p-weighted flux on every boundary ray, hence
  B_k=0 despite particle crossings. Rotational symmetry preserves equal sector
  masses while allowing particle reassignment.
- Equal sector masses do not determine within-sector shape. Conversely, a
  better centered residual distribution can coexist with worse sector masses.
- The current feature response q is the exact scalar flow of du/dr=-u^3 at
  internal r=.6. ISRU-D uses F2(SiLU(F1(h))); copied Free ISRU uses A q(Bh).
  This explains the cubic-to-ISRU-to-diagonal construction without equating r
  to global FM time or treating the two architectures as function-matched.

## Boundaries and open checks

The current saved arrays measure masses, paths and moments at epsilon=1. They
do not yet evaluate B_k or validate the linear regime. Next: compute the
p-weighted boundary-normal flux for the six frozen checkpoints and test weak
oracle/model interpolations before selecting an architectural constraint.
A mass-sensitive update rule would require that numerical validation, a
comparison with ordinary models and its measured fitting/compute tradeoff.

The theory is a symmetric Ring8 special case. Extending it to other mixture
geometries requires their invariant boundaries, not an assumed circle model.
Training-cache and shared-minibatch effects remain possible sources of the
observed common drift. These are separate from the measured mode imbalance.
