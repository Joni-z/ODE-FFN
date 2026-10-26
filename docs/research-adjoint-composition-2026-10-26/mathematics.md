# Local symmetry and complete adjoint composition

These identities scope the next architecture question. They are not a new approximation theorem or evidence of an accepted ODE architecture.

## Two local gradient maps can compose to a nongradient map

Let `L1(x)=Ax`, `L2(y)=By`, with symmetric positive definite A and B. Each is a gradient map: `Li=grad[.5*x^T*Mi*x]`. Each is also an exact time-one ODE flow with symmetric generator `log(Mi)`. Their composition has Jacobian BA, whose skew part is controlled by the commutator:

```text
J_(L2 o L1) - J_(L2 o L1)^T = BA - AB.
```

For example, `A=diag(2,1)` and `B=[[2,1],[1,2]]` give `BA=[[4,1],[2,2]]`, which is nonsymmetric. Scaling both A and B by1/4 makes both maps strict contractions; their logs become negative definite, while their product remains nonsymmetric. Thus even locally stable, gradient linear flows need not compose to a gradient map. For two nonlinear gradient maps, the corresponding two local Hessians must commute at the actual intermediate state to retain symmetry. Pairwise commutation is sufficient but not generally necessary for a longer product to be symmetric.

A locally completed original layer `C[L](h)=.5*(L(h)+J_L(h)^T*h)` is the gradient of `.5*dot(h,L(h))`. That does not make a sequential composition of completed layers a gradient. Unconstrained encoders, decoders, FiLM and normalization add further transformations; local properties cannot simply be assigned to the complete predicted field.

## Symmetric P is not enough even for one original layer

For the original once-computed scalar clock,

```text
L(h)=h+a(h)Ph+a(h)^2/2 P²h
u=Ph+aP²h
J_L=I+aP+a²/2P² + u*grad(a)^T.
```

If P is symmetric, the polynomial part is symmetric, but the final rank-one term generally is not. Its skew difference is `u*grad(a)^T-grad(a)*u^T`; it vanishes only when the two vectors are collinear, including their zero cases. The actual two-layer SiLU clock imposes no such condition. A deterministic original-layer example with symmetric diagonal P passes the implementation tests and retains nonzero skew. This does not prohibit spectral constraints; it rejects using P symmetry alone as a complete-field gradient guarantee.

## Exact cross-block decomposition

Write the actual hidden computation as three residual blocks with derivatives `J_Bi=I+Di`, evaluated at their native forward states. Di includes the complete normalization, time FiLM and FFN derivative, and is not P or a fixed generator. Let q be the final normalization/output-head sensitivity seeded by the actual input x. The complete raw-network adjoint is

```text
J_G^T*x = Win^T*(I+D1^T)*(I+D2^T)*(I+D3^T)*q.
```

The correctly ordered expansion is

```text
degree0: I
degree1: D1^T + D2^T + D3^T
degree2: D1^T D2^T + D1^T D3^T + D2^T D3^T
degree3: D1^T D2^T D3^T.
```

Multiplication order is preserved. These degrees count residual derivative factors and are unrelated to the local Taylor order K or the external sampling solver. A reverse polynomial recurrence computes them without forming dense hidden Jacobians: start with coefficient q at degree0; at each reverse block, add its derivative action on coefficient k to coefficient k+1. The resulting four output contributions sum exactly to `.5*(G+J_G^T*x)`, with `.5*G` included in component0.

Every coefficient depends on all actual forward states. Its input derivative must differentiate those states, gates, normalization factors and the seed x. A partial reverse sum does not delete the corresponding primal blocks, and is not guaranteed conservative. Individual skew components can cancel in the full Hessian. Component norms alone cannot allocate causal importance or predict retraining.

## A valid construction rule, with a narrower guarantee

If f is a gradient, then `E^T*f(E*x+b)` is a gradient by the chain rule. Sums of such pullbacks also remain gradients. Arbitrary sequential compositions do not share that guarantee. [Gradient Networks, Proposition0.1](https://arxiv.org/html/2404.07361v3) uses this construction with a suitable activation gradient. It is established prior art, not a new method or evidence that substituting a local ODE potential will improve generation. No such substitution is selected by this diagnostic.

Finally, a conservative vector field means a potential exists; it does not mean a gradient-descent trajectory conserves energy. Under `dx/ds=-grad(E)`, `dE/ds=-||grad(E)||²`. Neither property alone proves a useful hidden-state computation for JiT/FM.
