# Forced-response gate inside SwiGLU: design review

**Conditional PASS for one minimal controlled operator comparison.** The mathematics is clean, and the proposal preserves the independently projected gate/value structure of the strongest existing baseline. That is a useful distinction from the failed rebuilt FFNs. The evidence does not specifically favor the phi1 response, however. Its research status is a weak-prior hypothesis about a cheap adaptive gate, not a demonstrated FM mechanism or an advance obtained merely by embedding the baseline at initialization.

No code or training is performed by this review. Keep the proposed operator, fixed rate bound and existing width; do not add a slope, temperature, solver-order or width search.

## Exact conditional forced response

For each ordinary SwiGLU unit, let

```text
a = gate projection of h
v = independent value projection of h
s = sigmoid(a), b = a*s
lambda = tanh(eta), eta initially zero
c = lambda*s.
```

Hold a,b,s,c fixed during the internal solve `dy/du=c*y+b`, starting from y(0)=0 and ending at u=1. Its output is

```text
Psi = b*phi1(c)
phi1(c) = expm1(c)/c,  phi1(0)=1
         = integral_0^1 exp(t*c) dt.
```

Use the ordinary SwiGLU output projection on `Psi*v`. The forcing is the existing SiLU gate, not a new dense controller. This is a conditional linear solve in an auxiliary state, not the external flow-matching trajectory.

The augmented generator and its exponential are

```text
M = [[c,b],[0,0]]
exp(M) = [[exp(c),b*phi1(c)],[0,1]].
```

Applying the professor's Taylor-polynomial operation to initial augmented state `(0,1)` yields:

```text
K1: b                          ordinary SwiGLU gate
K2: b*(1+.5*lambda*s)           first adaptive correction
exact: b*phi1(lambda*s).
```

This is a valid extension from homogeneous to forced conditional dynamics. M now depends on the current input through b and s, so it is not the original globally fixed dense P. The construction supplies a reproducible operator family; it does not establish novelty or FM-specific necessity.

## Derivatives, bounds and stable evaluation

Let primes denote derivatives with respect to the scalar argument. Then

```text
phi1'(c) = [c*exp(c)-expm1(c)]/c^2,    phi1'(0)=.5
phi1(c)+c*phi1'(c)=exp(c)
Psi'(a)=s*phi1(c)+a*s'(a)*exp(c)
partial Psi/partial eta
  = b*s*phi1'(c)*(1-lambda^2).
```

The complete FFN input Jacobian is

```text
Wout [diag(v*Psi'(a))*Wgate + diag(Psi)*Wvalue].
```

This keeps both projection paths and includes the state-dependent clock derivative. For K2, replace phi1 by `1+c/2` and the exp(c) term in the displayed Psi' identity by `1+c`.

Since phi1 is increasing on the real line and |c|<=1,

```text
1-exp(-1) <= phi1(c) <= exp(1)-1.
```

This bounds the gate's gain relative to b, including its continuous interpretation at b=0. K2's relative gain lies between .5 and 1.5, so the exact and truncated arms have different permitted gain ranges. Both remain positive. Any exact-versus-K2 effect includes the response-curve and range difference; it is not evidence about numerical accuracy of the external generative ODE.

The augmented matrix has eigenvalues c and 0, but its norm can grow with b. Even at c=0 its exponential is a shear with off-diagonal b. Bounded eigenvalues therefore do not certify an augmented operator norm, and neither the relative-gain bound nor a scalar gate derivative bound controls the complete SwiGLU product or velocity network.

For evaluation near zero, use a polynomial branch, for example

```text
phi1(c)=1+c/2+c^2/6+c^3/24+c^4/120+...
phi1'(c)=.5+c/3+c^2/8+c^3/30+...
```

The degree-4 value remainder on |c|<=delta is bounded by `exp(delta)*delta^5/720`. Select numerical order/threshold from dtype error requirements and test derivatives near zero. Crucially, form a safe denominator for the non-series branch *before* division. Writing an inactive `expm1(c)/c` branch inside a selection still creates 0/0 and can contaminate autograd. Initial value and derivative must be exactly 1 and .5. This is numerical implementation hygiene, not a new architectural hyperparameter to sweep.

A concrete adequate proposal is degree-8 Horner evaluation for |c|<.05 and the safe-denominator ratio elsewhere. Its value remainder is at most `exp(.05)*.05^9/10!`, below 5.66e-19; the derivative remainder is at most `exp(.05)*.05^8/(10*8!)`, below 1.02e-16. These are mathematical truncation bounds, not floating-point error bounds. Test float32/float64 forward and backward behavior at zero, small positive/negative inputs, and both sides of the branch boundary. The quotient backward can suffer cancellation even when it is finite, so checking only NaNs is insufficient. No custom autograd is needed.

## Initialization and what the extra tangent really is

At eta=0, Psi=b exactly. All ordinary SwiGLU parameters, complete functions, input Jacobians and common-parameter derivatives therefore match the baseline. Zero-initialized extra vectors consume no RNG draws. The new eigenparameter response is

```text
partial output/partial eta_i
  = .5*Wout[:,i]*a_i*sigmoid(a_i)^2*v_i.
```

Adaptive K2 and exact phi1 have the same complete initial parameter Jacobian and NTK. Plain SwiGLU lacks these extra directions; its full NTK is not identical. A common nonzero probe head is necessary when verifying backbone responses because formal zero velocity-head initialization masks them.

This new direction has a generic interpretation. With b=SiLU(a),

```text
a*sigmoid(a)^2 = b + sigmoid(a) - b'(a).
```

Modulo ordinary output-column scaling and gate-bias directions, the added feature is a sigmoid-gated value, `sigmoid(a)*v`. Thus this is not an intrinsically ODE-specific tangent direction. The K2 arm already supplies the relevant generic control; another offset-gate arm is unnecessary for the initial comparison.

The direction is not universally redundant. For a single unit with independent a and v, its v-linear ordinary tangent span contains `b,b',a*b'`. Their negative-tail terms are polynomials times exp(a), whereas `a*sigmoid(a)^2` starts as `a*exp(2a)`; cancelling all exp(a) coefficients forces the putative combination to zero. Conversely, redundancy can occur: with scalar a=v=x at unit weights,

```text
.5*x^2*sigmoid(x)^2
 = .5*(value-bias tangent + value-weight tangent - gate-bias tangent).
```

A wide trained network may have further dependencies. Neither a universal new-capacity claim nor universal redundancy follows from the one-unit calculation.

## Static control and minimal fair comparison

The static gate is `b*phi1(.5*lambda)`, with the same extra parameter count and initialization. Its constant positive channel gains can be absorbed into Wout, including unchanged output bias. Its function class is exactly ordinary same-width SwiGLU; its extra eta direction at initialization is `.25*Wout[:,i]*b_i*v_i`, an output-column scaling direction. It can still alter optimization geometry.

Use only the proposed five arms: the two established SwiGLU widths, and static/K2/exact variants at one fixed existing width. Keep independent gate/value projections, output projection, biases, backbone and FiLM paired with the matching SwiGLU reference. Each new variant adds one gate-hidden parameter per block, with no extra dense matrix MACs; exact phi1 has additional pointwise cost and must be timed. Ordinary SwiGLU itself is the K1 control; do not count dormant eigenparameters in a fictitious K1 arm as useful capacity.

The hypotheses and rejection rules should be explicit before outcomes:

- A gain over the failed old ODE models says nothing here; the matching strong SwiGLU and better existing budget are the actual references.
- Adaptive K2 must materially improve those references and the static control before it is a useful generic adaptive-gate lead. Its success alone is not evidence for an ODE-unique or all-orders mechanism.
- Exact phi1 must additionally show a material advantage over K2 before crediting its higher response terms. A tie favors the simpler generic interpretation.
- Static-only improvement supports redundant reparameterization, not state-dependent forced dynamics.
- Passing the initial screen still needs the declared independent holdout, matched-budget/isolated timing check and transfer guards. Existing validation comparisons are not independent confirmation.
- Failure stops this operator setting; no rate-bound, gate-scale, width or extra-order rescue follows.

The controlled question is therefore narrow but meaningful: **can a canonical forced-response correction improve useful gate shapes inside an otherwise unchanged strong SwiGLU, and do its higher terms outperform their matched generic first correction?** Current evidence motivates preserving the baseline; it does not favor the answer. That distinction is the condition for treating this as one admissible operator experiment rather than another unconstrained gating sweep.
