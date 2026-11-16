# Next quantitative question: what bias does the hidden solver introduce?

The equal-dimension control rejects tangential dominance alone as a distinctive
mechanism. Potential-normal error is nevertheless enriched, especially on Ring8.
The next question is whether the exact learned operator and ISRU derivatives
predict that error, rather than describing it after observation.

## Established background and the scope of our question

Numerical-solver dependence in neural ODEs is established prior work:
[Ott et al., ResNet After All?](https://arxiv.org/abs/2007.15386) report that
changing a coarse training solver can reduce performance. [Zhu et al., ICML2022,
Theorem3.2](https://proceedings.mlr.press/v162/zhu22f/zhu22f.pdf) analyze inverse
modified equations for learning dynamics. These results motivate the check;
neither paper proves a benefit for our flow-matching FFN.

Our frozen hidden dynamics use a forward modified-equation calculation. This
is different from the inverse modified equation fitted to known trajectory
data in Zhu et al. In particular, its learning theorem cannot be applied to
our latent features, whose true trajectories are not supervised.

## Assumptions and local expansion

Write internal time as s and numerical step as eta=1/N, distinct from the
outer flow-matching time. Within each fixed stage,

```text
f(y) = P q(y),   q_i(y)=y_i/sqrt(1+a*y_i^2),   a=1.2.
```

For finite P, f is smooth and globally Lipschitz in real y. Finite-interval
solutions exist uniquely. Scheduled models use their exact fixed quarter-stage
boundaries. Taylor remainders refer to bounded neighborhoods of these paths;
no accuracy claim is automatic at the trained coarse step eta=1/4.

For a smooth autonomous stage, let J=f' and H=f''. The exact one-step flow is

```text
phi_eta(y) = y + eta*f + eta^2/2*Jf
              + eta^3/6*(H[f,f] + J(Jf)) + O(eta^4).
```

Explicit midpoint gives

```text
M_eta(y) = y + eta*f(y+eta*f/2)
         = y + eta*f + eta^2/2*Jf + eta^3/8*H[f,f] + O(eta^4).
```

Subtracting, without fitting any coefficient,

```text
M_eta(y)-phi_eta(y) = eta^3*g(y) + O(eta^4)
g(y) = -H[f,f]/24 - J(Jf)/6.
```

Thus the forward modified field is f+eta^2*g+O(eta^3). The inverse modified
field compensating this defect has the opposite leading sign; do not confuse
the two constructions.

## Explicit ISRU formula

With D_i=(1+a*y_i^2)^(-1/2),

```text
q'_i = D_i^3
q''_i = -3*a*y_i*D_i^5
Jv = P(q' elementwise v)
H[f,f] = P(q'' elementwise f elementwise f)
J(Jf) = P(q' elementwise P(q' elementwise f)).
```

The predicted bias depends on P, feature position, response slope and curvature,
not merely a spectral norm. Its two contributions can cancel, so separate RMS
magnitudes do not add. This algebra is standard numerical analysis, not a new
theorem or an architecture improvement.

## Global hidden endpoint prediction

Along the continuous hidden path y'=f(y), integrate

```text
z' = J(y) z + g(y),   z(0)=0.
```

Over the same fixed internal interval, the leading prediction is

```text
y_N(1)-y_ref(1) = eta^2*z(1) + O(eta^3).
```

For the scheduled models, switch P and g at the existing exact boundaries.
There is no extra fitted scale or time warp. Project the prediction onto
q(y_ref) and its complement to predict the same normal/tangential quantities
already measured. First validate the sign, derivatives and asymptotic behavior
on exact linear, scalar quadratic and deterministic ISRU examples.

## Falsifiable follow-up

After those mathematical controls pass, register a full-cohort comparison on
the existing80inputs, six layers and all six checkpoints. Reuse all available
native refinements and continuous references; retain the trained4-step result.
Check whether relative prediction error decreases as eta decreases, and
separately whether the parameter-free prediction explains the trained4-step
error and normal enrichment. Passing at fine steps alone does not explain the
actual trained model. A failure at4steps must remain a failed trained-scale
explanation, even if the asymptotic formula is correct.

This is a hidden-map prediction, not a surrogate metric for generation quality.
Only a subsequently qualified direct intervention could connect it to target
SW2. No new activation, energy constraint or image candidate is selected here.

## Mathematical readiness

The deterministic controls pass at actual exit0. ISRU first/second derivative
errors are5.66e-11/1.79e-10; local residuals decrease at the predicted order on
the exact quadratic and ISRU controls. The global ISRU prediction's relative
residual reaches0.002at64steps; the pure-rotation leading normal work is at
rounding scale. [Complete measured controls](solver-bias-math-check.json).
These controls contain no learned-checkpoint evidence.
