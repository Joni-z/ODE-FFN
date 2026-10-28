# Which mathematical object is being tested?

## Target and status

Status: COHERENT AS STATED. The invariant object is one frozen local map at
a fixed duration, A=cP. Compare its polynomial action T_K(A) with exp(A), then
measure the resulting complete predictor. This does not change the FM
regression objective or assert that a hidden-map error bounds image FID.

## Assumptions and notation

P is a fixed real channel matrix; c is a fixed scalar calibrated in cycle19.
K is the order used during training,1or2. Input vectors are columns; code's
Linear uses the transpose convention consistently. I is the identity, ||.||
is spectral norm and mu(P)=lambda_max((P+P^T)/2). Only the local linear
substep is the solution of dy/dr=Py. SiLU separates two such maps, and the
whole Transformer has additional residual and conditioning operations.

## Derivation map

1. Identify the exact linear flow and truncated polynomial (identity).
2. Bound the truncation tail under an induced norm (proposition).
3. Separate eigenvalue and nonnormal norm information (proposition).
4. Derive a finite semigroup defect (identity).
5. Return to frozen predictor behavior without replacing it by a proxy.

## Main derivation

**Identity.** For dy/dr=Py, y(c)=exp(cP)y(0). The professor recurrence at
constant c computes T_K(cP)y(0). With two solves it is M2 SiLU(M1h). Neither
the exact local flow nor increasing K guarantees lower FM regression error.

**Proposition.** Let a=||A||. Submultiplicativity gives

```text
||exp(A)-T_K(A)|| <= sum_{j=K+1}^infinity a^j/j!
                    <= exp(a)*a^(K+1)/(K+1)!.
```

The second inequality follows from (K+1+m)! >= (K+1)! m!. A large upper bound
does not establish a large actual error; compute the actual matrices too.
No eigenvector diagonalization or normality assumption is used.

**Proposition.** ||exp(cP)|| <= exp(c*mu(P)) for c>=0, by differentiating the
squared norm of a solution. Nonpositive eigenvalue real parts alone do not
imply contraction in the Euclidean norm. For P=-bI+aN with b>0 and N^3=0,

```text
exp(cP) = exp(-bc) * (I + caN + (ca)^2*N^2/2).
```

All eigenvalues are -b, yet transient gain can grow with a. Therefore neither
flattening a singular-value distribution nor clipping eigenvalues by itself
establishes whole-FFN or predictor stability. These are not new claims.

**Identity.** Exact exponentials satisfy exp(A/2)^2=exp(A). In contrast,

```text
T1(A/2)^2 - T1(A) = A^2/4
T2(A/2)^2 - T2(A) = A^3/8 + A^4/64.
```

These defects concern fixed linear maps. They are not semigroup statements
about the input-varying-duration layer or the entire nonlinear FFN. Their
size measures departure from this particular local-flow interpretation,
not whether a learned polynomial is a good image predictor.

## Return to the measured object

Formal exp and folded-polynomial endpoints use the same float32 matrix
application, with each matrix constructed in float64 then cast once. The
old literal recurrence is a separately retained rounding control. Comparing
these expressions prevents assigning every numerical reordering difference
to the exponential function. All downstream states are recomputed.

The operator history uses the final c at all saved steps. That is a fixed
retrospective comparison, not evidence that the original clocks were constant
while training. Validation/panels/generation must be measured separately;
there is no derived monotone relation from tail bounds, map norms or hidden
semigroup accuracy to weighted MSE or FID.

## Boundaries and open risks

No approximation theorem for the unknown learned feature dynamics is assumed.
The tested field is exactly Py, chosen by the professor parameterization.
No new trained method, population performance claim, globally invertible
denoiser, training-causation theorem or architecture nomination follows.
The old synthetic exponential-substitution failures remain unchanged.
