# Derivation Package: why shrinking or flattening P can cost accuracy

11 November 2026. Mathematical explanation with existing experiment bindings.

## Target

Explain how scalar contraction and singular-spectrum flattening change the
actual FFN function and its task error. Distinguish exact consequences from
possible explanations of the recorded negative experiments. This note adds
no training result and does not change the selected ISRU-D K8 candidate.

## Status

**COHERENT AS STATED** for the identities, counterexamples and conditional
local-loss argument. They explain why better norm/stable-rank summaries need
not improve quality. They do not isolate a unique causal mechanism behind
the recorded training differences.

## Invariant Object

The task prediction and its loss, L = E||v_theta(z,t)-u(z,t)||^2. A raw P
spectrum is a proxy for part of the parameterization, not this objective.

## Assumptions

- For the scalar-clock scaling calculation, entrance h and s(h) are fixed
  while P is changed to cP, with 0 < c < 1.
- For ISRU-D scaling, the entrance h0 and D(h0) are fixed while P changes.
  Holding D in internal time does not detach it during normal backpropagation.
- The single-increment calculations hold the distribution of q fixed and use
  its finite uncentered second moment M = E[qq^T].
- The toy target Bq uses M = I and no other changing network components.
- The local loss argument assumes differentiability, and invokes a positive
  semidefinite Hessian only when explicitly assuming a local minimum.
- The spectrum discussed here is the singular-value spectrum. No claim of
  equal eigenvalues, normality or negative real eigenvalues follows from it.

## Notation

P: learned matrix; q: feature sent to it; M: feature second moment;
A: frozen conditional generator; T_K: degree-K Taylor polynomial;
E: perturbation of A; R: perturbation of P; c: scalar contraction;
sigma_i: singular values; ||.||F: Frobenius norm.

## Derivation Strategy

Start from the function, show how the interventions change it, calculate their
effect on a task-weighted error, then return to the complete learned network
and the measured intervention results.

## Derivation Map

1. Exact scaling identifies which Taylor terms change under contraction.
2. A linear target with an explicit optimum demonstrates a cost from shrinking
   or flattening even when matrix energy is preserved.
3. Feature moments identify the data-dependent quantity omitted by raw spectra.
4. Polynomial differentials and a local loss expansion explain sensitivity
   of trained weights without asserting a universal loss increase.
5. Existing frozen, training-time and feature-response studies delimit what
   has actually been established.

## Main Derivation

### Step 1. Contraction changes the internal flow and the polynomial

**Identity.** For the original frozen scalar clock,

```text
T_K(sP)h = sum_{k=0..K} s^k P^k h / k!
T_K(scP)h = sum_{k=0..K} c^k s^k P^k h / k!
exp(scP)h = exp((cs)P)h.
```

At fixed h and s, shrinking P is equivalent to shortening the internal time
of the exact linear flow. For the finite polynomial, term k is multiplied by
c^k. With c = 0.5, term 1 is halved, term 2 is quartered and term 8 becomes
1/256 of its old value. These are relative term scalings, not a statement
that the eighth term was large or causally useful in a trained model.

For selected ISRU-D K8, A = P D(h0). At the same entrance, P -> cP gives
A -> cA and exactly the same c^k term rule. Compensation would require other
parameters/features or internal time to change; it is unavailable in a frozen
intervention and is not guaranteed during constrained training. The original
clock is bounded, so arbitrary inverse-time compensation is not available.

### Step 2. An exact accuracy-cost counterexample

**Proposition under the toy assumptions.** Let the desired increment be Bq,
with B = diag(3,1) and E[qq^T] = I. For any candidate matrix C,

```text
L(C) = E||Cq-Bq||^2 = ||C-B||F^2.
```

The native optimum C = B has L = 0. A half-scale contraction gives
C = diag(1.5,0.5) and L = 2.5. A complete Frobenius-preserving flattening
keeps the original singular vectors and replaces both singular values by
sqrt((3^2+1^2)/2) = sqrt(5), giving

```text
C_flat = diag(sqrt(5),sqrt(5))
||C_flat||F^2 = ||B||F^2 = 10
L(C_flat) = (sqrt(5)-3)^2 + (sqrt(5)-1)^2
          = 20 - 8sqrt(5) = 2.11145618...
```

Both interventions worsen a previously exact map. At q = (1,1), the desired
increment (3,1) changes to (1.5,0.5) or approximately (2.236,2.236).
The flat matrix has the same energy but the wrong directional response.
Adding the same outer residual to both sides does not change this difference.
This counterexample proves that smaller/flatter is not sufficient for lower
error; it does not prove that every regularized learning problem gets worse.

### Step 3. Actual features weight different singular directions differently

**Identity.** For a fixed feature law,

```text
E||Pq||^2 = tr(P M P^T)
E||(P-B)q||^2 = tr((P-B) M (P-B)^T).
```

If P = U diag(sigma) V^T, and v_i is a right singular vector, then

```text
E||Pq||^2 = sum_i sigma_i^2 * E[(v_i^T q)^2].
```

Frobenius preservation keeps sum sigma_i^2 fixed; it does not preserve this
weighted sum unless the relevant feature energies satisfy extra conditions.
It also does not preserve the target response even when M = I, as step 2
shows. With retraining, q and its alignment can change alongside P.

### Step 4. High-order computation depends on the complete matrix

**Identity.** At fixed entrance and A, the polynomial differential in direction E is

```text
D T_K(A)[E] h = sum_{k=1..K} (1/k!) *
               sum_{j=0..k-1} A^j E A^(k-1-j) h.
```

For ISRU-D at fixed h0, E = R D(h0). Multiple positions of the perturbation
appear because A and E need not commute. Flattening P's singular values
changes these matrix products. It generally cannot be reduced to a harmless
change of a scalar internal time.

**Counterexample.** P = I and P = -I have the same completely flat singular
spectrum. Their exact flows are e^s I and e^(-s) I: one expands and the other
contracts for positive time. A flat singular spectrum alone is not a
certificate of contractive dynamics. For the actual input-conditioned map,
gate/diagonal derivatives and downstream operations also enter sensitivity.

### Step 5. A trained solution need not improve under a spectral perturbation

**Local approximation.** Write p = vec(P), r = vec(R), with other parameters
fixed. For a smooth loss,

```text
L(p+r) - L(p) = grad L(p) dot r + 0.5 r^T H r + o(||r||^2).
```

At an exact local minimum, grad L = 0 and H is positive semidefinite. A small
perturbation with positive curvature raises the loss to second order. A
spectral edit is not selected to reduce task loss, so there is no general
reason for it to improve a trained model. At an approximate solution the
linear term can have either sign, consistent with small favorable cases.

This argument applies locally to frozen edits. During training, the extra
projection/relaxation changes the optimization path and possibly accessible
functions. Whether it helps generalization, hurts fitting or can be compensated
is an empirical question, not settled by the frozen-minimum argument.

## Remarks and Interpretation

- The original [decay comparison](../research-muon-pd-2026-11-02/wave2/results/findings.md)
  reduces Product16 scalar-K2 mean P norm from 19.173 to 2.601 at decay 0.1,
  while field MSE changes 0.023849 -> 0.072046. The second-term/input RMS
  falls 14.95 -> 0.868. Computation and sensitivity change together; this is
  not isolation of the second term's causal effect.
- The [frozen spectrum experiment](../autoresearch/professor-20261103/cycle7-results/findings.md)
  preserves Frobenius energy and retains paired norm-matched scalar controls.
  A small Ring8 K1 gain at 20k fails to persist at 80k; stronger flattening
  harms the tested fields. This is functional sensitivity, not a training result.
- The [training-time comparison](../autoresearch/professor-20261103/cycle8-results/findings.md)
  raises P stable rank as intended, while all constrained arms worsen field
  and sampling metrics. These are results for the registered relaxation setting.
- The [feature audit](../autoresearch/professor-20261103/cycle9-results/findings.md)
  measures Product16 block2/solve1 P norm 19.460 -> 16.131, but feature/input
  RMS .2972 -> .3885, actual gain 11.073 -> 11.304 and first-term/input RMS
  3.292 -> 4.392. Lower raw norm coexists with a larger actual response.
  The [existing feature-metric derivation](../autoresearch/professor-20261103/feature-metric.md)
  supplies the moment identities; no whitening or new optimizer is selected.

## Boundaries and Non-Claims

- The exact counterexample establishes possibility, not the unique explanation
  of the project's measured losses.
- Smaller norm can improve an upper sensitivity bound while reducing fitting
  ability. Neither useful selectivity nor harmful amplification is identified
  by a large singular value alone.
- Frobenius energy, operator norm, a full input Jacobian and generation quality
  are different objects. Equal energy does not imply equal function.
- The tests do not reject every spectral constraint, decay strength or training
  schedule. No sweep, new penalty or candidate change follows from this note.
- The selected ISRU-D K8 is still a pending image architecture; the older
  scalar/raw-diagonal experiments are not K8 image evidence.

## Open Risks

The relative contribution of removed task-relevant response, optimization-path
changes and feature compensation has not been isolated by a loss-matched
functional intervention. The current records support the measured distinctions
and exact identities, not a complete causal account of generation quality.

## Verification

The toy contraction/flattening losses and Frobenius preservation agree
numerically with their closed forms. The K8 polynomial differential agrees
with a centered directional finite difference within 9.30e-12. Fourteen quoted
empirical values match the linked source reports; all five local evidence
links resolve. All eleven protected Report2 hashes match and both finalized
report directories are unchanged. This is a derivation check, not new training.
