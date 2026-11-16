# What changes when the stage matrices change?

## Target

Explain the local feature-map difference between four learned stage matrices
and their fixed mean. The immediate target is an approximation with a
testable leading term. Sampling quality remains a separate measured outcome.
This concerns the state-evolving midpoint controls. Four midpoint steps use
eight matrix applications and have numerical order2; this is not the
original frozen-diagonal Taylor-K8 block.

## Status

COHERENT AS STATED for the small-internal-time expansion below. Its accuracy
at the trained internal horizon 1 must be measured, not inferred from the order.

## Invariant object

The local hidden-feature map at a fixed layer entrance h. Both maps use
four explicit-midpoint steps, the same response and the same average matrix.
This removes changes to the average generator from the local comparison.

## Assumptions and notation

- W0,...,W3 are fixed finite matrices during a local solve.
- q(y)=y/sqrt(1+1.2*y^2), coordinate by coordinate; fj(y)=Wj q(y).
- Wbar=(W0+W1+W2+W3)/4; fbar=Wbar q=(f0+f1+f2+f3)/4 exactly.
- Jj denotes the Jacobian of fj, evaluated at the fixed entrance h.
- The internal horizon is tau; delta=tau/4. It is not the flow-matching
  physical time t, which is fixed while collecting each layer entrance.
- M(delta,f)(h)=h+delta*f(h+delta*f(h)/2)is explicit midpoint.
- S(tau)applies M with f0,f1,f2,f3in that chronological order.
  C(tau)applies M with fbar four times. R(tau)uses the reversed stage order.
- The fields are smooth. ISRU has bounded derivative and finite matrices
  give globally Lipschitz bounded fields. Taylor remainders here are local
  in tau at a fixed h, uniform on a bounded set of entrances.

## Derivation strategy and map

1. Expand one midpoint update through second order.
2. Compose all four updates, keeping the cross-stage Jacobian terms.
3. Subtract the four-step mean-generator map.
4. Identify the noncommuting contribution and reverse its order as a check.
5. Measure the remainder at fixed horizons on actual trained entrances.

## Main derivation

**Approximation 1: one update.** Taylor expansion at h gives

```text
M(delta,fj)(h) = h + delta*fj + delta^2*Jj*fj/2 + O(delta^3).
```

**Approximation 2: composition.** Applying f0before f1creates J1*f0.
Keeping all such chronological terms gives

```text
S(tau)(h) = h + delta*sum_j fj
           + delta^2*(sum_j Jj*fj/2 + sum_(i<j) Jj*fi)
           + O(tau^3).
```

**Identity and subtraction.** The first-order term of C is identical. Its
second-order term is delta^2*sum_(i,j) Jj*fi/2, since four steps of fbar
have total time tau and midpoint has order 2. Therefore

```text
K(h) = sum_(i<j) (Jj*fi - Ji*fj)/32
S(tau)(h) - C(tau)(h) = tau^2*K(h) + O(tau^3).
```

The factor 1/32 comes from delta^2/2=tau^2/32. With the bracket convention
[fi,fj]=Jj*fi-Ji*fj, reversing the entire schedule changes its sign:

```text
S(tau)(h) - R(tau)(h) = 2*tau^2*K(h) + O(tau^3).
```

**Identity for this model.** q'(h)=(1+1.2*h^2)^(-3/2), so

```text
[fi,fj](h) = Wj diag(q'(h)) Wi q(h)
            - Wi diag(q'(h)) Wj q(h).
```

A matrix-only commutator misses the input-dependent derivative. The existing
two-stage diagnostic checks this formula; this extension tests the complete
four-stage sum, its coefficient and its accuracy on the new checkpoints.

**Shared-generator null.** If all Wj are equal, S=C=R exactly for the same
discrete scheme. K=0. A small defect alone cannot establish task benefit.

**Scalar-clock control.** If Wj=aj*P, the fields are scalar multiples of
one field and every bracket vanishes. For exact flows, the composition is
the flow of P*q for time delta*sum_j aj. Thus, if mean(aj)=1, it is the
same exact time-tau map as the constant generator. Finite midpoint steps
can still differ at order tau^3 because their truncation errors depend on
the individual durations. A nonzero finite-step difference does not by
itself prove a noncommuting effect.

## Remarks and interpretation

Sharing removes both stage-dependent directions and their ordering effects.
The leading difference at a fixed average generator is a feature-weighted
bracket sum. This gives a quantitative hypothesis, beyond merely observing
that reversing the stages changes the output.

This is classical flow-composition mathematics applied to the registered
block. The primary-source reference is Blanes, Casas and Murua, *Splitting
methods for differential equations*, Acta Numerica 2024, sections 1.1--1.2
(linear commutator expansion and nonlinear flow maps),
[full text](https://arxiv.org/html/2401.01722v1). The coefficient above is
derived directly for our four equal-duration stages and explicit midpoint;
it is not presented as a new numerical-analysis theorem.

## Fixed diagnostic and falsifiers

Extend the existing `sharing_probe.py` rather than add another trainer.
Use the existing five physical clocks 0.1/0.3/0.5/0.7/0.9 and the same 16
interpolation pairs, six layer entrances and fixed 20k checkpoints. Include
the original two shared models, two anchored releases and two new
mean-preserving releases. Keep the common selection assets; this is a
post-training mechanism diagnostic, not new confirmation of sampling gains.

Evaluate tau=1,1/2,1/4,1/8,1/16with the original stage order, mean generator
and reversed order. Record actual changes, signed vector predictions,
remainder RMS and prediction cosine. Test the formula independently against
linear matrix exponentials, nonlinear JVPs and the two null controls before
observing trained states. Preserve raw per-layer/clock values.

If the remainder dominates at tau 1, do not claim K explains the actual
trained map. If the tiny-horizon remainder does not approach cubic scaling,
report that failure rather than extend the horizon grid until it passes.
The existing full-field reverse/mean/base interventions remain descriptive
frozen-weight sensitivities; they do not measure the benefit of retraining.

## Boundaries and open risks

The average Wbar itself differs between independently trained models. This
within-checkpoint decomposition does not isolate its training effect.
The new coordinates have Wbar=P exactly; the anchored coordinates generally
do not. Effective matrices must be reconstructed accordingly.

An accurate local expansion does not establish a causal explanation of final
SW2 or KL. Such a claim needs the measured field/sampling outcomes and a
controlled intervention. The current density and solver queues stay frozen;
this diagnostic changes neither their inputs nor their execution.

## Documented native execution

Reuse the unchanged CPU environment and one thread. Freeze the verified
source in `schedule-decomposition/repo` under the common raw study root.
After the kernel checks pass, a fresh documented pass runs this command
once, verbatim, with absent output paths:

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=-1 /Users/mr.z/ode-workspace-prep/.venv-test/bin/python -u /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-decomposition/repo/studies/muon_pd_20261102/coefficient_20261116/sharing_probe.py --root /Users/mr.z/ode-workspace-prep/coefficient-20261116 --data-root /Users/mr.z/ode-workspace-prep/data/synthetic --contrast-study /Users/mr.z/ode-workspace-prep/coefficient-20261116/mean-preserving --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-decomposition/frozen-probe.json --arrays-output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-decomposition/arrays
```

Require exit0 and passed kernel checks; six model/task cases; thirty
layer/clock records per case; five horizons per record; exact native local
formulas; unchanged checkpoint tensors; and all bound arrays. All ordinary
performance comparisons remain in the existing primary analysis. These six
cases are the subset with defined internal stage operators, not new baselines.

The full frozen diagnostic retains actual entrances, effective matrices,
predicted bracket coefficients, observed local-map differences and all
full-field counterfactual predictions outside Git. Run `verify_schedule.py`
after completion to reconstruct the maps, scalars and field MSE in NumPy,
binding native checkpoints and the raw array hashes. The native formula
check passes before the entrance/matrix calculation; independent NumPy
agreement alone is not a substitute for it.
