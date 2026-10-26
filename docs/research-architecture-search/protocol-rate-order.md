# Rate placement: DP versus PD at K1

Registered before training. The archived geometry analysis does not select
P-only whitening: anisotropic input moments and standardized residual
alignment are shared with SwiGLU. Instead test a structural choice introduced
when the professor's scalar duration was extended to channel durations.

## Operator and motivation

The scalar layer h+s(h)Ph does not distinguish multiplying before or after P.
For channel rates D(h)=diag(s(h)), that choice becomes consequential:

```text
parent:    F_after(h)  = h + D(h) P h
candidate: F_before(h) = h + P D(h) h
FFN:       F2(SiLU(F1(h)))
```

Only this ordering changes in both solves. Controllers remain independent
two-layer SiLU networks with original sigmoid duration [.2,1], tau10.
Keep K1, middle SiLU, original parameter coordinates, ordinary bank-pair
training and the shared backbone. Add no constraints, losses, learned basis,
normalization, sharing or temperature change.

For scalar rates both operators and their full derivatives agree in exact
arithmetic. For vector rates their difference is [D,P]h. At a common anchor,
the same positive D and P obey

```text
P D = D^(-1) (D P) D.
```

Thus the frozen generators have the same eigenvalues, and their Taylor
polynomials are similar. This does not mean their action on the same h,
singular values, or complete state-dependent Jacobians agree. With J_s the
rate Jacobian, the complete K1 derivatives are

```text
J_after  = I + D P + diag(P h) J_s
J_before = I + P D + P diag(h) J_s.
```

For a downstream loss derivative q at a solve output, local P gradients are
(D q) h^T after P versus q (D h)^T before P. The latter gates features before
learned output mixing. These identities motivate the test but imply no
universal dominance. Scalar commutation, vector noncommutation, generator
similarity, full Jacobians and P gradients are independently tested.

No inference equates hidden P with the data-space velocity Jacobian. The
frozen spectral comparison is at matched D/P, not a promise that the two
trained models retain equal spectra. Input gating is not feature whitening.

## Frozen screen

Four arms: unchanged vectorK1 parent, vector-before candidate, small SwiGLU
(ffn_hidden96), full SwiGLU (ffn_hidden128). Data: Gaussian2D, Ring8, rotated
product-bimodal16D. All use existing validated splits and analytic oracles,
training seed0,5000 updates,batch512,AdamW3e-4,wd0,clip1, unchanged eval
and Heun32 sampling. Two one-thread CPU workers with10s telemetry; no GPU
reserved for these tiny jobs.

Parent and candidate initial tensors, RNG states, parameter counts and
linear MACs match exactly; both full velocity heads start at zero. Their
internal hidden functions generally differ for vector rates, so this is
not a same-function or same-function-class reparameterization claim.
Unchanged parent/SwiGLU endpoints must reproduce the preceding bank-mode
study. A scalar commutation unit test supplies an algebraic control without
training a duplicate scalar architecture.

Primary readout is the existing uniform-time validation population-oracle
MSE; the existing fixed-time common panels supply field and complete
Jacobian errors. Retain all training curves, sampled-distribution SW2,
P spectra, duration statistics and the actual-anchor commutator magnitude.
MACs, parameter counts and observed training time must accompany quality.

To extend the candidate it must improve primary MSE by at least10% over
both SwiGLU budgets on at least one nonlinear task, regress at most5%
against the better baseline on other nonlinear/Gaussian controls, and have
fixed-time Jacobian error and sampling SW2 within10% of the better baseline
on both nonlinear tasks. It must also improve the unchanged ODE parent on
the task supplying the lead. A small noisy-sampling difference alone is
insufficient. These are engineering continuation thresholds, not p-values.
Failure stops this ordering variant; do not rescue it with scale sweeps,
K2, different seeds or combined modifications.

Passing would justify the pre-existing full requirements: a longer matched
budget, fresh independent holdout, nonlinear case studies, practical cost
audit and mechanism/capacity controls. A5000-step screen alone cannot
complete the useful-architecture goal. Report1 is untouched.

## Provenance and audits

Freeze the new runtime separately from all earlier runtimes. Verify deployed
delta hashes and native tests before launch; persist actual Fourier buffers.
Audit all checkpoint identities/config/source/data hashes, exact initial
tensor pairing and training-generator state pairing at every snapshot.
Recompute analytic oracle arrays and held-out validation readouts; later
collect selected raw checkpoints and replay final fields from frozen source
on Mac. Scheduled-task state alone is insufficient evidence of live work:
inspect actual supervisors/children and telemetry.
