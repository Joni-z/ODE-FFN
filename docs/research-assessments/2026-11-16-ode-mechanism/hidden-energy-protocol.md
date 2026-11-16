# Next diagnostic: what does the trained hidden solver change?

Status: registered after the complete first-order calibration. Deterministic
mathematical controls pass; learned-state measurement has not launched.
Purpose: explain the existing observation that a more accurate hidden solve
can worsen sampling, without proposing another activation or spectral penalty.

## Exact structure of the current ISRU ODE

For q_i(y)=y_i/sqrt(1+a*y_i²), a=1.2, define

```text
E(y) = sum_i [sqrt(1+a*y_i²)-1]/a
grad E(y) = q(y)
S=(P+P^T)/2, K=(P-P^T)/2
y'=P q(y)  =>  dE/ds=q(y)^T S q(y).
```

The skew part contributes zero instantaneously because q^T K q=0, but it
can redirect the state and thereby change later symmetric-part action.
For scheduled matrices, use the same identity with each stage's symmetric
part. This is a feature-space balance; it is not a conservation law for images.
Positive energy change is allowed and does not justify shrinking P.

The entrance-frozen diagonal also has a related identity: for fixed D_0,
y'=P D_0 y and E_0(y)=y^T D_0 y/2 imply dE_0/ds=(D_0y)^T S(D_0y).
Do not equate this quadratic with the state-updated ISRU potential.

## Numerical prediction and a controlled example

At each native midpoint stage, accumulate dt*q(mid)^T S q(mid). Compare its
sum with E(y_end)-E(y_start). Their difference is a discrete energy-balance
defect, including the quadrature error on the discrete path.

Even a pure rotation illustrates why an accurate solver and a trained feature
map can differ. For q(y)=y and P=K skew, the exact quadratic energy is conserved.
One explicit midpoint step instead gives

```text
E(y_next)-E(y) = dt^4 * ||K²y||² / 8 >= 0.
```

This exact linear example does not assert a nonnegative defect for nonlinear
ISRU. Check the actual signs and magnitudes rather than transferring the example.

## Fixed cohort and measurements

Use the six existing shared-P/PAL/ALT20kcheckpoints, one of each per task.
Capture all six local-solve entrances on the same fixed interpolation states:
first two source/target pairs from each of the eight selection banks, at
t=.1/.3/.5/.7/.9. That is80whole-model inputs per checkpoint. No fresh randomness,
trained weights, internal interval, global-time inputs or data exclusions.

1. Reproduce native four-step feature outputs in independent NumPy, including
   the exact four-stage schedule. Check analytic grad(E)=q against finite
   differences and the linear skew example before using learned states.
2. Evaluate4/8/16/32substeps and an adaptive DOP853reference on every captured
   entrance. Preserve stage boundaries. Tighten reference tolerances once to
   determine its numerical floor; an unresolved point remains visible.
3. Record feature-map error, Echange, integrated symmetric work, balance defect,
   and their refinement. Never substitute a raw Pnorm or spectrum for these.
4. Decompose the coarse-minus-reference feature difference into the normal
   direction q(y_ref) and its orthogonal complement. Retain exact zero-gradient
   cases separately. Since0<q'_i<=1, independently check the bound
   |E(y)-E(y_ref)-q(y_ref)^T(y-y_ref)| <= ||y-y_ref||²/2.

## Falsification and next decision

Hypothesis: numerical energy distortion is a meaningful component of the
learned coarse feature transformation. Reject an energy-only explanation
where energy changes stay at the reference floor but directional feature
changes are resolved. If energy distortion is resolved, report its actual
scale and alignment with the normal feature component; do not infer improved
generation from that observation alone. Mixed outcomes remain model/layer
specific. A causal feature intervention requires a separate registered design
after these measurements, rather than adding an energy penalty in advance.

This bounded diagnostic directly studies the hidden ODE. It is not a new
architectural contribution or another distribution-ranking metric. MacCPUis
sufficient; use145physical4-7only if unoccupied, never0-3or another user's job.

## Numerical settings fixed before learned-state measurement

References use rtol1e-10/atol1e-11and rtol1e-12/atol1e-13, each integrating
state and symmetric work across the four exact quarter-intervals. Qualification
is per entrance state: every endpoint component must agree within
atol1e-9/rtol1e-9, reference work within1e-8*(1+abs(work)), and tight-reference
energy closure within1e-8*(1+abs(E_start)+abs(E_end)+abs(work)). Retain both
references and every unresolved state; no tolerance changes after outcomes.
Native TorchandNumPymidpoint endpoints must agree within1e-11absolute/relative
at every4/8/16/32level before interpretation. These are numerical checks,
not architecture performance margins.

The empirical feature-resolution floor is the norm of the reference-to-reference
endpoint change plus64*machine_epsilon*(1+norm(y_ref)); the energy floor is
the reference energy change plus64*machine_epsilon*(1+abs(E_loose)+abs(E_tight)).
These are measured numerical scales, not rigorous error bounds. Report the
normal-component share of feature squared error without selecting a percentage
threshold after inspection. Also separate coarse energy gap into the change
in symmetric work and the discrete balance defect; their signs can cancel.

## Completed mathematical readiness check

[check_hidden_energy.py](../../../studies/muon_pd_20261102/target_geometry_20261116/check_hidden_energy.py)
checks grad(E), symmetric/skew work, the smooth-potential remainder bound and
the exact linear midpoint defect. The potential uses its rationalized form
sum(y²/(sqrt(1+1.2*y²)+1))to avoid cancellation near zero. Actual exit0;
maximum gradient finite-difference error1.98e-9, skew work7.50e-18.
[Full measured controls](hidden-energy-math-check.json). No learned states
or architecture outcomes are included in that check.
