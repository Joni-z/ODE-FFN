# Cases that change the decision

## Exact integration does not resolve the controlled gap

On sign-code, field MSE is 0.255043 for exact rotation, 0.258663 for Euler,
and 0.244833 for normalized Euler. Swi129 reaches 0.0409672 and the
initial-function-matched ordinary SiLU reaches 0.0700284. The exact flow,
its Euler discretization and its conserving control all remain well behind
ordinary controls at the registered endpoint. This does not isolate one
cause, but it gives no reason to promote this solver/constraint setting.

## Sampling distance alone would give a misleading decision

On Product16, rotation SW2 is 0.0979024 versus Swi129 0.102716. However,
rotation field MSE is 0.294785 versus 0.0772904, and Jacobian error is
54.3291 versus 20.7857. A small sampling-distance advantage does not satisfy
the predeclared conjunction. The older freeISRU control also has much lower
field MSE, 0.0371592. This is why all strong controls remain in the table.

## The learned clocks are active; conserved amplitude is not contractivity

At 5,000 updates, the six rotation P matrices have largest singular values
in ranges 0.262–0.615 (Product16), 0.333–0.757 (sign-code), and 0.367–0.526
(half-mixture). Across-example clock standard deviations, averaged over the
six layers on fixed visited inputs, are 0.165, 0.227 and 0.236. Thus the
candidate did not remain at its zero-frequency initial function.

The largest observed complete local derivative gain on the stored Product16
anchors is 3.700 despite exact preservation of each pair's amplitude.
The explicit derivative includes state-dependent clock feedback. This is
one actual-input witness, not a supremum over inputs or a whole-network
stability statement. The P/clock CSVs preserve all layers and checkpoints.

## Decision

Stop this fixed-plane, invariant-clock rotation setting. Do not add a learned
basis, more steps, a new optimizer or another initializer as a rescue. The
result does not reject every constraint on P, every ODE block or the earlier
image K1 result. It rejects promotion of this particular candidate under its
registered quality and cost requirements.
