# Where the clock derivatives matter

This diagnostic holds selected gate durations at their native sample-wise forward values, preserving every predicted velocity, while removing only those durations' input derivatives. It evaluates the final20000-step scalar/vectorK1 references on512 common validation pairs at five times. This is a local trained-model diagnostic, not a gate-removal training experiment. Derivative changes from different gate groups are not additive.

| Task | Model | Frozen clock group | Native Jacobian error | Anchored error |
|---|---|---|---:|---:|
| ring | scalar_k1 | ode1 | 0.312867 | 1.700429 |
| ring | scalar_k1 | ode2 | 0.312867 | 0.444050 |
| ring | vector_k1 | ode1 | 0.360002 | 2.639217 |
| ring | vector_k1 | ode2 | 0.360002 | 0.394867 |
| rotated16 | scalar_k1 | ode1 | 1.211267 | 3.455211 |
| rotated16 | scalar_k1 | ode2 | 1.211267 | 5.375553 |
| rotated16 | vector_k1 | ode1 | 1.286913 | 3.503966 |
| rotated16 | vector_k1 | ode2 | 1.286913 | 5.210063 |

The first solve's clocks dominate this derivative correction on Ring8. On rotated16D the second solve's clocks matter at least as much. Therefore a blanket deletion of all second gates is not currently supported by a task-independent redundancy claim. A retrained smaller model could recover a different function, but that remains an experiment, not a conclusion from these numbers.

For a singleK1 layer the clock term is diag(P h) J_s. Its contribution to the full output derivative is multiplied by upstream and downstream Jacobians, including the intermediate activation, normalization and time conditioning. Gate standard deviation or P norm alone cannot determine that contribution. This explains why attributing the difference to a single hidden matrix's eigenvalues would be incomplete.

The screen rejecting SiLU deletion also matters here: removing a diagonal derivative factor looked attractive mathematically but removed a useful fixed source of nonlinearity for Ring8 learning. Any subsequent cheaper clock design must retain the successful nonlinear composition initially and examine what local adaptation it removes. Do not combine clock pruning with a new activation, spectral constraint or loss term.

The current full16D target still has only a2D state-dependent covariance subspace. A future rank-capacity claim needs a task with genuinely higher-rank state-dependent posterior covariance; Gaussian nuisance directions alone do not supply that evidence. No such new task or architecture is claimed to have been tested yet.
