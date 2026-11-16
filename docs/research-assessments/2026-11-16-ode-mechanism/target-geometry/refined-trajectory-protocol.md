# Refined test of the error-transport account

## Question and unchanged population

Does the oracle-path linear sensitivity predict the trained model's endpoint
displacement? Keep all twelve20kcheckpoints, their original hidden resolution,
and the same256selection sources:32from each of eight banks. The completed
native-only qualification passes every model, ten at4096and two at8192.
This qualifies native endpoint precision, not the sensitivity approximation.

The original1024-cap pilot remains a completed unresolved result. This is a
separately registered precision extension, with no retraining or new inputs.

## Computation and numerical decision

Reuse and bind the original1024oracle, tangent and four-blend arrays. Compute
all four blends(lambda=.125,.25,.5,1) and the oracle/sensitivity system at
2048and4096steps in FP64. If the4096versus2048gate fails, compute8192once.
Keep every level and every source, including any unresolved8192row.

Use the original componentwise endpoint/oracle/tangent tolerance:
atol=1e-4,rtol=1e-3. The full-scale explained fraction must change by at most
.01between the final two levels. Every component and source must pass; a
global RMSalone cannot qualify a case. Passing2048versus1024does not skip4096.

The four-blend packed execution was checked against all twelve separate
native Heun128trajectories on all256sources; every endpoint and recorded
state matched exactly. At every available new resolution, lambda=1must also
match the independent native-only endpoint and nine recorded states exactly.
The oracle/tangent system uses the unchanged native Heun solver, exact oracle
JVPand original error forcing. Preserve all weights and runtime buffers.

## Scientific decision

Use the previously registered quantity:

```text
delta = model_endpoint - oracle_endpoint
F = 1 - mean((delta-tangent_endpoint)^2) / mean(delta^2)
```

F>0means the linear prediction improves upon predicting no displacement.
F<=0rejects that benefit at the trained error scale for that model. Interpret
only numerically qualified rows. Small-lambda results test local expansion
behavior; they cannot replace the lambda=1decision.

Report every model/bank, full-scale F, all four error multipliers, the change
from1024and exact input-bank bootstrap intervals. Jointly resample residual
and displacement energy across the eight banks; use the existing exact
8^8enumeration and twelve-contrast adjustment. This is fixed-weight input
uncertainty. The native large-bank SW2scores remain the generation evidence.
Paired displacement is not a distribution distance and cannot establish a
sampling explanation by itself. No new architecture follows automatically.

## Execution

Warm MacCPUenvironment, two one-thread workers, no CUDAorMPS. Actual native
stage calls, source/checkpoint hashes, new field-call counts, per-level elapsed
time and process exits are recorded. Use the existing monitored precision
queue and independent collector. Before formal dispatch, freeze source and
run a fresh documented actual-model witness in both tasks, then its NumPy
verifier. Check145physical4-7before anyGPUuse; never touch0-3or occupied cards.
