# Execution notes

Machine verified: 145 / hw3242-gpu1. All eight GPUs are occupied, utilization 63–88% at preflight. Existing CPU gate-study checkpoints are under /data2/zz5070/ODE_FFN/gates_20261021_cpu/output. No other user's GPU tasks will be interrupted for this small diagnostic.

The initial status lookup used control-status.json; the completed study actually uses status.json. This path lookup did not affect any scientific output.
