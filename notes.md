# Order-by-response evidence notes

Previous completed wave: docs/research-assessments/2026-11-15-professor-execution/.
ISRU/tanh K1 results alone do not determine their interaction with K8.
C8 projection is a field diagnostic, not the new architecture or a training change.

145 GPUs4-7 are idle RTX6000Ada. /data2 free46GB; warm py312/Torch2.8.0+cu128.
Reuse data root /data2/zz5070/ODE_FFN/professor-followup-20261115/data.
Ring8 K1 ISRU: previousgpu6; tanh: previousgpu4.
Product16 K1 ISRU: originalcycle13 on145; tanh: previousgpu5.
The native source/optimizer/evaluator remain pinned. Report1/2 are unchanged.

Formal four-run launch uses9f72bd3 in the separate order-response worktree.
GPU4/5 train Ring8 ISRU/tanh K8; GPU6/7 train Product16 ISRU/tanh K8.
The existing persistent queue monitors record progress every30seconds and
automatically run common-bank evaluation followed by frozen-order diagnostics.
Readiness passed after explicit-path imports removed a historical module-name
collision. Full diagonal Jacobian coverage is dim2; dim16 has parameter/RNG,
finite-gradient, checkpoint, and continuation checks. No formal retries.
