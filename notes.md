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

All four formal pipelines completed with exit0 and released their cards.
Ring8 ISRU K1->K8 pooled SW2:0.13261->0.16779; Product16:0.07206->0.07321.
Product16 tanh improves0.07006->0.06427, seven of eight paired banks better.
Both K8 responses still benefit strongly from C8 projection on Ring8.
Higher terms are active: deleting them from K8-trained weights is damaging.
The inspected K8 exponential error is <=1.60e-4. These facts distinguish
learned dependence on terms from an advantage over a separately trained K1.

Published evidence lives in docs/research-assessments/2026-11-15-order-response/.
The initial follow-up recommendation was paired C8 training augmentation on
ISRU-D and ordinary SwiGLU. In the subsequent planning discussion, prioritize
the existing CelebA checkpoint diagnosis to keep the main question tied to
the measured image-model gap. C8 augmentation remains an optional small
control, not a prerequisite for answering the professor. The sole retained
image operator pair is frozen-D K2 versus refreshed ISRU midpoint, subject
to review of the diagnosis and a concrete run decision. Synthetic exponential
accuracy does not establish image accuracy or prove a benefit from feedback.
No extra architecture or image training was added.
Report1/2 preservation covers all96 tracked files against4c336ef.
