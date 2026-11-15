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
control, not a prerequisite for answering the professor. The subsequently
proposed ISRU-midpoint pair is no longer the default selected direction:
the matched response evidence does not justify privileging ISRU. Any new
operator needs a measured limitation and appropriate ordinary controls.
Synthetic exponential
accuracy does not establish image accuracy or prove a benefit from feedback.
No extra architecture or image training was added.
Report1/2 preservation covers all96 tracked files against4c336ef.

Use full architecture names when discussing order. The professor's original
layer uses a learned scalar clock s(h) and A=s(h)P. The recent CelebA K1/K8
models are widened ISRU-D with A=P diag(1/sqrt(1+1.2*h^2)); recent synthetic
models include both ISRU-D and Tanh-D at K1/K8. All inherit the shared-P Taylor
construction, but the latter are not unchanged copies of the original scalar
clock layer. D is computed once per solve and reused across terms; K1 equals
h+P ISRU(h), whereas higher terms do not repeatedly apply ISRU to the new term.

The requested concrete CelebA candidate is now proposed as Tanh-D K8:
the same widened768->1200 topology, two frozen-D Taylor solves around SiLU,
and only d(h) changed to the verified scaled tanh ratio. At the registered
synthetic20k endpoint, tanh-K8 beats ISRU-K8 on both tasks/metrics; earlier
checkpoints are mixed. This motivates one transfer control, not an image
superiority or novelty claim. Keep AdamW and the existing image recipe,
reuse the native/wide-ISRU baselines, and propose one100epoch seed0 trajectory.
No implementation or job is launched in this architecture-selection discussion.
