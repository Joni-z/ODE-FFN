# Execution notes

Machine verified: 145 / hw3242-gpu1. All eight GPUs are occupied, utilization 63–88% at preflight. Existing CPU gate-study checkpoints are under /data2/zz5070/ODE_FFN/gates_20261021_cpu/output. No other user's GPU tasks will be interrupted for this small diagnostic.

The initial status lookup used control-status.json; the completed study actually uses status.json. This path lookup did not affect any scientific output.

The five new tests passed locally and on 145. Preflight completed in 0.429 seconds. Full diagnostic PID 663299 completed at 2026-10-21T04:55:40Z after 8.76 seconds. Source commit fe0bd5d; the runtime is frozen separately from later analysis commits. Monitoring observed ~98.6% of one CPU core and 519.7 MiB peak RSS at the five-second sample. All GPU devices were hidden from the process.

Collected archive SHA256: bc0d2264083b40830a98b175ac2f869434cf5ebc7c6c4ea40abd04dffb97bfd7. All 100 panels and 32,000 saved rows were audited locally. The derivative gain identity, all strata, common-state pairing and zero-correction controls were rechecked. Largest finite-difference relative error: 1.0763909770872594e-6.

Gate contributions help local derivatives in both Gaussian and Ring8, including low-entropy contracting states. This does not isolate the proposed multimodal-boundary explanation. Ring8 SwiGLU has lower velocity and derivative errors on common interpolation states. One late ODE trajectory point dominates its local error average; all tails remain in the evidence.

The experimental-design skill was used to freeze the comparison and interpretations before execution; daily-coding was used for implementation checks. Workflow attribution is recorded in the preceding [reassessment notes](../research-2026-10-21-reassessment/notes.md), separate from scientific claim support. Submitted Report 1 was not edited.
