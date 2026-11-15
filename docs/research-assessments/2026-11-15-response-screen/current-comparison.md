# Partial response-screen comparison

This is a status audit, not the completed selection analysis. At this check,
12 of36 pipelines have completed training and selection evaluation. All shown
numbers use the20k checkpoint, new CPU cohort, eight common4096sample banks,
256 directions and Heun128. Lower is better. Other arms and the automatic
confirmation stage are still running/pending. Do not fill gaps with older
CUDA scores or the earlier single-bank metric.

## Ring8 completed comparisons

| Response | K1 field MSE | K8 field MSE | K1 pooled SW2 | K8 pooled SW2 |
|---|---:|---:|---:|---:|
| isru | 0.010467 | 0.011250 | 0.115780 | 0.100238 |
| tanh | 0.009478 | 0.011456 | 0.113893 | 0.105239 |
| arctan | 0.010302 | 0.011395 | 0.097694 | 0.090299 |
| softsign | 0.010443 | 0.011340 | 0.090968 | 0.069559 |
| erf | 0.009703 | 0.011352 | 0.122258 | 0.106584 |
| silu | 0.013458 | pending | 0.162098 | pending |
| gelu | 0.012422 | pending | 0.152849 | pending |
| linear | pending | pending | pending | pending |

The same-host SwiGLU64/Muon reference is still pending. The Product16 response
and SwiGLU192/AdamW comparisons are also pending. Thus an all-model ranking
against SwiGLU cannot yet be reported.

Softsign-K8 is the current Ring8 sampling lead, while tanh-K1 has the lowest
field MSE among the audited cases. Softsign-K8's pooled SW2 is0.06956 versus
ISRU-K8 0.10024 and tanh-K8 0.10524; its field MSE is0.01134 versus0.01125
for ISRU-K8. The ranking differs by metric. Keep it as a candidate to watch;
no CelebA response is selected before the ordinary controls, Product16 and
the registered disjoint confirmation are available.

All12 completed arrays were independently recomputed in NumPy: maximum
metric difference2.8e-17. Checkpoint/config/array hashes and native restoration
identities passed. The ordinary controls were registered after focal cases
in each dataset, explaining why they are not yet in the completed table.
The existing queue continues; no new experiment or image allocation was added.

[Exact audited rows](partial-status-audit.json), [queue snapshot](status-snapshot.json),
[protocol](protocol.md). Finalized Reports1/2 and the image baseline are preserved.
