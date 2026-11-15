# Partial response-screen comparison

19of36 pipelines have completed20k training and selection evaluation. Ring8
now has all16 response/order cases and both ordinary controls. Product16 is
in progress; the final cross-task shortlist and confirmation are unfinished.
The existing queues continue. No new experiment or image allocation was added.

All numbers below use this Mac mini CPU cohort, the same eight4096sample
selection banks,256directions and Heun128. Lower is better.

| Response | K1 field MSE | K8 field MSE | K1 pooled SW2 | K8 pooled SW2 |
|---|---:|---:|---:|---:|
| isru | 0.010467 | 0.011250 | 0.115780 | 0.100238 |
| tanh | 0.009478 | 0.011456 | 0.113893 | 0.105239 |
| arctan | 0.010302 | 0.011395 | 0.097694 | 0.090299 |
| softsign | 0.010443 | 0.011340 | 0.090968 | 0.069559 |
| erf | 0.009703 | 0.011352 | 0.122258 | 0.106584 |
| silu | 0.013458 | 0.014150 | 0.162098 | 0.151971 |
| gelu | 0.012422 | 0.014315 | 0.152849 | 0.130494 |
| linear | 0.015525 | 0.018280 | 0.121731 | 0.123481 |

| Ordinary Ring8 reference | Field MSE | Pooled SW2 |
|---|---:|---:|
| SwiGLU64 / Muon.02 | 0.012929 | 0.118231 |
| Free ISRU / AdamW.0003 | 0.011497 | 0.112653 |

Softsign-K8 remains the Ring8 sampling lead. Its pooled SW2 is0.06956 versus
the current SwiGLU64 reference0.11823. Its field MSE is0.01134 versus0.01293.
Tanh-K1 has the lowest response-family field MSE0.009478. The selected ordinary
optimizer settings differ and are labeled; this table does not isolate an
optimizer or ODE-specific cause. The cross-task eligibility rule still awaits
Product16, and disjoint confirmation has not started.

We have previous synthetic SwiGLU baselines and a qualified native CelebA
baseline. This CPU SwiGLU run is the same-host comparison for the new screen;
it does not replace or invalidate those earlier results. The current partial
table must not mix older banks/backend values into the new cohort.

All19 completed20k arrays were independently recomputed in NumPy; maximum
metric discrepancy2.8e-17. Checkpoint/config/asset hashes and native saved
sampling restoration passed. Finalized Reports1/2 and the image baseline remain
preserved.

[Exact audited rows](partial-status-audit.json),[queue snapshot](status-snapshot.json),
[protocol](protocol.md). No CelebA candidate is selected yet.
