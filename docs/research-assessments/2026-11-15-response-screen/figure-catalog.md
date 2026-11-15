# Figure catalog

## 1. Complete response screen

File: [response-comparison.png](figures/response-comparison.png).
Purpose: compare all eight response families and both Taylor orders at20k.
The four panels show field MSE and pooled SW2 for Ring8 and Product16 on
selection arrays. Lower is better. Ordinary controls appear in the companion
table and cost figure; response-only comparisons share model size and optimizer.

Observation: Softsign has the lowest Ring8 sampling error, while its Product16
field fit fails the eligibility gate. Increasing order changes rankings rather
than providing a uniform improvement. The SiLU/GELU configurations are weak
Product16 alternatives at this budget.

Decision: do not choose from Ring8 SW2 alone. Use the frozen cross-task rule
and independent confirmation before promoting a response.

## 2. Sampling quality against measured CPU cost

File: [quality-cost.png](figures/quality-cost.png).
Purpose: put selection pooled SW2 beside warmed whole-network forward time.
Both tasks use the same M1 CPU, one thread, batch512; medians come from three
20-forward repeats. K1/K8 are connected within a response. Ordinary SwiGLU and
Free ISRU references are included. Horizontal position is measured CPU cost,
not FLOPs or predicted image-GPU time.

Observation: Product16 ordinary controls achieve better sampling with less
measured work than the ODE responses. Arctan's pointwise implementation is
expensive here even atK1; high order adds further work.

Decision: a JiT GPU throughput check precedes a formal arctan image run.

## 3. Selection versus independent confirmation

File: [confirmation-comparison.png](figures/confirmation-comparison.png).
Purpose: compare the same final weights under the two prebound evaluation
streams. Gray circles are selection; teal diamonds are confirmation.
Columns are Ring8/Product16; rows are field MSE/pooled SW2. Ordinary references
and both orders of the frozen ISRU/arctan shortlist are shown.

Observation: Arctan beats the Ring8 ordinary references on confirmation, but
its K1/K8 sampling order reverses from selection. Product16 SwiGLU remains
the strongest sampling configuration; Free ISRU also beats all confirmed
ODE variants. Vertical changes reflect different evaluation arrays, not
continued training or architectural edits.

Decision: prioritize arctan-K1 over arctan-K8 for a bounded readiness check.
No cross-task superiority or blanket benefit of high-order approximation is
established. Per-bank paired comparisons are in the statistical appendix.

All three figures were inspected as rendered images. They use the measured
CSV/JSON values without smoothing or invented error bars. Captions must retain
the evaluation-stream distinction and the CPU timing unit.
