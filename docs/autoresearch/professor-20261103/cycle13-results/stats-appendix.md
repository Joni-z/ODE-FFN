# Statistical scope and complete paired effects

The unit is one registered configuration/task endpoint. Settings, checkpoints,
layers and sampling projections are not independent replications. This is a
fixed seed0 cohort; no p-values, cross-setting error bars or architecture-level
confidence intervals are computed. Exact finite-cohort paired effects answer
the registered questions. Validation selects; test sampling is read separately.

| Task | Optimizer | Gain vs constant | Gain vs free ISRU | SW2 change vs constant | SW2 change vs free | Continue |
|---|---|---:|---:|---:|---:|---|
| gmm_ring8_2d | adamw | 32.1802% | 2.2480% | -26.6066% | -15.8481% | False |
| gmm_ring8_2d | muon | -4.8314% | 43.0917% | -2.4533% | 13.4754% | False |
| rotated_product_bimodal_16d | adamw | 78.7041% | -4.5777% | -15.5929% | 1.6840% | False |
| rotated_product_bimodal_16d | muon | -48.3781% | -2.6033% | 54.5728% | 29.3598% | False |

Positive field gain is better; negative SW2 change is better. All four
contrasts remain. Thresholds were registered before outcomes: >=5% versus both
constant/free controls plus lower MSE than the full ordinary frontier. Generic
ledger keeps (two AdamW parent gains) are not advancement or method acceptance.
No new model is selected by test SW2.

Native field replay maximum error5.68e-10; NumPy SW2 versus native8.59e-9;
maximum generated replay difference0; all10archived numerical state/sample
anchors exact. Checkpoint/config/cache identities are enforced in the native
audit. Source bindings record raw receipt and historical CSV hashes.

Resource monitor:145 completed its queue in617.95s; Lenovo832.11s. Median
GPU0 utilization in recorded two-worker snapshots was89%/90%, including
startup/dispatch snapshots with zero utilization. These are monitored
snapshots, not continuous guaranteed utilization. Native process wall sums
are1161.35s/1568.45s and overlap across workers; not GPU hours or isolated
latency. Failed cells0. Both queues ended without pending or running cells.
