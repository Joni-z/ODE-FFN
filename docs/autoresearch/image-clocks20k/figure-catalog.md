# Figure purposes and interpretation

All three figures were visually reviewed; the [review receipt](results/visual-review.json) binds the PNG hashes. PDF companions show the same data. No statistical error bars are plotted: these are fixed checkpoints and finite interventions, not independent training repetitions.

| Figure | Purpose and observation | Caption requirements and interpretation limits |
|---|---|---|
| [All endpoints](results/figures/fixed-comparison.png) | Compare all nine fitting/FID endpoints; both constants lose the ordinary-control FID lead despite much smaller aggregate loss changes. | State 20k weights, frozen tensors, all seven reused controls, 1,024 generated/2,048 reference images and Heun128. Parameter/work differences do not disappear under intervention. |
| [Time panels](results/figures/time-panels.png) | Show every fixed time for native/constant pairs and each arm's declared ordinary references; late paired error rises, more strongly for K2. | Both axes share a log y-scale; each panel has 64 paired examples. These are not generated-trajectory errors or independent model repetitions. All 45 scores are retained in CSV. |
| [Fixed cases](results/figures/generation-cases.png) | Display the same first eight starts across all nine endpoints. K1 constant has stronger horizontal banding; K2 constant shows a grid-like foreground in column 4. | No quality-selected cases. Conversion clips; coarse structures neither certify semantics nor causally decompose FID. |

The comparison is in [comparison.csv](results/comparison.csv); native-to-constant deltas and strict-order versus margin decisions are in [contrasts.csv](results/contrasts.csv). Negative retained-gap fractions indicate a reversed lead, not a causal share of the original improvement. Calibration SDs describe token variation and must not be used as trained-model uncertainty.
