# Figure interpretation

1. Comparison: all 12 methods on all three laws, separately normalized by the best registered practical control for each metric. Inspect the fixed candidate across every axis; a favorable point alone cannot satisfy the conjunction. Historical rows are reused, with no claim of new training.
2. Field curves: all seven fresh arms at the four fixed checkpoints. Inspect convergence at the fixed endpoint; no favorable-checkpoint selection.
3. P and feedback: six-layer mean/layer range of the largest singular value, and mean complete local derivative gain on fixed actual-input anchors. The gain includes state-dependent clock feedback. A small P does not certify a small full derivative. All exact layer/clock summaries are in the CSVs; no confidence bands are shown.
