# Statistical scope and numerical verification

- Comparison unit: a named final20k checkpoint under two fixed transfer functions.
- Cohort:24 checkpoints, four controller/output corners by two tasks and three
  existing optimizer/rate settings. Model seed 0; no new training or resampling.
- All 48actions use the unchanged validation oracle and separate test sampling:
  4096 samples,256 fixed projection directions,Heun32. Source/config/manifest and
  every compressed array are hash-bound. Targets are identical within each task.
- Report each paired absolute score and ratio. Native/swapped field losses and
  prediction-displacement norms also satisfy the triangle bounds for the shared
  validation bank. No grouping of checkpoints, rates or tasks is treated as
  independent replication.
- This is an enumerated intervention cohort, not a random sample of methods.
  Population-level hypothesis tests, confidence intervals, mean±SD summaries and
  multiplicity-corrected p-values are not estimated from these unlike settings.
  Figures have no inferential error bars. Full per-case effects are the statistics.
- Missing outcomes:0. All 48sampling metrics are finite. The producer preserves
  nonfinite swaps with null scores if they occur; none were dropped or clamped.
- Independent NumPy float64 projection/sort recomputation covers all 48saved
  generated/target arrays with unchanged directions. Maximum absolute difference
  from native float32 sliced-W2 is 4.175740730971711e-8.
- All 24native generated-array anchors and24 round-trip probe comparisons are
  exact. All 24 full parameter/buffer checks pass. Native field versus source CSV
  discrepancy is at most 1.3877787807814457e-17, and native test versus archived
  independent score discrepancy at most 8.384914373937846e-9.
- Test scores and example plots do not select a rate, checkpoint or method.
  Frozen transformations are not trained-model comparisons or a validation of
  an architectural improvement. Training TSV remains 132 unique rows.
