# Conditional input uncertainty

Fixed checkpoints,32768paired source/target inputs per task, five repeated times.
The eight banks are the paired resampling units. Registered percentile bootstrap:
10000draws,RNG0,95% and97.5%two-contrast intervals. Raw primary outputs remain
in ring-summary.json/product-summary.json. An additional exact sensitivity check
enumerates6435sorted resample multisets with integer multiplicities summing to
8^8ordered resamples. Its intervals are reported alongside the original draws.
No training-repeat uncertainty or across-model significance is inferred. The
small number of banks limits tail resolution even when every source is retained.
No normality assumption, t-test or p-value is used. Stage-order weights and the
generation comparison are unchanged from the earlier registered experiment.
