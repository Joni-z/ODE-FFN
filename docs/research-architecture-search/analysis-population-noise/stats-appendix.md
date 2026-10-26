# Units, pairing and measurement scope

Training follows the user's fixed seed0 protocol. The six arms of each task see exactly the same sampled target/source/time bytes, verified at every checkpoint and by independent full-stream regeneration. These are paired descriptive comparisons, without between-training significance tests or confidence intervals. Checkpoints, coordinates and FM times are not independent training replications.

The primary outcome is final random-time validation population-oracle MSE. Ratios use full stored values before rounding. Both SwiGLU sizes are retained, including when their relative ordering changes. Raw pair and mean training losses have different irreducible floors and are not compared as performance outcomes. Clipping fractions refer to the final500 updates and are observational.

Conditional-mean equivalence applies to the intended continuous analytic population; implementation uses tested double-precision target/oracle calculations and the established float32 input/label contract. The risk identity concerns gradients before clipping. Adam/clipping responses, input sampling and finite training budgets remain. A change versus older cached-bank runs additionally changes the training input law and random-draw sequence, so it is not a noise-only contrast.

Native audit covers source/data hashes,72 checkpoint identities,60 analytic arrays, initial pair/mean tensors, all checkpoint RNG/fingerprint values and10000 regenerated minibatches. Mac replay separately covers12 final checkpoint fields/60 arrays. No sampling-trajectory replay or fresh holdout is claimed. Exact summaries and hashes accompany these artifacts.
