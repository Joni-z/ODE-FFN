# Figure purposes and interpretation

All seven PNGs were visually reviewed. Source values are in [comparison.csv](results/comparison.csv), [curves.csv](results/curves.csv), [time-panels.csv](results/time-panels.csv) and [P-history.csv](results/P-history.csv). There are no statistical error bars: these are recorded trajectories and fixed banks, not independent trained replications.

| Figure | Purpose and observation | Interpretation limit |
|---|---|---|
| [Final comparison](results/final-comparison.png) | Show every 20k model on fitting and FID; K1 has lowest FID, Swi129 lowest MSE. | Caption must retain the fixed endpoint, sample counts and cost mismatches; neither axis substitutes for the other. |
| [Regression curves](results/continuation-curves.png) | Retain every 1k checkpoint; both losses improve from 10k to 20k. | Connecting recorded points does not establish convergence or monotonic improvement at every update. |
| [FID trajectories](results/FID-trajectories.png) | Show all 5k/10k/20k measurements; every 20k FID exceeds its 10k value. | Only three budgets were measured; lines do not locate an optimum, and 10k is not substituted for the primary 20k result. |
| [Time panels](results/time-panels.png) | Compare all five fixed times at both new endpoints; late paired error improves despite FID worsening. | Panels contain 64 paired examples each. Log-axis ranges differ across the two subplots; use the CSV for exact cross-budget comparisons. These are not generated-path errors. |
| [P trajectories](results/P-trajectories.png) | Show every layer at all four saved snapshots; final-block ode2 grows most. | Largest singular values of hidden P are not complete-predictor stability bounds or a causal FID explanation. |
| [10k examples](results/generation-10000.png) | Display the first eight common Gaussian starts for all seven arms. | No case is selected by quality; coarse structures do not establish semantic accuracy. |
| [20k examples](results/generation-20000.png) | Reuse the exact rows/columns from 10k; column 4 becomes bluer and column 8 less green across families. | Display conversion clips. These visible changes neither explain FID nor demonstrate a semantic breakthrough. |

Each caption should state the actual budget, model identities and fixed evaluation bank. FID figures use 1,024 generated/2,048 reference images and Heun128. P lines are layer trajectories, not confidence intervals. The [review receipt](results/visual-review.json) binds each inspected PNG to its hash; PDF plot companions contain the same data.
