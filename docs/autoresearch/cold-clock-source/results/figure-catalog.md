# Figure catalog

All seven PNGs were inspected after the actual plotting exit0; one-page PDF versions accompany them.

## primary-FID

- File: [PNG](figures/primary-FID.png), [PDF](figures/primary-FID.pdf).

- Purpose: Compare all 17 primary endpoints.

- Data: primary-20k.csv.

- Observation: Time172.04 is the lowest observed score but misses the 5%threshold against native173.44.

- Caption requirements: State20k,1,024generated/2,048reference, and lower-is-better.

- Interpretation: Retain the small change and stop the registered setting; do not compare only with default Swi86.

## budget-comparison

- File: [PNG](figures/budget-comparison.png), [PDF](figures/budget-comparison.pdf).

- Purpose: Check whether fitting and generation move together.

- Data: comparison.csv.

- Observation: Both new arms improve FID at 10k and worsen at 20k while MSE decreases; controls show a similar pattern.

- Caption requirements: Three fixed checkpoints, shared starts;20k remains primary.

- Interpretation: Training loss is not a sufficient quality proxy; no 10k substitution.

## training-curves

- File: [PNG](figures/training-curves.png), [PDF](figures/training-curves.pdf).

- Purpose: Inspect optimization and clipping history.

- Data: training-curves.csv.

- Observation: New-arm validation curves nearly overlap and late clipping is rare.

- Caption requirements: Original logging intervals; clipping is windowed.

- Interpretation: No large fitting advantage or late clipping crisis is demonstrated.

## P-trajectories

- File: [PNG](figures/P-trajectories.png), [PDF](figures/P-trajectories.pdf).

- Purpose: Compare physical P development.

- Data: P-history.csv.

- Observation: Mean largest singular values increase from 1.1202 to 1.3984–1.4719 across four families.

- Caption requirements: Means over six local matrices, not a predictor Jacobian.

- Interpretation: Do not infer a new spectral regime or causal stability mechanism.

## clock-profiles

- File: [PNG](figures/clock-profiles.png), [PDF](figures/clock-profiles.pdf).

- Purpose: Verify that clock inputs lead to learned nonconstant profiles.

- Data: clock-profiles.csv.

- Observation: All time-only means increase over the five displayed times; two late clocks approach1.0. Feature-clock late means differ.

- Caption requirements: 20k; bands are within-panel SD, not confidence intervals; increasing FM time approaches clean data.

- Interpretation: Clock learning is active, but does not establish the cause of a small FID change.

## P-eigenvalues

- File: [PNG](figures/P-eigenvalues.png), [PDF](figures/P-eigenvalues.pdf).

- Purpose: Keep individual complex eigenvalues visible.

- Data: P-eigenvalues.csv.

- Observation: Initial and 20k clouds overlap, with both positive and negative real parts.

- Caption requirements: All six matrices per family; gray initial and colored20k; no stability certificate.

- Interpretation: A sign or spectral cap is not justified by this descriptive plot.

## fixed-start-cases

- File: [PNG](figures/fixed-start-cases.png), [PDF](figures/fixed-start-cases.pdf).

- Purpose: Show source-paired qualitative examples.

- Data: private case-images.npz; original sample banks.

- Observation: Broad layouts remain similar across rows; feature/time differences are subtle at 32×32.

- Caption requirements: Original starts 0–7 and eight named models; standard clipping/rounding.

- Interpretation: No clear semantic improvement or diversity conclusion follows from these fixed examples.
