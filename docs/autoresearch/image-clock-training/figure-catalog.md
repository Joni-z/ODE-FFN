# Figure catalog

Lines connect saved measurements; there are no inferential error bars. PDFs
accompany each PNG. Solver states are retained raw. Clipping to RGB display
bounds occurs only during the standard image conversion.

## FID-budgets

[Figure](results/figures/FID-budgets.png). Source: `results/comparison.csv`.
Left: all nine cold/native families at 5k/10k/20k. Right: every one of the
fourteen primary 20k endpoints, including failed frozen replacements. Lower
FID is better. All points use 1024 starts, 2048 references and Heun128.

Notice fixed P near native K1, matched-init SiLU below default SiLU64, and the
shared worsening from 10k to 20k. The caption must retain all controls and the
20k primary decision. Lines do not imply evaluation at intermediate budgets.
This supports finite contrasts, not a new ODE method or a switch to 10k.

## Training-curves

[Figure](results/figures/training-curves.png). Source: `results/training-curves.csv`.
Left/middle: all saved weighted/clean validation values for both new controls,
native K1, Swi86 and default SiLU64/131. Right: the two new controls' clipping
fractions over preceding intervals. These are not uncertainty estimates.

Fitting curves are close and continue down while generation FID can rise.
The 2.5k-to-5k clipping fractions are .1572 for fixed P and .1892 for matched
SiLU. This is compatible with different coordinate-dependent updates, without
isolating clipping as their cause. Use the exact CSV for small fitting gaps.

## Matrix-histories

[Figure](results/figures/matrix-histories.png). Source: `results/matrix-history.csv`.
Each point averages six local matrices. Left/middle: largest singular value
and singular-value CV in P coordinates. Native/fixed P are directly trained;
SiLU P is inferred as `(W-I)/c`. Right: effective W values for the constant
controls. No common constant W is implied for native adaptive clocks.

The means grow smoothly. Fixed P passes the quality criterion while its
singular values become less uniform; the ordinary coordinate history grows
more. These summaries do not bound the full FFN or sampler Jacobian. Neither
a spectral target nor a causal spectral explanation follows.

## Primary-examples

[Figure](results/figures/primary-examples.png). Source: each complete 20k bank's
first eight common Gaussian starts. All fourteen rows are shown: 112 fixed
examples. Captions must preserve row names, shared starts and the common uint8
conversion. No example is chosen by visual quality.

Native/cold fixed/matched SiLU have similar coarse layouts. Start 3 retains a
dark foreground against blue. Posthoc fixed clocks are more orange at start 6
and show stronger grid structure at start 7. These ambiguous 32-pixel examples
illustrate differences; they do not independently establish semantic quality
or the population FID ordering.

## Earlier-control-examples

[Figure](results/figures/earlier-control-examples.png). Source: the two new
controls at 5k/10k, first eight common starts: 32 examples. Together with the
primary grid, all 144 prescribed examples are displayed.

Both models change global color/layout across budgets: start 0 changes from
mostly green at 5k to mostly brown at 10k/20k. This exposes training
evolution without replacing the 20k primary endpoint. Neither the cases nor
the FID mean term identifies the cause of that change.
