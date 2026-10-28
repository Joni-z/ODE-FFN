# Figure catalog

All three figures exist as PNG and PDF. No uncertainty bars are shown: the
comparisons use fixed trained models, sources and numerical meshes. The first
eight examples were fixed by index before finer-grid outcomes.

## FID-resolution

Purpose: test whether the original ranking reverses and quantify gap changes.
Data: all 12 rows of comparison.csv and all ODE-versus-Swi contrasts. Left:
raw FIDs; right: signed pairwise differences with a zero reference line.
Observation: rankings persist; all ODE-versus-Swi differences stay negative,
but the Swi129 gaps shrink. Caption must give 1,024/2,048 sample counts,
state that lower is better and preserve all arms. Do not describe a continuous
limit, significant effect or selected best solver.

## sample-resolution

Purpose: separate endpoint sensitivity from FID. Data: every raw-pixel RMS
comparison and all raw out-of-range fractions. RMS uses a log vertical axis;
its three horizontal categories are distinct pair comparisons, not a time
trajectory. Observation: 64-to-128 changes are smaller than 32-to-64 for every
arm; ODE has larger out-of-range fractions here. These do not prove global
stability, convergence order or a local P cause. The finest grid is not truth.

## generation-resolution

Purpose: show all three meshes for the same fixed examples, without appearance
selection. Data: first eight starts per model, arranged in two rows of four.
Observation: coarse layouts persist within each model across grids and all
outputs remain blurred. The recurring pale/white and dark regions provide
concrete comparison points. Display clamp only; raw sampler states remain
unclipped. Do not label ambiguous objects, claim mode coverage or use these
examples to explain feature-mean changes causally.
