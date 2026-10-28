# Figure purposes and interpretation

All figures are generated from audited tables with the source fixed before
new FID extraction. Each has a PNG and standalone one-page PDF. No error bars
are implied. All five PNGs were visually inspected.

## primary-FID
Purpose: compare every retained 20k endpoint, including stronger ordinary and
frozen controls. Source: primary- 20k.csv. Caption:15 models/interventions,
1024 common starts, 2048 references, lower FID is better. Notice 185.85 versus
195.31 default, 173.44 K1 and 178.06 matched SiLU. This keeps initialization as a
factor but rejects the claim that it removes K1's remaining lead. Check all
15 labels and numerical endpoints; do not interpret the ordering as significance.

## budget-comparison
Purpose: expose fitting and generation orderings at all three fixed budgets.
Source: comparison.csv. Notice new FID 204.42→171.08→185.85 while both losses
fall. This rules out using the lower late regression loss as evidence of
better generation. The 20k endpoint remains primary; connecting lines show
stored checkpoints, not a densely measured continuous curve.

## training-curves
Purpose: compare learning and clipping history rather than only final scores.
Source: training-curves.csv. Notice similar decreasing losses and different
early clipping windows; new clipping falls to 0.003 in its final window. These
differences motivate checking optimizer coordinates but do not establish
a causal explanation. Retained and new runs have different logging density;
lines interpolate their stored records. Clipping fractions are window summaries.

## branch-and-matrix-histories
Purpose: check whether the added SwiGLU capacity actually moves during training.
Sources: matrix-history.csv and branch-history.csv. Notice nonzero value and
extra output norms, with smooth matrix growth through the late FID increase.
This rejects a frozen-extra-capacity explanation. It does not justify an
eigenvalue cap, a whole-model Jacobian claim or a claim that any branch causes
the FID gap. Left panels average three native matrices of each named type.

## fixed-start-cases
Purpose: show paired examples that can be inspected alongside distributional
metrics. Source: first eight original starts at 20k for the six preregistered
models. Notice the changed blue region at start 1 and yellow/green shift at
start 6 for new Swi86. These changes show why a few images do not yield a
uniform visual ranking. Keep all 48 cells; no object-recognition, diversity or
class-fidelity conclusion is inferred from these coarse examples.
