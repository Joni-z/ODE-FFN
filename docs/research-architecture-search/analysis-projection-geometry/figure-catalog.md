# Figure interpretation

## conditioning-and-residual-alignment.pdf / .png

Purpose: check whether input anisotropy and its error overlap distinguish
ODE from the practical reference. Source: verified geometry summaries and
raw G/gradient arrays from the four archived arms at5000 updates.
Notice: large conditions occur in both families; standardizing feature scale
makes the lowest-quarter gradient shares comparable in magnitude. Median
points and layer-range whiskers are descriptive. The25% line marks the
selected fraction of dimensions, not a statistical reference distribution.
Decision: a P-only whitening proposal is not selected by this evidence.
Do not infer that every feature direction or every architecture behaves alike.

## participation-trajectory.pdf / .png

Purpose: distinguish a structural initialization issue from features that
change during fitting. Source: uncentered uniform-time moments at0,1000,5000.
Notice: product16 participation falls in both the original ODE and small
SwiGLU. Lines are medians, bands are layer ranges. This changes the question
from static input rank to the interaction of learned features and operator
responses. It does not justify enforcing a fixed high participation rank.

Both plots use exact recorded checkpoints. Plot source and output digests
are recorded in provenance.json; the numeric table and CSV permit replotting.
