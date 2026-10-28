# Statistics and comparison units

The primary question is time-only versus the best retained20k control,
including the paired feature arm. Native K1 is that control at 173.4369143077.
Time-only reaches172.0421340118; its FID ratio is 0.991958. The threshold is
164.7650685923. Both weighted/clean MSE ratios are below the 1.05 guards.
The exact fixed decision is in[decision.json](decision.json).

The unit is a fixed trained-model endpoint. Checkpoints share a trajectory;
image starts and panels are not independent trained-model replicates. This
registered screen uses descriptive differences and ratios, not inferential
significance claims. No p-values, model confidence intervals, or invented
error bars are used. FID is computed on 1,024 generated and 2,048 reference
feature rows per endpoint, using the original uint8 conversion and weights.
It is not a population FID or a standard50k-image benchmark claim.

All35 historical rows plus six new rows are retained. Seventeen20k endpoints
form the primary table; all 32 contrasts from each new arm to every other
endpoint are reported without selecting favorable comparisons. Continuous
ratios/differences are in[primary-contrasts.csv](primary-contrasts.csv).
The 5%gate predates these outputs; no comparison or gate was changed after
seeing the new FIDs. Analysis source/layout were frozen at 2afd4e0 before
metric-output existed. The formal source snapshot is f7bbdd5.

P plots average six physical matrices at each of seven states; individual
records are retained, not treated as model repetitions. Singular-value CV
uses population std/mean of each matrix's64 singular values. Eigenvalues are
local P spectra, not predictor stability bounds. Clock bands are within-panel
SD over actual image/patch values; time-only SD is exactly zero by construction.
The five fixed times provide descriptive profiles, not a fitted continuous
causal model. All clock profiles, head norms/biases and local conditional
matrix summaries remain in the CSVs.

Training curves preserve original logging schedules. Clipping fractions
summarize each stored update window; different checkpoint intervals do not
create equal-duration observations. Resource timing is from one concurrent
CPU cohort and is descriptive, not a repeated hardware speed benchmark.

All three registered audits pass. Sample/feature equality is bitwise; FID is
independently recomputed with the original 1e-6 audit tolerance. Actual terminal
receipts are preserved, including the parent controller's observed exit0.
No full optimization-trajectory replay, causal decomposition of FID, or
large-model transfer result is claimed.
