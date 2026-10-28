# Comparison units and numerical checks

The primary unit is the fixed 20k model endpoint under the registered protocol.
The complete table contains 35 rows:32 retained historical endpoints and three
new 5k / 10k / 20k rows from one new trajectory. The primary table has 15 rows and
the contrast table 14 comparisons. Checkpoints and common image starts are
not independent trained models. All contrasts are descriptive measured
differences; no p-values, confidence bands or population winner claims are
computed. The original model seed/data-stream settings are in the protocol.

New/default Swi86 retain 211412 trainable values and 16512 FFN forward MACs per
token/block. The comparison changes initialization only. Matching the initial
function and shared/input derivatives does not match all parameter tangents,
gradient clipping factors or subsequent optimization trajectories.

The registered decision is FID<=0.95 times default Swi86, with weighted and
clean validation MSE each<=1.05 times its default value. Actual ratios are
0.9515854711, 0.9902192324 and 0.9933672030. FID misses; both fitting guards pass.
The numerical cutoff 185.5445923 is retained without rounding into a pass.
No multiple-comparison correction is applied because there are no inferential
tests; all 14 continuous endpoint contrasts are reported together.

The training audit reconstructs 20000 input draws and all 21 stored stream
boundaries, replays seven 1024-image validation banks and 35 time panels,
checks 63 matrix arrays and reproduces six saved final optimizer updates.
It does not replay the entire 20000-update optimizer trajectory.

The sampling audit independently reproduces all 3072 Heun images bitwise.
The metric audit replays 3072 generated and 2048 reference features bitwise,
independently spells the uint 8 conversion and recomputes FID through a separate
SVD implementation. Absolute FID errors are 4.55e-13, 3.41e-13 and 3.98e-13.
All formal files and frozen parents retain their hashes. Actual terminal
controller/curation/plot exits and all six child exits are preserved.

Matrix rows describe the native gate/value/output arrays. Their singular
values do not bound the whole denoiser. A zero matrix's singular-value CV is
undefined; the curator leaves such a value blank. Branch norms quantify
parameter movement, not contribution to FID or the causal use of a channel.

Historical interventions remain labeled as frozen interventions; their
inference-active and training-active parameter counts are kept separate.
All FIDs use the original 1024 generated/2048 real-reference banks and Heun128,
without clipping intermediate sampler states. These are the existing small
image-screen metrics, not a full-scale image-generation benchmark.
