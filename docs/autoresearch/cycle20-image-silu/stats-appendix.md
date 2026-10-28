# Statistical and numerical scope

The new cohort contains precisely three cold ordinary models, with widths
fixed by shape, parameter coverage and K2 matrix work before outcomes. Model
seed0, final update5,000 and all inherited training settings are fixed. Four
native models and two frozen-clock interventions are reused from sealed
studies. The latter have no new training trajectory.

All comparisons use the same1,024 image/noise/time validation pairs and1,024
Gaussian generation starts. All five64-image fixed-time panels are retained.
Per-image SD and quantiles in the public table describe paired example
variation; pixels, panels, images and checkpoints are not independent trained
replications. No p-values, confidence intervals over training or population
ordering claims are made. The strict both-quality-lower field in
contrasts.csv means only lower final weighted MSE and FID on these banks;
it is not a cost-dominance or significance test.

Measured FFN forward/forward-plus-backward matrix MACs are8,192/24,576 for
SiLU64,16,768/50,304 forSiLU131 and24,704/74,112 forSiLU193. These include
input and weight gradients in the dispatch fixture. They exclude pointwise
operations, Adam, attention and the remaining shared backbone. SiLU193 matches
K2 matrix work but has11.46% more whole parameters. SiLU131 is within0.18%
of K1 whole parameters but uses1.55% more matrix work. No equal-wall-time
claim follows. Costs for frozen-clock interventions are left blank rather
than presented as new trained cost controls.

Every new model shares the native non-FFN cold tensors and buffers. FFN
initial functions and widths differ, despite equal zero-head predictions.
Before quality training, standalone float64 SiLU values/JVPs, actual
nonzero-head backbone gradients, measured costs, cold Heun integration and
physical three-update checkpoint resume pass. The resume compares full
model/optimizer moments, RNG states, buffers and paired streams. It does
not establish equality of later trajectories across architectures.

The post-training audit restores33saved checkpoints, replays predictions
bitwise, checks checkpoint RNG/moments and independently redraws all5,000
stream batches. It replays60new time panels,72matrix spectra and all3,072
generated images with a manual Heun expression. This is saved-state and
sampler verification, not independent retraining of every optimizer update.
All157formal training-output files remain unchanged through that audit.

All FIDs use1,024generated/2,048reference counts, fixed Heun128 and the pinned
stock compatibility Inception extractor, weights and uint8 conversion.
Small-sample FID bias remains. New3,072generated features and2,048reference
features reproduce bitwise; six reused banks retain prior full audits/seals.
All nine scores are independently recomputed using SciPy gesvd and differ
by at most5.40e-13, below the unchanged1e-8 limit. All161formal training and
metric files remain unchanged through the feature audit. Raw fractions are
recomputed from actual128-step arrays, including every reused endpoint.

Four figures show all nine endpoints, all77recorded cold checkpoint rows,
all72new matrix records, and the same first eight generated cases per model.
Full singular values are retained, with eigenvalues only for square maps.
No extreme checkpoint, image subset or best width is selected as a new method.
Submitted Report1 and all parent source/data/checkpoints remain unchanged.
