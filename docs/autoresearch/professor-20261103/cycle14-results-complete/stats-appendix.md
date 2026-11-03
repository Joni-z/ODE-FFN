# Statistical scope

Complete: ten frozen model/settings, three checkpoints, two solver resolutions,
eight common banks per comparison;480evaluations and30field arrays. Allmodel
fits retain seed0. Means and sample SD describe conditional evaluation banks;
paired differences use common arrays. Checkpoints and solver resolutions are
sensitivity axes, not independent training replicates. No p-values or training
confidence intervals are computed. The20k/Heun128 endpoint is the preregistered
primary for this diagnostic; no checkpoint or rate is selected by SW2.

The complete paired_differences.csv retains every reference/time/resolution
comparison, including reversed signs. No nonfinite result is omitted and no
reference floor is subtracted. The separate Ring8 pooling analysis is not
reclassified as the primary endpoint. See replay.json and native_anchors.csv
for independent numerical and byte-level source evidence.
