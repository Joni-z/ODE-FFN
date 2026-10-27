# Statistical and numerical scope

The comparison unit is one frozen model at the registered 5,000-update endpoint.
The four arms are different architectures, not repeated observations of one
method. They share initialization of the surrounding backbone, training draws,
held-out pairs and generated starts. Saved checkpoints are repeated points
on those same trajectories. Report deterministic finite-cohort differences;
no p-values or confidence intervals over model reruns are claimed.

## Validation

Primary weighted MSE averages 1,024 held-out image/source/time pairs and 3,072 pixel
coordinates per image. It equals clean residual squared divided by the saved
float32 delta squared. Independent audit computes the explicit difference of
predicted and target velocities in float64. This is paired regression error,
not distance to an analytic population vector-field oracle.

[Per-image distributions](results/per-image-distribution.csv) retain mean, SD,
median and 10th/90th percentiles. The SD describes variation across those
image/noise/time pairs; it is not an error bar for training reproducibility.
Pixels/tokens/time panels are not treated as independent model repetitions.

All 44 checkpoint predictions are replayed bitwise. Intermediate train-window
loss/clip statistics are preserved producer logs, rather than claims of a
full independent retraining. Source/stream/buffer/moment checks and the exact
native three-update disk-resume fixture establish the stated execution checks.

## FID

Exactly 1,024 generated versus 2,048 validation-reference images, identical counts
and ordering across methods; compatibility Inception 2,048-dimensional features, pinned
weights SHA256 6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2.
There is no comparison with FID-50K or an unopened final-test score. Official
CIFAR test data are not used. The finite-sample bias remains.

Mean/covariance accounting is post-result algebra of this same registered
score. It adds no quality criterion or image correction. A smaller covariance
trace alone does not prove worse diversity, and a lower FID does not establish
mode coverage. The stored unclipped image values have out-of-range fractions
2.213%/2.734%/1.265%/0.994% for K1/K2/Swi86/Swi129. Only the common display/
uint8 feature conversion clips; sampler states are never clipped.

CPU/MPS feature fixture gates were maxabs 1e-3/RMS 1e-4 before training;
observed errors 5.72e-6/5.72e-7. Fresh MPS feature replays are bitwise. Exact
empirical covariance-FID is computed through the low-rank identity; the
independent implementation uses a different SVD driver and agrees within
4.3e-13. No fitted epsilon, tolerance or feature selection.

## Effect sizes and selection

K2 versus Swi129: weighted MSE +3.275%, clean MSE +1.904%, FID -5.515%.
K2 versus K1: weighted MSE +0.255%, clean MSE -0.007%, FID -0.818%.
The predeclared fitting/order thresholds fail. No best checkpoint, model
subset or additional metric is substituted to reverse that decision.

No inference test is justified by counting the 44 checkpoints, 1,024 images, 
4,096 tokens in a time panel or 64 singular values as independent trained models.
The numerical tables describe this fixed screen; its decision does not
establish ImageNet transfer or a new architecture's effectiveness.
