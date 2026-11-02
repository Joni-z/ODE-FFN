# Saved Product16 sample-structure diagnostic

Registered before computing the additional sample readouts. All training and
Heun32 sample arrays stay frozen. This is a CPU/NumPy read-only diagnostic; no
new sampler, data cache, checkpoint selection or training family is introduced.

## Question

Some ordinary Product16 models have a large true-field error but a low stored
random-direction SW2. Two explanations remain possible: terminal generation may
be good despite a different field, or random projections may hide mixture
structure. Do not assume either explanation before inspecting the samples.

## Population and readouts

The target is the known orthogonal image x=Rz of independent coordinates
z_j~.5N(-a_j,.04)+.5N(a_j,.04). Transform saved generated and target samples by
z=xR using the manifest rotation. Keep the original256-direction SW2 as primary.
Additional descriptive readouts, fixed in advance:

- Axis SW2: square root of the average squared sorted-sample difference along
  the16 known latent axes. This tests univariate mixture shapes in meaningful
  directions, not the complete joint law.
- Central-gap mass per axis: fraction |z_j|<a_j/2, with exact population values
  and the saved independent target reference. This detects filling the gap
  between the two modes; it is not a universal generation metric.
- Per-axis second moments and near-mode squared distance `(abs(z_j)-a_j)^2`.
  Retain the target reference rather than interpreting these alone as quality.

All74 stored Product16 cells are included:14 first-wave,15 wave2, and9/9/11/9/7
from autoresearch cycles1–5. Three technical anchors repeat earlier samples;
retain and label their content hashes, not as independent evidence. Verify
checkpoint/config/manifest hashes, shared targets, completion and original
replayed field/SW2 before joining the new readouts. No outcome-based exclusions.

## Fixed case studies

Plot latent coordinates0,7,15 (small/middle/large target separation), with exact
population marginal density and saved-sample histograms. Fixed cases are scalar
K2Muon.02, rawK1Muon.006, absK1Muon.02, Swi128Adam.0003, Swi192Adam.0003,
constantK1Muon.02 and bilinear64Adam.0003. The last case is chosen to diagnose
the already observed field/SW2 disagreement, not nominated as a winning model.
Histogram limits are +/-[a_j+6sigma]; use counts divided by total sample count
and bin width, without renormalizing away out-of-range mass. Save overflow rates.

Random versus latent-axis ranking changes would show a directional limitation
of this readout, not make the new metric a replacement primary. Agreement would
leave the different-field/terminal-quality explanation viable. Axis agreement
still cannot certify joint mode dependence. Do not select architecture or a
checkpoint from these new test readouts, and do not reopen absK1 from a favorable
secondary number. Use the findings to decide what a next registered study must
measure before spending more GPU time.
