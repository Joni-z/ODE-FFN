# Figures and interpretation

All five PNGs were visually inspected. The first four are the registered
core curation; the last is explicitly post-outcome reporting. Numeric tables
accompany all plots. No error bars represent independent training repeats.

## 1. Primary comparison and FID accounting

[Figure](results/figures/reference-comparison.png),
[PDF](results/figures/reference-comparison.pdf).

Purpose: compare both fixed classical references against every primary neural
endpoint. Data: [primary-comparison.csv](results/primary-comparison.csv).
The left panel has all 16 primary FIDs; the right shows the feature-mean and
covariance accounting terms for six selected rows, with every term retained
in the CSV. PPCA293.73/isotropic405.30 exceed fixedP174.71/native173.44.
Caption must distinguish train-only fits from neural 20k training and state
the 1,024/2,048 generated/reference counts. The feature-mean term does not
identify a pixel-mean cause; the two terms do not localize a mechanism.

## 2. Original time panels

[Figure](results/figures/time-panels.png),
[PDF](results/figures/time-panels.pdf).

Purpose: locate fitting differences without changing the global gate. Data:
[time-panels.csv](results/time-panels.csv), seven models at five original
times, 64 original pairs each. Much of the PPCA gap is at intermediate times;
its t=0.9 error is lower than native K1, so failure is not uniform across
times. The isotropic reference is poor at most panels. Curves connect fixed
evaluations, not a training trajectory. No time is selected or reweighted.

## 3. Fitted covariance spectrum

[Figure](results/figures/fitted-spectrum.png),
[PDF](results/figures/fitted-spectrum.pdf).

Purpose: show exactly what the train-only covariance fit retains. Data:
[retained-spectrum.csv](results/retained-spectrum.csv) and
[summary.json](results/summary.json). Left: all 59 retained eigenvalues on
a log scale, with average residual variance as a dashed line. Right:
85.7781% of training trace in retained modes, 14.2219% in 3,013 complementary
modes. Caption must state rank was fixed by scalar storage before outcomes;
variance explained is not semantic fidelity or neural-feature importance.

## 4. Fixed examples

[Figure](results/figures/fixed-examples.png),
[PDF](results/figures/fixed-examples.pdf).

Purpose: inspect recognizable differences that scalar FID cannot show.
Each row uses the first eight original common starts; all 16 primary rows
are shown, giving 128 cases. Display conversion follows the native clipping
and quantization, while raw tails are retained in the numeric table.
PPCA retains broad layouts at starts 3/5 but adds fine speckling; isotropic
outputs look largely unstructured. Neural rows remain coarse and blurred.
These examples do not prove a residual-noise cause or a semantic category.
Their fixed order prevents selecting attractive examples after outcomes.

## 5. Reporting-only fixed-subspace decomposition

[Figure](results/subspace-analysis/subspace-comparison.png),
[PDF](results/subspace-analysis/subspace-comparison.pdf).

Purpose: quantify the coarse-layout observation using one unchanged training
PCA basis. Data: [subspaces.csv](results/subspace-analysis/subspaces.csv), all
16 original generated banks plus the original real reference. Left: retained
and complement RMS per original coordinate. PPCA complement.18537 is close
to real.18670 despite a much worse FID than neural rows. Right: complement
share of paired squared difference from PPCA; native55.77%. PPCA's own zero
denominator is omitted. The real reference has no paired generated start.
Caption must say this was added after seeing outcomes and cases. It is not
a FID decomposition, an identified neural mechanism, a manifold-tangent
measurement, or a reason to choose another covariance/noise setting.
