# Figure purposes and interpretation

## time-and-direction.png
All three models, both checkpoints; top: common-path velocity MSE over33times
(log axes); bottom:20k-minus15k radial/angular squared energy. Error components
use the origin-based geometric frame, not an assumed continuous manifold.
Notice late fitting improvements alongside mixed early changes. This selects
time/direction questions rather than a uniform error-amplitude explanation.

## oracle-switches.png
SW2 for all36switches on the preselected bank0; full-oracle score is dotted.
Solid15k/dashed20k; blue learned prefix and orange oracle prefix. Cuts.25/.5/.75.
Bank0reverses the eight-bank mean trend, and some hybrids score below the
oracle on this empirical target. Interpret together with endpoint map error
and sector counts, not as a new generative sampler or an aggregate-bank result.

## directional-propagation.png
Median observed secant gain along each actual error direction, excluding only
norm<=1e-8and retaining counts/extrema in CSV. Reference line1. The curves are
close across checkpoints; they do not establish a global contraction bound.
This favors examining net mode allocation over a median-amplification story.

## precision/checkpoint-precision.png
Blue: all eight banks concatenated at32768samples. Gray: all dependent omission
pools at28672samples. All six checkpoints remain shown. Native deterioration
is clearest for Swi64, weaker for ISRU-D and small for Free ISRU. No endpoint
selection or training-seed uncertainty is implied.

## sectors/sector-crossings.png
Cumulative gross crossings in saved native paths; dashed verticalt=.75.
Most events are already complete by then. Gross crossings can cancel, so use
sector-masses.csv and endpoint-sector-error.csv for probability mass and
map-error interpretation. The oracle has zero crossings.

## shape-mass/shape-and-mass.png
The same SW2 formula on original samples, assigned means and centered residuals,
all32768samples. Assigned-mean discrepancies increase while centered-residual
discrepancies decrease across all three models. Axis scales differ; these are
not additive terms. This identifies mode allocation as a measured next target.

All six figures were visually inspected before publication; labels, curves
and table values match their saved-data definitions. Full case membership
is source-bound. No synthetic result is a FID claim.
