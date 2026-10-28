# Figure catalog

Three PNG/PDF pairs were visually reviewed. Endpoints are fixed models and
interventions; no uncertainty bars over trained repetitions are implied.

## fixed-comparison

Purpose: show all six fitting/FID endpoints without choosing a favorable metric.
Data: comparison.csv. Points display exact values; the fitting axis is logarithmic
and zoomed to this cohort, not a zero-based effect-size bar. The FID reference
line is the better ordinary control. Observation: constant clocks retain and
increase the measured FID leads, with tiny primary fitting changes; SwiGLU
still fits better. Caption needs counts, Heun128 and frozen-intervention scope.
Do not claim an ordinary-model training win, new architecture or significance.

## time-panels

Purpose: locate fitting changes over all five prescribed times. Data: all
30rows of time-panels.csv, including 20 reused native and ten new intervention
panels. Each panel uses 64 images. Observation: native/constant curves nearly
overlap, while the late regression gap to SwiGLU remains. Log vertical axes;
these are paired regression errors, not oracle field errors. Time panels are
correlated measurements, not trained repetitions or a generalization interval.

## generation-cases

Purpose: assess the visual size of the finite intervention. Data: the same
first eight Gaussian starts for all six endpoints, in two rows of four per
model. Observation: K1 changes little; K2 shows stronger contrast in some
fixed examples, notably the third start. All outputs remain coarse/blurred.
Display clamp only; raw states remain intact. The lower score and larger raw
tail do not identify clipping as the cause, establish semantic improvement,
or prove mode coverage. No example is selected by score or appearance.
