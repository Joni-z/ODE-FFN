# Preparation-only dtype correction

The first runtime `oct28-cycle19` preserves registered source and preflight
exit1. Its native replay/calibration starts, but the promoted algebra witness
fails before any formal intervention: JiT's sinusoidal timestep features are
float32 while its promoted MLP weights are float64. No output/sampling
directory or FID is created.

In the new v2 runtime, witness_model lifts these same time features before
the promoted time MLP. Only the algebra witness uses this hook. Calibration,
formal predictors and sampling use unchanged native float32 models. Native
source, parameters, constants, datasets, thresholds and tolerances are not
altered. The old preparation/source/log are retained; no training or sample
run is restarted.
