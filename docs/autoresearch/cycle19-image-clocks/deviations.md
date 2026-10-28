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

## Native raw-fraction metadata

The first metric producer reused paired fitting records from cycle17, whose
raw out-of-range field came from 32-step sampling, while this diagnostic's
FID correctly used the cycle18 128-step banks. The four native raw-fraction
fields therefore had the wrong grid provenance. This affects no predicted
array, feature, FID or fitting loss.

The original producer JSON and audits remain unchanged. The separately saved
correct_metadata.py recomputes all six fractions from their actual registered
128-step raw banks and creates metric-metadata-v2/result.json plus a receipt.
It changes only the four native metadata fields; every score and loss is
checked identical. Public curation uses this corrected file, and archive.py
independently recomputes each fraction again. No generation/metric retry or
numerical tolerance change occurs. The public future producer now collects
the correct-grid fractions directly.
