# Exact evidence and comparison limits

## Protocol

The six new cells use seed0, 5000 updates, batch512, AdamW3e-4, weight decay0,
gradient clipping1, cached-bank pair labels and the unchanged data/evaluation
streams. Width64/depth3 and the two SwiGLU sizes are unchanged. Formal runs
use two concurrent one-thread CPU workers on the Mac mini; no GPU is reserved.
The only training intervention is the explicit common-backbone initialization.
The full preregistered control is [protocol-backbone-pairing.md](../protocol-backbone-pairing.md).

`comparison.csv` retains all 24 endpoint rows: 18 reused Stage15 rows and six
new baseline rows. `group=original/paired` distinguishes their origin; the raw
model labels remain unchanged. The model and budget columns are retained.
`baseline-shifts.json` reports new/original ratios separately for each width,
dataset and metric. It is not a pooled performance estimate.

Primary `validation_oracle_mse` uses the original uniform-time validation bank
and averages squared error per output coordinate. `field_mse` and
`jacobian_error` use 512 common states at each of five fixed times, equally
weighted; the latter is squared Frobenius error per state. They have different
sampling/normalization conventions and cannot be numerically substituted for
one another. Validation SW2 uses 2048 source/target samples, 128 directions and
Heun32. These figures are descriptive endpoint observations and do not prove
joint mode quality in product16.

## Audit coverage

- `v1-native-audit.json`: four cells, 24 checkpoints, 20 oracle arrays.
- `product16-native-audit.json`: two cells, 12 checkpoints, 10 oracle arrays.
- `v1-endpoint-replay.json` / `product16-endpoint-replay.json`: six final
  checkpoints and 30 complete saved field/Jacobian arrays; maximum replay
  difference is zero on this Mac/Torch runtime.
- `initialization-audit.json`: all six new initializations contain the exact
  22 common state tensors and Fourier buffer from the hash-bound references;
  their 12 native SwiGLU FFN tensors and initial Torch RNG state are unchanged.
  Every snapshot data-generator state equals all four reused ODE references
  at the corresponding step, totaling 144 comparisons.
- `provenance.json`: analysis source hash, input hashes and replay totals.
  Both frozen runtime file manifests were rechecked before analysis.
- `source/` preserves the posthoc analysis and endpoint replay programs.
  The native audit program is the unchanged file in the new frozen runtime.

Full checkpoints, source manifests and raw arrays remain local at:

```text
/Users/mr.z/ode-workspace-prep/architecture-search/additive_generator_20261024_m1/
/Users/mr.z/ode-workspace-prep/architecture-search/backbone_pairing_20261024_m1/
```

## Monitoring and operational notes

`monitoring-evidence.json` records actual first/last telemetry samples and
available memory. Initial audit inspection found v1 already complete and a
recent product16 sample naming supervisor47706 and workers47707/47708 at
step2000. A subsequent direct PID check found those processes already exited;
this audit therefore does not claim its own independent live observation of
those worker commands. The root launcher performed its separate live check.
Queue completion is recorded at 2026-10-24T13:06:24.787525Z.

The monitor's final file updates the terminal state while its `training` list
may be the last 10-second sample. Terminal verification therefore uses actual
completion markers/checkpoints/status, not that potentially stale list. No job
was restarted or duplicated.

Two read-only discovery errors were corrected: system Python lacked psutil,
so inspection used the recorded venv; the endpoint replay script was absent
from the frozen training tree, so its existing source was copied into this
analysis directory and run against the frozen model source. Neither error
changed training, weakened a numerical tolerance, or modified a frozen runtime.

## Decision limits

The old and new baselines stay visible. The screen is reapplied descriptively
against both groups, without replacing old STOP decisions or declaring a win
because a baseline weakened. The better paired baseline can differ by metric;
its envelope is a comparison device, not one model with all those scores.
Actual paired-baseline changes are large enough to change width ordering, but
not to remove the ODE fitting/cost deficit.

No new independent holdout, training-trajectory replay, or regenerated sample
array is claimed. The full architecture goal requires those additional forms
of confirmation if a genuinely useful candidate is later identified.
