# Research notes

This file holds current research facts and decisions. The complete prior
journal is preserved in [history](docs/history/2026-11-12-repository-organization/README.md).
See [STATUS.md](STATUS.md) for changing job states and training parameters.

## Current method

- The selected image candidate is frozen-diagonal ISRU-D K8, implemented in
  [ffn.py](ffn.py). Compute d0 once at a layer entrance, then accumulate
  the input plus eight Taylor terms; use two independent layers around SiLU.
- The ordinary image control is the author's native SwiGLU. Free ISRU is a
  separate available FFN, not the current image candidate.
- Model/loss/sampler/EMA source is pinned to the author-derived runtime.
  [The operator note](docs/research-assessments/2026-11-11-image-candidate.md)
  and [source attribution](UPSTREAM.md) explain the implementation.

## What the evidence currently says

- Completed synthetic studies retain a Ring8 field-fitting lead, while
  Product16 sampling favors ordinary references. Read the
  [complete common-bank comparison](docs/autoresearch/professor-20261103/cycle14-results-complete/analysis-report.md).
- The [transport diagnosis](docs/autoresearch/professor-20261103/cycle15-results/analysis-report.md)
  links better fitting with changed mode allocation. This is a mechanism
  question alongside the current image transfer.
- Both image models completed 100 epochs on L40S. K8 has 5.41% higher training
  loss and 1.75x steady epoch time; GPU use is 96–98%. All 101 P snapshots are
  verified, with more concentrated raw spectra. [Measured results](docs/celeba128-20261112/results.md).
- Completed CelebA128 FID50k is 35.91785 for SwiGLU and 43.90767 for K8. K8 is
  worse by 7.99 points (+22.24%) at 100 epochs, with fewer parameters and slower
  training. Both evaluation jobs completed 0:0. [Verified comparison](docs/celeba128-20261112/results.md#standard-fid-results).
- Subsequent image training evaluates FID50k every 20 completed epochs, keeps
  complete numbered checkpoints and preserves training RNG around evaluation.
  [Metric/cadence and time budget](docs/celeba128-20261112/periodic-fid.md).
- Reports 1 and 2 are final. Later evidence belongs outside their folders.

## Current decisions

Use seed 0, the selected K8 operator and a matched native control. Keep the
objective and optimizer fixed while testing the FFN. The latest duration and
resource decision is in the [budget note](docs/celeba128-20261112/budget-20261112.md).
Readiness measurements, CPU data probes and complete training results are
different records. The ordinary CelebA128 baseline meets FID50k below 50; the completed K8
comparison is worse in both fitting and generation. Increasing the same K8
training budget is not selected by this status check.
