# Research notes

This file holds current research facts and decisions. The complete prior
journal is preserved in [history](docs/history/2026-11-12-repository-organization/README.md).
See [STATUS.md](STATUS.md) for changing job states and training parameters.

## Current method

- Authorized 14 November: retain the qualified native baseline and run widened
  ISRU-D K8/K1, both internal width 1200, seed 0, 100 epochs and periodic
  FID50k. [New protocol](docs/celeba128-wide-20261114/README.md). This supersedes
  the audit's narrow-SwiGLU recommendation below; K1 uses the same expanded
  architecture as K8, changing only order. Full CPU/CUDA readiness passed; K8 job23015406 is training on H100 gh011,
  K1 job23015408 is training on L40S gl068. Monitor460860 records queue/progress/FID every60s.
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

The user authorized the parameter-matched wide K8/K1 comparison on14November.
Keep the qualified native SwiGLU baseline fixed. New jobs use internal1200,
seed0,100epochs,63,500updates, and FID50k at20/40/60/80/100. Both ODE orders
share initialization and a130,402,560-parameter architecture. No narrow
SwiGLU, Muon switch or spectral constraint is selected. Source, CUDA readiness,
measured time limits and monitoring belong to
[the new study](docs/celeba128-wide-20261114/README.md).

The earlier [audit](docs/celeba128-20261112/next-step-audit.md) remains an
evidence snapshot. Its proposed narrow-control sequence was superseded by
the user's choice to widen the candidate and preserve the complete baseline.
