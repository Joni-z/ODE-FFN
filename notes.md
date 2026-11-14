# Research notes

This file holds current research facts and decisions. The complete prior
journal is preserved in [history](docs/history/2026-11-12-repository-organization/README.md).
See [STATUS.md](STATUS.md) for changing job states and training parameters.

## Current method

- Three seed0 trajectories continue to160: native SwiGLU, expanded ISRU-D K1
  and expanded ISRU-D K8. Standard FID50k is evaluated at120/140/160.
  [Matched status](docs/celeba128-wide-20261114/continuation-160/matched-results.md).
- Native FFN is the author's unchanged SwiGLU. Both expanded ODE FFNs use
  internal1200, two square operators around SiLU, and entrance/exit projections.
  Freeze each entrance diagonal across its Taylor series. [FFN](ffn.py).
- Both ODE orders have130,402,560parameters; native has130,405,632. Use seed0,
  batch256, AdamW5e-5 and the existing data/time/noise/EMA/sampler recipe.
- Runtime17b61a5 is frozen. The baseline continuation restores the original
  source65864ad checkpoint in a separate output, with native math and full
  checkpoint prediction compatibility verified. [Baseline provenance](docs/celeba128-wide-20261114/continuation-160/baseline/README.md).

## What the evidence currently says

- The parameter-matched image pair completed: wideK1 FID37.92486, wideK8
  38.13574, fixed native 35.91785. Expansion improves on oldK8 but does not
  beat native; high order adds no final FID benefit in this setting.
  [Completed wide comparison](docs/celeba128-wide-20261114/results.md).
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

Finish the authorized160epoch evaluations before selecting another experiment.
K1 finished; native and K8 remain active. Preserve the100epoch evidence and
final reports. No new architecture, optimizer, spectral penalty or seed is
selected by the current status check. Compare models only at matching epochs
in the main results table; pending160metrics remain blank.

The [14 November architecture assessment](docs/research-assessments/2026-11-14-next-architecture/recommendation.md)
recommends checkpoint diagnostics before a conditional, equal-work comparison
of frozen-D K2 and state-dependent ISRU midpoint. The earlier sigmoid-clock
feedback experiment failed, so feedback alone is not a supported remedy.
No new model or job is selected by this recommendation.
