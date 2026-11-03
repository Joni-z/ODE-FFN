# ODE-FFN

JiT with a replaceable FFN. Training code is copied from
[the authors' PyTorch repository](https://github.com/LTH14/JiT), pinned to
`cbc743a2ada5e9762697da2c83f8c4f8379e8c17`. The authors' MIT license is
retained. [UPSTREAM.md](UPSTREAM.md) describes every local change.

Default `--ffn swiglu` keeps the author's SwiGLU unchanged. Our candidate
is `--ffn isru`; its implementation and hook are in [ffn.py](ffn.py).
The loss, time distribution, EMA equations, Euler/Heun sampler and metric
implementation come from upstream, not the old training framework.

The professor-guided autoresearch loop tests Muon/spectral geometry and
input-dependent diagonal generators in a separate synthetic backbone. The
[research map](docs/autoresearch/professor-20261103/research-map.md) connects
questions, derivations, controlled tests and decisions.

The [complete frozen-checkpoint validation](docs/autoresearch/professor-20261103/cycle14-results-complete/analysis-report.md)
retains a Ring8 field-fitting lead for ISRU-D, with sampling close to selected
SwiGLU64. Product16 favors ordinary references: mean SW2 is0.08900 for ISRU-D,
0.08596 for Free ISRU and0.08350 for selected SwiGLU192. Both tasks use common
sampling banks and fixed checkpoints. The next [transport-error diagnostic](docs/autoresearch/professor-20261103/cycle15.md)
compares saved models by time, direction and oracle-field replacement.

The preceding [bounded-odd input-diagonal study](docs/autoresearch/professor-20261103/cycle13-results/analysis-report.md)
finds an AdamW Ring8 lead over SwiGLU, but only2.25%field separation from
ordinary free ISRU; Product16 favors the ordinary model. All eighteen20k runs
and ten native references replay exactly. No ODE-specific advantage is established.

The [matched 80k study](docs/autoresearch/professor-20261103/cycle6-results/findings.md)
retains a limited fitting-versus-sampling tradeoff, with no overall advantage
over ordinary SwiGLU. The [frozen spectral-shape test](docs/autoresearch/professor-20261103/cycle7-results/findings.md)
rejects post-hoc flattening as a general repair. The subsequent
[training constraint comparison](docs/autoresearch/professor-20261103/cycle8-results/findings.md)
also rejects the registered weak-relaxation setting: P spectra become less
concentrated but field fitting and sampling worsen. The completed [native feature audit](docs/autoresearch/professor-20261103/cycle9-results/findings.md)
explains the mismatch between P spectra and actual layer responses. Its
[derivation](docs/autoresearch/professor-20261103/feature-metric.md) does not
select whitening or a new optimizer.
The completed [radial-diagonal study](docs/autoresearch/professor-20261103/cycle10-results/findings.md)
reduces hidden responses, with only a small Ring8 fitting gain and worse
Product16 fitting. Neither passes its registered continuation criterion.
The completed [controller/output factorial](docs/autoresearch/professor-20261103/cycle11-results/findings.md)
finds one controller/output interaction, while all new cases remain behind
near-parameter-matched SwiGLU.
The [frozen output swaps](docs/autoresearch/professor-20261103/cycle12-results/analysis-report.md)
keep all tensors fixed and reject a simple present-day output repair.
These vector experiments do not establish JiT image quality.

## Code

```text
main_jit.py       # upstream entry point; small device/dataset adapters
engine_jit.py     # upstream epoch loop and evaluation
denoiser.py       # upstream objective, EMA and sampler; FFN hook in __init__
model_jit.py      # upstream JiT and original SwiGLU
ffn.py           # replaceable FFN
util/            # upstream utilities
tests/           # one upstream-compatibility check
reports/         # preserved reports, separate from training
docs/            # historical evidence; not imported by active code
```

## Install

For the original CUDA stack use [environment.yaml](environment.yaml).
For Mac mini CPU/MPS:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main_jit.py --help
```

The Mac profile pins **the author's torch-fidelity fork**. No custom FID
calculation is included in the active code.

## Run

Use [the upstream ImageNet instructions](https://github.com/LTH14/JiT/tree/cbc743a2ada5e9762697da2c83f8c4f8379e8c17)
for the published recipe. Add `--ffn swiglu` or `--ffn isru`.
For a single Mac process pass `--device mps --num_workers 0 --no_pin_mem`.
CUDA distributed execution retains the upstream path.
The native Windows RTX 4060 deployment is documented in
[the Lenovo run notes](docs/lenovo-baseline/README.md).
ImageNet online FID additionally needs the authors' reference files in
`fid_stats/`, available in the pinned upstream repository. These data files
are gitignored here; CIFAR uses the standard dataset reference instead.

### CIFAR-10 status

**A qualifying CIFAR baseline has not been reproduced.** Upstream publishes
ImageNet recipes, not a CIFAR-10 recipe. Our dataset adapter and `JiT-B/2`
patch-size variant are local ports, not published author results. The first explicitly authorized
SwiGLU baseline is paused after four completed epochs for a bounded performance
study, now complete; see [current status](docs/lenovo-baseline/status.md).
It has not yet met the quality gate.

The relevant flags are:

```text
--dataset cifar10 --img_size 32 --class_num 10 --model JiT-B/2
--label_drop_prob 1 --cfg 0 --ffn swiglu
--data_path /path/to/cifar10 --device mps --num_workers 0 --no_pin_mem
```

These select unconditional training. Add `--download` only if the official
torchvision dataset is absent. Other training/batch settings must be established
by baseline reproduction; the full B model is much larger than the old tiny
experiment. FID defaults to 50,000 generated images and the standard
`cifar10-train` reference through upstream torch-fidelity. **FID below 50 is
the minimum gate before interpreting an FFN comparison.** Code checks are
not a substitute for that result.

## Check the copy

```sh
git clone https://github.com/LTH14/JiT.git /tmp/jit-reference
git -C /tmp/jit-reference checkout cbc743a2ada5e9762697da2c83f8c4f8379e8c17
python tests/check_upstream.py --upstream /tmp/jit-reference
```

The tiny fixture compares default state, active forward/backward, one actual
upstream epoch update, both EMAs and both samplers against the pinned source.
The CUDA-only reference receives placement-only adaptations for CPU comparison.
A brief MPS update runs when available. This does not establish image quality.

## Previous work

The [revised Report2](reports/report2/README.md) follows the original ODEFFN
line, with a [complete attempt ledger](reports/report2/attempts.md). It covers
local clock/P/order studies, full-gradient readouts, conditional directions,
nonlinear flow, Muon/spectral geometry and the latest diagonal-feature studies
with their ordinary controls.
The PDF is three pages including figures/tables, updated3November2026. The synthetic gains remain task-specific;
a general advantage of the original Taylor coupling is unproven.

The [synthetic evidence audit](docs/synthetic-audit-2026-11-01/EXPERIMENT_AUDIT.md)
reproduces a Product16 field-fitting gain for free-matrix ISRU in a small
velocity MLP. It does not establish a general ODE or image-generation advantage.
Further training is awaiting a concrete experimental decision.

Complete earlier code is preserved on
[research-archive-20261101](https://github.com/Joni-z/ODE-FFN/tree/research-archive-20261101).
The original local checkout and frozen experiment artifacts are retained.
Report1 is final and unchanged; [Report2](reports/report2/README.md) remains an
interim draft. Old CIFAR scores above 200 cannot substantiate architecture
claims. Use the archive to reproduce those historical studies.
