# Hidden versus output offset: a matched local-function control

Declared before new outcomes, 26 October 2026. This follows the sealed [P/W experiment](../research-effective-map-2026-10-26/analysis-report.md). It tests a generic response mechanism; it does not reopen the stopped original-ODE affine-forcing setting or nominate an ODE method.

## Question and mathematical distinction
Both unbiased P/W cores fit known fixed-flow targets accurately and retain large errors on SwiGLU targets. Direct W freedom alone does not close that gap. For a bias-free single-SiLU core, the exact global odd component is linear because SiLU(u)-SiLU(-u)=u. An output offset cancels from the odd component. A hidden offset can move activation thresholds and introduce nonlinear odd response. Actual common features need not be centrally symmetric, so this identity is not asserted to be a data-distribution approximation bound.

Compare three cores:

```text
No offset:      W2 SiLU(W1h)
Hidden offset:  W2 SiLU(W1h+b)
Output offset:  W2 SiLU(W1h)+b
```

The two new arms each add exactly one 64-vector, initialized to zero, with 8256 parameters. They share the exact cold W matrices and bitwise initial function/input Jacobian of the preceding 8192-parameter W core. Implement the zero vector as a separate addition so the comparison does not change the matrix-product convention. No output and hidden offsets are combined.

## Targets, data and runs
Keep both previous target families and all three positions: the folded fixed-flow targets with exact representers in all cores, and the unchanged SwiGLU128 local targets with unknown representability in the smaller cores. Reuse the exact feature/label caches from effective_map_20261026: 100000 training rows, 20000 validation rows and five 512-row time panels. No new test set or artificial mirrored-target training is used.

Twelve new cells: two placements × two target families × three positions. Reuse the six no-offset W trajectories and six ordinary SwiGLU trajectories from the audited prior cohort after checking constructors, optimizer, targets, input/index digests and native outputs. Those twelve rows remain historical references, never new training.

All new cells keep seed 0, the identical 512-index stream, 5000 updates, AdamW 3e-4/default moments, weight decay 0, clip 1 and no schedule. Only placement changes between the two new arms; they have identical matrix work and parameter counts. Record actual costs, gradient/clipping windows and every offset vector. The larger SwiGLU reference remains a strong control, not an equal-parameter comparison.

## Outcomes and interpretation
Primary: all-validation relative MSE for every target/position/arm. Secondary: absolute risk, all time panels, first-128-input local Jacobian error, all saved trajectories and offset vectors at 0/1000/2500/5000.

A substantial hidden-specific improvement is operationalized as at least two SwiGLU positions with hidden-offset relative MSE at most 1% and no more than half both the no-offset and output-offset risks. Report all continuous results, known-representer controls and exceptions. This is a descriptive criterion, not a statistical test or architecture acceptance rule. If both placements help comparably, do not attribute the effect specifically to movable nonlinear thresholds. If neither closes the gap, the translation account weakens; feature count and multiplicative response remain unresolved.

The exact odd-component distinction is a function-class fact. Any measured training advantage also depends on optimization and the actual nonsymmetric input distribution. Placement does not isolate expressivity from all learning dynamics. No result automatically authorizes an original ODE bias, width, clock, spectral or optimizer rescue. A later method still needs a distinct ODE mechanism and ordinary FM/transfer/cost evidence.

## Checks and execution
Preflight all three positions for bitwise initial values/input Jacobians against the existing W core. At nonzero deterministic offsets compare analytic input Jacobians and offset gradients against native autograd, so merely passing zero-offset tests cannot hide a wrong branch. Check six exact fixed-flow representers, every new three-versus-two-plus-resume history, and all twelve historical references' cold-state/stream definitions.

Save weights, optimizer, global/sampling RNG and actual consumed index/input/label digests at four snapshots, metrics every 500, status every 100 and real CPU/RSS every five seconds. All endpoint arrays/Jacobians and all complete streams must replay independently. Freeze sources/bindings, use at most two one-thread CPU children, launch once and preserve failures. No restart on observation expiry.

Procedural guidance uses planning-with-files, hypothesis-generation, experimental-design, daily-coding and results-analysis. Workflow provenance is [Kassis et al. (2026), Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065), with metadata verified in the preceding review; it is not model evidence.
