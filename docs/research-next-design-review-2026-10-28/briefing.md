# Research review: why the ODE search is not producing a useful successor

## User problem and scope

Zhizhe Zhang wants to improve a JiT/DiT feed-forward architecture using a
mathematically grounded ODE design that earns its cost under flow matching.
The original professor block is two input-clock linear Taylor maps separated
by SiLU. The user asks for evidence-led direction, strong SwiGLU baselines,
simple changes, actual P trajectories and useful results before lengthy
mathematical explanation. They recently objected that formula checks were
being presented as progress without an improved model. Experiments run on
Mac mini/Lenovo, seed0 only; MBP/torch and multiseed are excluded.

## Current evidence, to verify in primary files

1. A frozen complete nonlinear Taylor upgrade to already trained SwiGLU
   worsens all four synthetic laws. This does not compare cold solver-aware
   training. Its recorded stopping decision forbids a Taylor-training rescue
   without new independent task evidence.
2. Most late tangent derivative error is symmetric, not rotational. An
   input-dependent orthogonal frame around the original P polynomial cuts
   thin128D field error by78% versus SwiGLU and recovers clusters. Same-frame
   K1 nearly matches fitting and has better sample SW2. Thick-data generation
   fails. The particular frame/K2 setting is stopped, not universally refuted.
3. Adding an independent second coefficient gives modest thin improvement,
   fails thick transfer and does not materially beat ordinary added branches.
4. Image native K2 has FID183.698 versus192.018 for Swi129, but worse fitting.
   Freezing all six clocks yields179.718. Ordinary SiLU193 reaches180.045 at
   equal FFN dense work, with11.46%more whole trainable parameters; fitting
   is better. SiLU64's cold FID197.950 is much worse despite representing each
   fixed-clock function. No optimization cause has been isolated.
5. Replacing learned polynomial maps by exact exponentials is bad for K1,
   mixed for K2. Objective-level math distinguishes a Bayes conditional mean
   from a trajectory-invariant endpoint map and shows a posterior precision
   equation needs a third-moment term. None selects a useful hidden ODE.
6. A new practical export preserves the entire frozen folded1024-image bank
   exactly and saves CPU generation time8.68to6.00seconds. This is useful
   inference simplification, not a novel trained architecture.

## Primary reading set

All paths below are relative to `/Users/mr.z/ODE-FFN-autoresearch`.

- `docs/autoresearch/program.md`
- `docs/autoresearch/cycle01-complete-field/analysis.md`
- `docs/autoresearch/cycle10-field-derivatives/analysis.md`
- `docs/autoresearch/cycle11-conditional-frame/analysis.md` and `protocol.md`
- `docs/autoresearch/cycle14-independent-coefficient/analysis-report.md`
- `docs/autoresearch/cycle17-cifar-baselines/analysis-report.md` and its curves/results
- `docs/autoresearch/cycle20-image-silu/analysis-report.md` and its curves/results
- `docs/autoresearch/cycle22-denoiser-dynamics/analysis-report.md`
- `docs/autoresearch/nonlinear-swiglu-review/decision.md`
- `docs/autoresearch/folded-image-inference/README.md`
- `docs/research-architecture-search/nonlinear-drift-readout.md`
- `docs/research-joint-dependence-2026-10-26/next-question.md`

Historical local-family screens (covariance/gradient readout, recurrence,
ISRU/cubic responses, time adapter, spectral caps, frame/independent-C) have
explicit stopping records. Verify a proposed mechanism against those actual
files before calling it a new direction. No need to read every raw checkpoint
or all existing audit machinery to answer the scientific question.

## Questions requiring judgment

Are the scientific conclusions correctly scoped, or are we overgeneralizing
negative screens? Are stopping rules or a demand for a K2-specific win being
mistaken for mathematical impossibility? Do not relax past gates to relabel
failures. If a prospective different question is warranted, state why and
which new evidence would distinguish it.

Propose at most one implementation-ready next direction if it truly follows
from these results: exact state/update, its relationship to the professor
layer, task-specific reason, full derivative/cost caveats, strong ordinary
controls, minimal decisive experiment and clear failure meaning. If no
architecture is justified, name one missing empirical premise that would
actually change selection. A further generic identity, benchmark expansion
or architectural rename is inadequate. Consider whether all5000-update
screens answer the intended quality question; inspect trajectories before
recommending more training. No positive result or novelty should be assumed.

The reviewer is fresh but same-family/provisional; actual returned model
identity may be unavailable. Review completion and empirical method success
are separate. Do not start training or modify model files.
