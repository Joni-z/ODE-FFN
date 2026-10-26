# Audit and measurement details

The registered protocol fixes three Stage23 `vector_parent` final checkpoints,
the original 595-file runtime, six independent K1 solves per model, and all five
times (0.1, 0.3, 0.5, 0.7, 0.9). Each panel contains 512 inputs. No model was
trained or selected during this diagnostic.

Calibration reads only `xt[:256]` and `t[:256]` from each original probe bank.
The six duration tensors and six 64-channel means per model are saved in
`clock-means/`. Their float32 bytes match the independent controller-hook
capture exactly, and every overall mean exactly matches its original training
log. Means are converted to float64 for evaluation; no per-time calibration,
label fitting or hidden coefficient choice is used.

`clock-collapse.csv` contains all 15 panels and three equal-time means. Field
MSE averages across examples and coordinates. Jacobian error sums over both
matrix axes, then averages examples. Each risk change equals its signed
alignment term plus its nonnegative finite-change energy. The CSV retains
both terms, their numerical residual and declared tolerance. No denominator
or time weighting was changed after outcomes.

The main diagnostic completed once in session 50959, launched at approximately
2026-10-24 18:32:01 UTC and closed with exit 0 at 18:32:56 UTC. It reproduced
all native predictions and Jacobians with zero discrepancy and verified
checkpoint, config, data, runtime and original-array hashes. Its original
summary still states that independent folding was pending; the separate
receipt below records completion without rewriting the historical receipt.

The independent checker completed once in session 93063 with exit 0. It hooks
native controller outputs, independently applies the sigmoid clock transform,
and replaces each entire FFN by buffer-only folded matrices. This differs from
the main script's constant `step_size` substitution. A fixed small-matrix
contract checks function and Jacobian equality before checkpoint use. CPU
`vmap(jacrev)` computes all complete input derivatives. Frozen source imports,
the exact 595-file tree, runtime buffers, unchanged native/folded states,
preserved non-FFN tensors and RNG-free folding are verified.

| Independent check | Maximum absolute difference |
|---|---:|
| Historical versus main native outputs/Jacobians | 0 / 0 |
| Independent native outputs | 0 |
| Independent native Jacobians | 3.41394e-14 |
| Folded versus patched outputs | 9.76996e-15 |
| Folded versus patched Jacobians | 3.06422e-14 |
| All 18 CSV rows, every risk component | 0 |

The native and folded evaluation tolerances were 1e-10 for fields and 1e-9 for
Jacobians. CSV comparisons used absolute 1e-10 and relative 1e-11 tolerances.
The source and receipt are archived under `independent-check/`; all main input
and output hashes are retained in `provenance.json`, and the final artifact
manifest covers the complete analysis bundle.

Orchestration disclosure: a wording-correction guard failed before root could
create its intended launch-review receipt, but the next tool call still
started the already authorized read-only diagnostic. Functional review,
self-test and source/protocol checks had passed; no functional check failed.
The one execution was retained. `qa/prewording-review.md` preserves the review
bytes at launch, and `qa/clock-collapse-code-review.md` corrects only a
CPU-versus-MPS description to the actual float32 aggregate comparison. The
explicitly postlaunch `qa/clock-collapse-orchestration-qa.json` documents the
sequence. It is not presented as reconstructed prelaunch approval. No source,
protocol, checkpoint, ready receipt or training result was changed.

The independent check passes, both figures were visually inspected, and the
diagnostic is complete. These measurements concern fixed checkpoint dependence,
not a retrained function class, a primary validation improvement, a new holdout,
sampling quality, runtime savings or completion of the architecture goal.
