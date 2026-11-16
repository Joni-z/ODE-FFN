# Independent all-bank confirmation of the output-response association

Registered after the selection observation and before any confirmation derivative
evaluation. Selection used128pairs: focal point directions agree with sampling,
but both paired input-bootstrap intervals include zero. Ordinary Product16
SwiGLU has better sampling despite higher average Jacobian error. This follow-up
tests the focal association only, not a universal response-error explanation.

## Frozen question and population

Keep the same primary metric, five times, 20k weights and expected two task
directions from protocol.md. Evaluate only palindrome and alternating on each
task. Use **every one of the8x4096 existing confirmation source/target pairs**,
all five times per pair:163840states per checkpoint, four checkpoints total.
No new data are sampled. Keep all sources and banks; no adaptive enlargement,
checkpoint selection, favorable-time selection or new response is allowed.
The earlier128-pair selection result remains intact and is not replaced.

Use the eight bank means as paired uncertainty units, keeping all4096pairs and
five times in each bank together. Report all eight differences, their SD,
10000 paired-bank bootstrap draws with RNG0,95% percentile intervals and97.5%
two-contrast intervals. Also show the all-source point gap. A contrary direction
rejects the association on that task; an interval spanning zero leaves precision
unresolved. Confirming it would still require a causal intervention or trajectory
test before an architectural conclusion. No new training, image or density run.

## Numerical and execution controls

The only new execution path chunks complete Jacobians in128-point batches.
Its first128-point output must reproduce the prior unchunked derivative exactly;
repeat a batch at a different size in the readiness check. Preserve FP32 native
anchors, runtime buffers, FP64 conversion, finite-difference JVPs, tensor hashes
and all raw derivative arrays. The independent NumPy summary verifier uses
the same full population and checks every scalar summary.

Use the same warm one-thread Mac CPU environment. One worker per task, at most
two workers; no GPU is needed. The source freeze and fresh documented witness
must pass before full dispatch. Retain process IDs/logs/actual terminal exits.
Raw arrays stay outside Git; publish the four-case/every-bank summaries/hashes.
