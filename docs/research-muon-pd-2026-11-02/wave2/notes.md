# Evidence used to select wave2

First-wave endpoint records are complete but independent replay and sample-metric
recalculation were pending when this continuation was authorized. Complete those
before selecting the next intervention. Candidate questions: does Muon's update
direction matter after controlling its magnitude; and does diagonal modulation
become useful with that optimizer or a better motivated placement?

Historical signed/wider gates and controller reparameterizations already exist
under docs/research-architecture-search/. Do not relabel those as new directions
or repeat them without a specific change in evidence and a registered control.

Replay complete:36/36, generated arrays exactly equal; independent NumPy SW2
maxerror1.003e-8, float64 field accumulation maxerror4.402e-10. First-wave curves
and update spectra are now curated with three figure sets under analysis/.

Registered30new cells onRing8/Product16:8direction controls,8growth controls,
14diagonal-placement/optimizer arms. No new activation or loss. Original controls
are reused only after exact three-update native compatibility checks.

CPU checks passed onbothhosts atfa0c13b. Lenovo's first CUDA attempt exposed a
test-only CPU/CUDA device mismatch in a scalar diagnostic assertion; corrected
at539b1a4 before any native smoke or formal wave2training.

CUDA readiness found an actual inherited trainer resume failure on both hosts: RNG bytes were mapped to CUDA. Fix only `data_rng.set_state(... .cpu())`; preserve failed smoke outputs and use a new `native-smoke-rngfix` witness path. Formal queues have not started.
