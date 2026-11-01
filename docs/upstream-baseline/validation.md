# Documented validation

This is a source-copy check, not a training run or image-quality result.
The frozen author reference is JiT commit
`cbc743a2ada5e9762697da2c83f8c4f8379e8c17` at
`/Users/mr.z/ode-workspace-prep/upstream-review-20261101/jit-official`.
The test verifies its file hashes before executing reference calculations.

The separate environment is
`/Users/mr.z/ode-workspace-prep/.venv-upstream-jit`; its declarative spec is
[env-spec.json](env-spec.json). Existing experiment environments are unchanged.
The original pip Git transport failed with earlyEOF; the same author commit
was installed from its pinned codeload archive instead. No source/version
substitution was made.

A fresh agent reads this document, README, UPSTREAM.md and the provider ledger.
It runs each exact command once, sequentially. Do not inspect/repair source,
change arguments, retry failures, install packages, or launch a benchmark.
Consume actual terminal exits. Report documentation discrepancies and results.

```sh
/Users/mr.z/ode-workspace-prep/.venv-upstream-jit/bin/python -B /Users/mr.z/ODE-FFN-upstream/main_jit.py --help
```

Expected: exit0, normal argparse help, including `--ffn {swiglu,isru}`.

```sh
/Users/mr.z/ode-workspace-prep/.venv-upstream-jit/bin/python -B /Users/mr.z/ODE-FFN-upstream/tests/check_upstream.py --upstream /Users/mr.z/ode-workspace-prep/upstream-review-20261101/jit-official --output /Users/mr.z/ode-workspace-prep/upstream-review-20261101/documented-check.json
```

Expected: exit0, `UPSTREAM_COMPATIBILITY_PASS`, PASS JSON. The original
CUDA-only reference receives placement-only adaptations for CPU. Default
state/RNG, active full forward/input/parameter gradients, the actual epoch's
Adam update and both EMAs, and Euler/Heun outputs must match bitwise on CPU.
The candidate must preserve non-FFN tensors/RNG, pass gradcheck, save/load the
upstream checkpoint schema, and perform a brief real MPS update when available.
The authors' deprecated AMP call may emit a FutureWarning; this is expected
and is retained to avoid an unrelated source change.

The fixture has two small blocks and32features. It is test input, not a
selected CIFAR architecture or training recipe. The full B/2 port is not
benchmark-trained here. FID computation is delegated to the pinned author
fork, and actual FID/convergence remain unverified.
