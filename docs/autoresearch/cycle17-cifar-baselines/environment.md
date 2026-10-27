# Mac mini image environments: documented validation commands

Provider ledger: `/Users/mr.z/ODE-FFN-autoresearch/.aris/compute/macmini.md`.
The tracked JSON specs define the packages, phases, kernels and cache.
The training environment is reused unchanged from cycle16. The metrics
v2environment is separate; the first failed SciPy1.15.3environment remains
preserved. The v2pin is SciPy1.18.1, already known to load on this macOS.

Run both commands below verbatim from `/Users/mr.z/ODE-FFN-autoresearch`.
They exercise actual kernels on seeded synthetic inputs, not CIFAR training.
Each must exit0and print a line starting `WITNESS_PASS`. Do not fix an error
inside the fresh-agent pass; record it as a documentation/environment mismatch.

Training environment: actual JiT forward/backward, CPU, output shape2x3x32x32,
positive P gradients. The synthetic and training environments are not modified.

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python /Users/mr.z/ODE-FFN-autoresearch/docs/autoresearch/cycle17-cifar-baselines/environment_witness.py --mode training
```

Metrics environment: unmodified torch-fidelity Inception compatibility features,
MPS, output shape2x2048. Its own loader uses the pre-staged91MBweights cache
at `/Users/mr.z/ode-workspace-prep/image-bridge-inception/checkpoints/`.
The sentinel includes the actual weight SHA256 and nonzero feature sum.

```bash
/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python /Users/mr.z/ODE-FFN-autoresearch/docs/autoresearch/cycle17-cifar-baselines/environment_witness.py --mode metrics
```

An import-success check alone is insufficient. Parent and fresh-agent witnesses
must both pass before the environment is declared ready. These checks do not
verify model quality, full training runtime or cross-backend feature agreement;
the separately registered metric fixture and trainer preflight cover those.
