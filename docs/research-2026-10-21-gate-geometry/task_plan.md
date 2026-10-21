# Gate geometry diagnostic

Goal: evaluate the reassessment's gate mechanism using existing final checkpoints on 145. No model training.

- [x] Freeze scientific protocol and implement diagnostic.
- [x] Verify analytic oracle Jacobians and anchored-gate derivatives locally.
- [x] Smoke-test/time on 145; launch with persistent monitoring and frozen inputs.
- [x] Audit outputs and write evidence-bounded findings.

Scope: the latest CPU-trained seed-0 study, step 5000, Gaussian anisotropic and Ring8; input/fixed/scalar/SiLU/SwiGLU. Compare complete velocity derivatives on common validation interpolants and model-specific trajectories. A derivative decomposition is not a retrained model or a generation-quality ablation.

Status: complete. The run finished in 8.76 seconds on one CPU thread; all panels audited. No new training direction selected.
