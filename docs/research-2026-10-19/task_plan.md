# ODELayer research and SwiGLU baseline

Goal: Establish a fair SwiGLU baseline, assess existing evidence, and derive a falsifiable mathematical research direction centered on ODELayer.

- [x] Verify remote state and applicable instructions; inspect official SwiGLU implementation.
- [x] Add parameter-matched SwiGLU to the existing toy protocol, verify unchanged ODE behavior, and run seed-0 comparisons if remote access permits.
- [x] Derive generator, gate, truncation and external-flow error relationships; validate key algebra numerically.
- [x] Write research question card, evidence notes, concrete next experiment and status.

Decisions: seed 0 only; preserve historical runs; existing 15 ODE/FFN runs serve as first comparisons. New baseline uses the same toy backbone and protocol, not a claim of full JiT reproduction. New architectural candidates require clear mechanism before training.

Status: Inspection.

Checks: local legacy models bitwise unchanged for both K values; SwiGLU arithmetic and nonzero gradients pass; parameter count 81792 vs ODE 82056. Existing 4 model tests pass. Preflight initially rejected missing CLI choice; fixed by adding swiglu without altering training loop. Math Jacobian and determinant checks pass in float64. Remote search initially used wrong completion filename COMPLETE.json and wrong image model root path; corrected to COMPLETE and src/.

Remote preflight and sbatch test-only passed. Test-only ID 20314109 is not an actual training submission. Full design and research card written; prospective structured generator remains unimplemented, pending mechanism diagnosis.

Submitted SwiGLU array 20314159 (3 datasets, seed0, 5000steps, GPU types a100|l40s|h100|h200). Training outputs use new experiment ID; old15 preserved.

Diagnostic array 20314171 submitted with afterok:20314159 and kill-on-invalid-dep. Last status: baseline PENDING(None), diagnostics PENDING(Dependency); no completed new results yet. Pure-rotation counterexample also invalidates a recent preprint's discrete Euler bound; documented and not used as a theorem.

Progress: Gaussian and Ring8 SwiGLU reached step5000. Final SW2 0.07461066 / 0.20801781; oracle MSE 0.00625680 / 0.01680949. Await Moons and diagnostic runs before complete interpretation.

Moons SwiGLU started on A100 ga019. Time embedding frequency1..1000 identified as a potential common numerical confound; documented, not changed. Diagnose128→256 convergence before interpreting numerical reference.

Final status: COMPLETE. All3 SwiGLU jobs and all3 diagnostic jobs completed0:0. Unified18-run summary verified. Full mechanism results in results.md; all three JSON diagnostic files downloaded. New structuredP remains a design, not a trained result. Recommended next first ablation is gate input vs time-only, preservingP/K, followed by same-family exact-vs-Taylor. README and synthetic data docs corrected to remove obsolete untrained/multiseed claims.
