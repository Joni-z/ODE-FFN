# Frozen transport diagnosis: execution record

Completed 4 November 2026 on Lenovo's RTX 4060 Laptop GPU0, using the existing
synthetic environment. Numerical source was clean and detached at
`6fe9e7fe5242e66f86f4023282f21122b2a7422e`. No packages were installed.

## Readiness and run

The [documented invocation](cycle15-validation.md) passed once in a fresh
documentation-only witness. Its [receipt](readiness/cycle15/documented-witness.json)
records all six physical checkpoint replays, exact native 4,096-sample outputs,
exact helper rollouts, switch-policy anchors, finite model Jacobians and
unchanged tensors. The analytic oracle Jacobian fixture error was 9.85e-9.

Runtime root: `D:/Research/ODE-FFN/professor-transport15-20261104/`.
Source, common reference and formal outputs are separate `repo/`, `common/`
and `output/` directories. The native queue used two one-thread CUDA workers;
all six cells completed with exit 0, no unstarted or failed cells. These are
frozen field diagnostics; there are no new training trajectories. A fresh
post-completion GPU process query found no Python compute task on Lenovo.

Each checkpoint used the preselected cycle14 bank0, native Heun128 and 33
common-path profile times. Saved results contain 198 profiles, 768 step rows,
48 endpoints and six full 129-state learned paths. Oracle128/256/512 reference
endpoints were also retained. Independent replay gives zero residual for the
discrete step identity and at most 2.78e-17 SW2 error. The actual common-state
oracle finite-difference Jacobian check is 6.57e-10.

## Sealed artifacts and analysis

The [runtime seal](cycle15-results/runtime-seal.json) covers 95 files; all 366
cycle14 parent files were rehashed unchanged. The 72,270,540-byte `runtime.zip`
is stored outside Git on Lenovo and locally under
`/Users/mr.z/ode-workspace-prep/muon-pd-20261102/nov04-transport15/`.
SHA256: `ff1acce5672c701136e848322e3b0917780114d6b7c171e4a7686f931aafaedf`.

The source-bound NumPy analysis retains all cases. The bank0 reversal led to
registered saved-array follow-ups: all eight pools/omissions, component counts,
sector crossings and transformed-sample SW2. These follow-ups made no model
calls and did not change the primary estimator. Six figures were visually
reviewed, including the source-bank sensitivity and shape/mass panels.

Read the [analysis and tables](cycle15-results/analysis-report.md),
[statistical scope](cycle15-results/stats-appendix.md), and
[mode-mass derivation](mode-mass-derivation.md). The next proposed test calibrates
boundary flux under weak oracle/model interpolation; it has not been launched.
The completed Report2 PDF/TeX and all Report1 files stay unchanged.
