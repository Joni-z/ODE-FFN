# Evidence and execution notes

Professor suggestions: inspect Muon and its effect on spectra; test input-dependent diagonal weights in P D(h), and other simple parameterizations.

Official Muon source: https://github.com/KellerJordan/Muon at f98f1cacc0263b04290753e32be8d498c1efc806. The approximate Newton–Schulz polar update applies to momentum/gradient, not the learned parameter itself. Keep native five iterations/BF16/momentum.95 and apply standard AdamW to the remaining parameters.

Prior PD K1 study: docs/research-architecture-search/analysis-rate-order/analysis-report.md. PD worsened Ring8 and Product16 field error under its5000step AdamW screen; no new PD discovery is claimed. The new structure question retains K2 with frozen initial-state D: A=PD or DP and T2(A)=I+A+A²/2, including the noncommuting D/P cross terms. Existing original scalarK2 and ordinary SwiGLU remain controls.

The new user instruction authorizes this study on145/Lenovo. It does not authorize deleting old runs, modifying submitted Report1 or resuming the long inadequate image baseline. Local Mac use is orchestration/documentation only; training and kernel benchmarks run remotely.

Native documentation witness caught a Python module-name collision: queue.py shadowed the standard-library queue imported by torch2.8. Renamed the controller to run_queue.py before training; no formal cells had launched.

Lenovo's first source check caught Git autocrlf rewriting pinned Python files.
The normalized hash matched but raw bytes did not. Added study-local LF
attributes and refreshed the clean new checkout; verification still requires
the exact registered bytes. Historical worktrees were not touched.
