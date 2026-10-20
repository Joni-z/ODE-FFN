# Evidence and implementation log

Prior results: ../research-2026-10-19/results.md and diagnostics/*.json, 18 complete seed0 models. Gate correction measured nonzero; cap reduces internal J without consistent generation benefit; Heun32→256 does not remove main ranking differences; closer-to-exponential K2 is not universally better than K1.

Current authorization: user requests dispatch necessary experiments, retain decision over continued directions, and prepare a stage report supported by evidence and reasoning. No combined tricks. Only existing seed0 preference remains.

Implementation checks: all legacy branches remain bitwise-equal at identical state/input; all6 ODE gate/order combinations share identical initial state_dict. Time-only gate fixed-context Jacobian equals frozen Taylor polynomial; logged gates use actual time embeddings, not hidden inputs; gate input derivatives are exactly zero. Extra budget checkpoints leave final model/data RNG bitwise unchanged on a short CPU training test. Seed0 used in new tests/runs.

Budget accounting: total forward linear MACs checked via Linear hooks plus explicit functional P applications. K1/2/4/8=80576/105152/154304/252608; SiLU80192; SwiGLU80000. New30run/54checkpoint protocol frozen before submission.

SSH initially rejected authentication; user restored login. No new training was submitted before restoration.

## Actual submission

Training array:20378694; CPU report job:20378698, dependency afterany:20378694. Both accepted and verified in squeue. Frozen29-file runtime checked before submission.30 training trajectories,54 diagnostic checkpoints, seed0 only; no new P constraint. New-results report remains pending until validated artifacts exist.

Startup verified: tasks0/1 running on H200 gh115/gh122, CUDA and29-source hashes pass, stderr empty. Actual first-run metrics reached Gaussian step1000 and Ring8 step500. Task2 waits for max2 concurrency; report waits for array completion. These are startup checks, not final research results.

## Completion follow-up
All30/54 complete; full descriptive analysis and interpreted report written,5 PNG/PDF figures visually inspected. Strongest finding: K4/K8 SW2 changes <0.2% despite63.7% higher forward-linear MAC; no uniform gate/budget benefit. P sigma_max rises and stable rank falls. Single seed only; no inferential tests.

Audit issue resolved: source snapshot annotation used local data-manifest hash while runs used actual remote hash. All selected12 NPZ files/26 array payloads are bitwise identical. Archive bytes, generator inventory and tiny Ring8 moment metadata differ. Added correction artifacts; future preflight now checks data manifest. Frozen runtime left unchanged.

Monitoring: detached local SSH poller every60s, stall hint after15min, failure/missing-artifact checks, bounded24h, no auto-resubmit or external messages. Current complete jobs passed real end-to-end check. Main test command first used systemPython lacking torch; rerun with .venv-test succeeded (8 tests). ResourceWarning from monitor lock lifetime fixed with context manager.
