# ODELayer isolated experiments and stage report

Goal: Dispatch two focused waves centered on original ODELayer; prepare a reproducible evidence-based stage report for the user's direction decision.

- [x] Freeze protocol: gate source crossed with K1/K2; original input gate at K4/K8; SiLU/SwiGLU concurrent and extended-compute controls. Seed0 only.
- [x] Implement time-only gate, evaluation context, planned budget checkpoints; regression and math tests.
- [x] Prepare isolated configs, Slurm waves, posthoc analysis and reproducible report builder; verify all locally/remotely.
- [x] Submit training waves and dependent evaluation/report work; record IDs, source/config hashes and actual status.
- [x] Deliver report draft with existing evidence, pending questions, decision criteria and exact paths.

Constraints: no new P constraints, no structured-Q trick, no optimizer/schedule/data change, no new seeds. Preserve old18 runs. Training run is unit; checkpoints/layers are not replicates. Technical baseline re-runs are not independent seeds. New result cells remain pending until artifacts validate.

Status: protocol setup.

Status: code, protocol and substantive report draft prepared; all local model/milestone checks pass. Preparing remote preflight and submission.

Remote preflight passed for time gate and K8; Slurm GPU and CPU report resource checks passed. Tiny full pipeline (training→checkpoint provenance→six NFE settings→time bins→gate Jacobian) passed on CPU. Frozen runtime archive created and installed;29 files hashed. Source remains uncommitted; immutable snapshot supplies provenance.

Submitted: training array20378694 (3 dataset tasks, max2 simultaneous GPUs); report job20378698 depends on afterany:20378694. Queue verified both accepted, initially pending. New results remain pending; report draft already contains prior18-run evidence. Submission receipt: submission.json.

Startup verified: tasks0/1 running on H200 gh115/gh122, CUDA and29-source hashes pass, stderr empty. Actual first-run metrics reached Gaussian step1000 and Ring8 step500. Task2 waits for max2 concurrency; report waits for array completion. These are startup checks, not final research results.

## Completion audit and monitoring follow-up
- [x] All Slurm tasks complete; pull30/30 runs and54/54 diagnostics.
- [x] Audit results, figures and training dynamics; write analysis bundle and interpreted report.
- [x] Implement persistent monitoring and submission integration; test success/failure/disconnect handling.
- [x] Sync completed artifacts and monitoring, deliver evidence and next-direction options.

Completed audit: 30 trajectories,54 diagnostic checkpoint hashes,29 source/config hashes,15 actual data files. Frozen source-manifest had local data hash; all26 selected arrays are bitwise identical to remote. Preserve frozen history and document correction. Full interpreted report plus5 figures prepared;8 stage/model/monitor/submission tests pass. Real detached monitor reached SUCCESS and exited. No new research branch submitted.
