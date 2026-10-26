# Units and comparison scope

All18 runs follow the user's fixed training seed0 protocol. Endpoint differences and ratios are descriptive. No between-training confidence interval, p-value or significance claim is produced. FM times, coordinates, hidden probes and successive checkpoints are repeated measurements, not independent training replicates.

Primary endpoint: random-time final validation oracle MSE. All six arms and all three tasks are retained. Common-field/Jacobian panels and sampling SW2 support interpretation; they do not replace a failed primary/baseline screen. Complete costs are recorded in the frozen budgets. Wall times include diagnostics and concurrent scheduling, so precise latency or hardware speedup claims require another measurement and are not made here.

The tied/untied initial functions match exactly, not merely through a zero output head. The tangent control matches the value/Jacobian at zero only; its initial hidden difference is recorded in covariance-initialization.json. Its changed feasible set is part of the control, not a claim of pure optimizer equivalence.

Remote audits verify source/data/configuration hashes,108 checkpoint identities,90 analytic arrays, common inputs, final validation MSE and the registered initial-tensor invariants. Local endpoint source/weight replay is a separate scope and is reported only when its summaries exist. No fresh holdout was opened for selection. The36 internal energy rows cover six layers for each tied/tangent model-task pair; they are not36 independent model successes.
