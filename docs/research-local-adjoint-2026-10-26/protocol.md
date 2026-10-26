# Cold Phase A: original encoder with its exact local adjoint

Registered after correctness/resource profiling and before any cold quality outcome or three-update pipeline smoke. The full local-adjoint FFN is the only candidate; its frozen-clock version is a mechanistic control. Keep original K2 and all four generic controls. This is a new local function class, not a continuation of the stopped full-model potential, shared-depth or P-depth settings.

## Fixed seven arms and recipe

Product16 first: original two-solve ODEFFN; local full adjoint; local adjoint with clock derivative omitted; tied SiLU ridge128 and192; ordinary SwiGLU128 and192. All use width64/depth3, unchanged time FiLM/normalization/output head and original independent cached-bank Gaussian/target pairs. Seed0,5000 updates, batch512, AdamW3e-4, weight-decay0, clip1, uniform training time. No loss, data, head, spectral, activation, initializer, slope or optimizer change accompanies the FFN intervention.

The candidate reuses the original first layer's actual initial parameters. Common surrounding tensors and buffers match exactly. The zero output head makes initial full outputs zero, but hidden functions and head parameter responses differ. Do not claim full NTK/function pairing between different FFNs. Baseline compatibility reruns must reproduce actual old saved native controls; they are not additional independent training evidence.

Preserve checkpoints at0/1000/2500/5000,500-update training summaries, all learned P spectra and local encoder/adjoint activity probes, full final validation fields, five-time Jacobians and common-source Heun32 sampling. Use the unchanged reference banks. Two one-thread CPU workers run with ten-second child CPU/memory monitoring. Lenovo is offline; no remote GPU job is claimed.

## Prospective continuation rule

Only local_full may qualify, and only if all base contrasts/guards hold:

1. Primary uniform-time oracle MSE at most90% of original ODE and95% of local_frozen_clock.
2. Either quality route: at most90% of the best of all four generic MSEs; or parameter-reuse route: at most105% of the better SwiGLU MSE, at most90% of the better tied-ridge MSE, and at most75% of SwiGLU128 parameters.
3. Complete five-panel Jacobian error and common-source SW2 at most110% of the best respective generic value. Keep all primary field and sampling metrics distinct.

A Phase-A pass only permits a separately verified Gaussian2D/Ring8 replication with the same seven settings and recipe, then fresh holdout and cost confirmation. It is not architecture acceptance. If neither route passes, stop this exact setting without longer training, coefficient/gate omission selection, width or initializer rescue. A winning generic control remains a generic result, not success of the ODE goal.

## Required preflight and audit

Four prototype derivative/parameter-reuse tests and the fixed resource screen pass. Before formal5000 updates, run all seven three-update pipeline cells and replay original/SwiGLU128/SwiGLU192 against the actual old training function at three updates. Verify complete model/optimizer/buffers and data-stream RNG where applicable, activation capture for the local helper, changed P/controller parameters, source hashes and the actual supervisor CLI. No smoke quality score selects a setting or threshold.

After formal completion independently reconstruct native fields/Jacobians/sampling metrics, source/input/checkpoint bindings, all P trajectories and the decision. Preserve all failures and earlier frozen reports. No commit/push or submitted-report edit.
