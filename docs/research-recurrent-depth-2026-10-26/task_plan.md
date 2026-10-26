# Recurrent-depth experiment plan

## Active user objective

Find a new evidence-based idea and continue experiments; use Mac mini and Lenovo concurrently when possible. Keep intermediate reasoning internal and bring the user a useful empirical result rather than another speculative explanation.

## Selected question

Can a shared hidden-state evolution built from the original ODEFFN become useful when intermediate iterations receive direct supervision? Compare this with ordinary SwiGLU evolution, independent layers, and the same supervision on ordinary dense networks. This is not a claim that parameter sharing alone is an ODE-specific innovation.

## Phases

- [x] Revalidate previous completed screens and local runtime state.
- [x] Attempt Lenovo SSH; connection timed out. Ask asynchronously while local work continues.
- [x] Specify the fixed ablation and empirical continuation criteria.
- [x] Implement isolated recurrence and loss helpers; test their derivatives, pairing and compute accounting.
- [x] Freeze and smoke the portable trainer; prepare a Windows package.
- [x] Run complete same-host controls on Mac mini; Product16 also completed locally while Lenovo remained offline.
- [x] Audit all outcomes, internal-depth behavior, P trajectories and actual costs.
- [x] Apply the continuation rule: no leads qualify, so longer training and fresh holdout are not triggered.
- [x] Record the result and stop this setting; the broader research goal remains active with the anchored-corrector screen.

## Boundaries

- Training seed0 only. Preserve all previous frozen runtimes and submitted Report1.
- Keep the professor's scalar K2 solves and middle SiLU. Change whole-network sharing and the placement of supervision, not spectral constraints, activations, optimizers, gates or P initialization.
- Use the strong established uniform-time direct-velocity recipe. The stopped prediction-space bridge is not combined with this new axis.
- Every host runs all candidate/control arms for any assigned dataset. Do not infer a model effect from candidates on one host and baselines on the other.
- Each host uses two one-thread workers with process/memory monitoring. The tiny-model CPU path is the initial execution choice based on prior measured throughput; no GPU reservation is required.

## Goal-turn classification

Initial turn: progress. Authoritative prior results and local idle state were checked; Lenovo SSH failed with a verified timeout. There is meaningful local work available, so the goal is not blocked.

## Status

All 42 registered cells completed on Mac mini. Independent42-cell audit passes; none qualify. Four figures and conditional paired evaluation summaries are complete. The setting is stopped; Lenovo remains offline.

## Preparation notes

- Eight focused test methods pass, including exact native-forward compatibility, cloned/shared functions and input derivatives, summed tied-parameter gradients, normalized supervision, actual repeated-call costs, checkpoint structure and first-order fixed-horizon refinement.
- Static review caught a metadata-key collision between the training supervision label and inference-budget label before any preflight training. Budget metadata now uses readout_mode. The unused first draft source is preserved; source-v2 is the preflight candidate.

## 2026-10-26 continuation

Previous goal turn: progress. A primary-paper review was completed and source-specific counterfactuals were extracted; it clarified that looping plus supervision alone does not establish an ODE advantage. Current authoritative state: Gaussian/Ring8 group completed 28/28 at 2026-10-25T17:51:10 UTC; Product16 group completed 14/14 at 17:57:16 UTC. No recurrent-depth training process remains. Source-v2 is unchanged. Tailscale peer Lenovo is offline, last seen 2026-10-25T11:20:00 UTC. Analysis launched from saved arrays/checkpoints, without training changes.
