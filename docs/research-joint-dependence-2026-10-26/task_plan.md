# Task plan: Test whether the ISRU observation transfers beyond independent signs

Goal: turn the new ordinary-ISRU observation into evidence for the next ODE design question, without reopening the stopped cubic-field screen.

- [x] Inspect the completed cubic-field report, its stop rule, current Git state and working conventions.
- [x] Identify a controlled joint-law contrast: full independent-sign Product16 versus a symmetric Hadamard sign-code mixture with the same latent marginals, covariance and radial laws.
- [x] Register the exact distribution, target-geometry derivation, controls and descriptive decision before outcomes.
- [x] Verify the distribution/oracle, source coupling, native model initialization and resume; freeze code and inputs.
- [ ] Run one monitored five-family transfer screen, explicitly reusing the five Product16 references.
- [ ] Independently audit outputs/streams/derivatives, report every result and push the completed work to GitHub main.

Constraints: model seed0 only; unchanged original ODE, ordinary free ISRU and SwiGLU64/128/160; no cubic-candidate continuation, coefficient/clock/order/width/optimizer rescue or teacher training. Preserve all old artifacts and Report1. No subagents. At most two one-thread CPU workers on the Mac; Lenovo connection has again been probed and must be checked through its original session handle.

Previous goal turn: progress. It completed and sealed the bias and cubic studies, pushed current code/evidence to GitHub and synchronized Mac mini, torch and MBP at58849ec. The research goal remains active and unachieved.

Status: preflight passes exact sign-marginal, radius, native-oracle, Jacobian, five historical initialization and five resume checks. The fixed noncommuting-Jacobian witness has norm4.0947. Ready file f6e789bd7315766e49d749011e702a913204dc1031b0f119b973092bcb5843a9 binds173 inputs and158 prior documents. Publish verified protocol/code before the single launch; no new training outcomes exist yet.
