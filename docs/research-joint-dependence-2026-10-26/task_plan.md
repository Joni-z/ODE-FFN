# Task plan: Test whether the ISRU observation transfers beyond independent signs

Goal: turn the new ordinary-ISRU observation into evidence for the next ODE design question, without reopening the stopped cubic-field screen.

- [x] Inspect the completed cubic-field report, its stop rule, current Git state and working conventions.
- [x] Identify a controlled joint-law contrast: full independent-sign Product16 versus a symmetric Hadamard sign-code mixture with the same latent marginals, covariance and radial laws.
- [x] Register the exact distribution, target-geometry derivation, controls and descriptive decision before outcomes.
- [x] Verify the distribution/oracle, source coupling, native model initialization and resume; freeze code and inputs.
- [x] Run one monitored five-family transfer screen, explicitly reusing the five Product16 references.
- [x] Independently audit outputs/streams/derivatives and report every result.
- [x] Finalize artifact checks and prepare the completed evidence for GitHub main.

Constraints: model seed0 only; unchanged original ODE, ordinary free ISRU and SwiGLU64/128/160; no cubic-candidate continuation, coefficient/clock/order/width/optimizer rescue or teacher training. Preserve all old artifacts and Report1. No subagents. At most two one-thread CPU workers on the Mac; Lenovo connection has again been probed and must be checked through its original session handle.

Previous goal turn: progress. It completed and sealed the bias and cubic studies, pushed current code/evidence to GitHub and synchronized Mac mini, torch and MBP at58849ec. The research goal remains active and unachieved.

Status: complete. Five new fits completed once in56.1 seconds through session65081/PID7005. Audit session65333 passed in8.2 seconds. The ordinary-ISRU transfer criterion fails. Interaction check session50876 verifies exact target ranks16/136 and a scalar-clock counterexample; it is not a complete-network lower bound. All10 results, six figures and post-outcome generation cases are reported and checked. Protocol/code were published before training atc5f0eaf; the completed evidence is ready for a normal main push.
