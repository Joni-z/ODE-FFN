# Local primal/adjoint ODEFFN prototype

Active goal: find useful ODELayer designs and run justified experiments, using both hosts if possible. Current evidence localizes the large direct Product16 gap to conditional own-coordinate fitting. A full-model adjoint readout had improved that component, but its setting remains stopped. This new question changes the local FFN function rather than tuning that readout, recurrence or P-depth.

- [x] Derive one-original-layer primal/adjoint map and identify the known tied-ridge prior art and negative limitations.
- [x] Verify values, mixed state/parameter derivatives, local gradient identity, actual whole-model finite differences and preserved common initialization. All four tests pass in the worktree and exact37-file frozen prototype source.
- [x] Measure exact parameters, executed inference/backward matrix work and isolated CPU timings with all seven controls. Candidate58899 parameters,106752 inference MACs,317120 loss/backward MACs; all three fixed resource conditions pass.
- [x] Specify a separate cold comparison with fixed controls and no quality outcomes used for selection. protocol.md registers7 Product16 arms and strict quality/parameter routes.
- [x] Verify the new supervisor and all seven three-update pipelines against actual old native controls. Six0/3-step native model/optimizer/buffer/RNG states match the old training function bitwise; all15 candidate encoder parameter tensors move and all3 local visits are captured.
- [x] Execute monitored Phase A and independently audit its complete outcomes and continuation rule. Seven cells complete;28 checkpoints,12 historical control states,7 field/35 complete Jacobian/7 sampling replays and24 local VJP/gradient checks pass. Independent screen agrees: no continuation.

Preserve original sources, reports, learned weights and dirty work. Seed0 only. Lenovo remains offline. No git commit/push.

Profile result: candidate has70.23% of original whole-model parameters and99.82% of its counted forward/backward matrix work. Median batch512 CPU forward/backward timings are4.34%/7.89% slower than original and remain substantially slower than same-MAC SwiGLU192. This is a parameter-reuse candidate, not a latency advantage. Frozen prototype and profile remain unchanged when preparing the new training source-v2.

Formal seven-cell Product16 Phase A is now launched from163 frozen source files with two one-thread workers and actual child monitoring. Source/protocol/recipe remain unchanged after the three-update checks. This is not an accepted architecture or a completed result. The preflight supervisor's handle was absent after its terminal COMPLETE state; all seven terminal cell states and the actual old-function comparisons were inspected rather than rerunning it.

Actual formal supervisor status is now COMPLETE7/7; every cell is COMPLETE at5000 and every recorded child PID is absent. Formal session1616 was polled through all seven completions; no restart occurred. Raw candidate/parent/omitted MSEs are.190290/.325527/.343629, while tied128/tied192/Swi128/Swi192 are.179585/.135191/.092326/.070423. The candidate also misses the derivative/sampling guards, so do not launch other tasks while the independent audit is completed. A system-Python status check lacked psutil; the existing venv status check verified the actual terminal processes instead. This was inspection tooling, not a training failure.

Complete/audited negative Phase A. Source-v2 remains unchanged. Both resource and base-ODE contrasts pass, but quality/parameter routes and Jacobian/SW2 guards fail. No other-task training, gate omission selection, longer fitting or coefficient/width rescue. The broader useful-ODELayer goal remains active and unachieved. Two figures and the evidence report are complete.
