# Follow-up diagnostic stage

- [x] Specify frozen-order intervention and controlled-matrix hypotheses.
- [x] Preserve seed 0, existing checkpoints, existing data, and baseline controls.
- [x] Implement and validate polynomial/exponential actions and exception-safe intervention.
- [x] Complete CPU-only controlled-operator calculations.
- [x] Freeze and verify 17 source files, dataset files, and 18 checkpoint/configuration pairs.
- [x] Submit one-GPU job 20405164 and start a detached monitor.
- [x] Observe the actual GPU benchmark, worker choice, utilization, and evaluation progress.
- [x] Confirm 66 completed cells, three successful workers, and a successful standalone runner.
- [x] Collect the complete archive and independently audit its local sample hashes and native controls.
- [x] Analyze within-checkpoint effects and update the evidence report on CPU.

Both GPU evaluations are complete and locally audited. Analysis and Report 1 include the results.

## Alternative host preparation

- [x] Audit host 145 GPU occupancy, environment, and storage.
- [x] Copy only the needed runtime, checkpoints/configurations, and small dataset version.
- [x] Verify all input/source hashes and pass four CPU intervention tests on 145.
- [x] Pass six local routing tests without allocating GPUs.
- [x] Release one GPU through its owner's provided helper, with explicit user authorization.
- [x] Restore SSH access to 145 through MBP and refresh the controller.
- [x] Recheck availability, complete GPU numerical preflight, and cancel the pending original Slurm job.
- [x] Start and monitor one execution route; confirm remote completion of all 66 cells.
- [x] Start a detached result collector after the gateway connection was lost again.

The original job was cancelled before full standalone execution. Host 145 completed the stage in approximately 114.5 seconds. The completion monitor captured success; a later gateway outage delayed result transfer. See migration-145.md.

## Report completion

- [x] Analyze the full 66-cell matrix and fixed-input probes.
- [x] Run the focused 48-cell Ring8 solver check without new training.
- [x] Verify all solver-check samples and native array reproduction.
- [x] Produce real figures, exact tables, claim boundaries and a statistical appendix.
- [x] Update Report 1 with three main-text pages and supporting visualizations.
