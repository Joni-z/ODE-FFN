# Follow-up diagnostic stage

- [x] Specify frozen-order intervention and controlled-matrix hypotheses.
- [x] Preserve seed 0, existing checkpoints, existing data, and baseline controls.
- [x] Implement and validate polynomial/exponential actions and exception-safe intervention.
- [x] Complete CPU-only controlled-operator calculations.
- [x] Freeze and verify 17 source files, dataset files, and 18 checkpoint/configuration pairs.
- [x] Submit one-GPU job 20405164 and start a detached monitor.
- [ ] Observe the actual GPU benchmark, worker choice, utilization, and evaluation progress.
- [ ] Validate all 66 cells and Slurm exit status.
- [ ] Analyze within-checkpoint effects and update the evidence report on CPU.

Submitted work is still pending at the initial status check. New GPU results must not be reported as completed evidence.

## Alternative host preparation

- [x] Audit host 145 GPU occupancy, environment, and storage.
- [x] Copy only the needed runtime, checkpoints/configurations, and small dataset version.
- [x] Verify all input/source hashes and pass four CPU intervention tests on 145.
- [x] Pass five local routing tests without allocating GPUs.
- [x] Release one GPU through its owner's provided helper, with explicit user authorization.
- [ ] Restore SSH access to 145 and refresh the controller.
- [ ] Recheck availability, complete GPU numerical preflight, and reconcile the original Slurm job.
- [ ] Start and monitor one execution route; validate all 66 results.

The migration coordinator was stopped after Slurm rejected a hold. The subsequent 145 SSH session expired. No full standalone evaluation has started; the original torch job remains queued. See migration-145.md.
