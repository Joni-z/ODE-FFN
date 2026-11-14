# Task: resume native SwiGLU baseline to160epochs

User authorized14November2026: baseline must also continue, matching K1/K8.
Restore the original native checkpoint100, seed0, fixed recipe; evaluate
FID50k120/140/160. Preserve original100epoch source/checkpoint/FID evidence.

- [x] Verify baseline checkpoint SHA/state and native-math/initialization
      compatibility with the validated periodic-evaluation runtime17b61a5.
- [x] Add a minimal baseline continuation recipe and register source/outputs,
      budget, tests and checkpoint provenance.
- [x] Preserve original100checkpoint, stage continuation state, validate
      documented Slurm command and submit one priority job.
- [x] Attach baseline monitor and publish launch records.
- [x] Verify allocation and wrapper resumes after100epochs.
- [x] Confirm native optimizer load and epoch101 updates after startup.

ODE K8 remains authorized through160; no architecture or optimizer change.
Baseline runtime bookkeeping may change for periodic FID, with native math
and initial/resumed state checked explicitly.

Compatibility PASS: source65864ad ->17b61a5 preserves native math, initial
state/RNG and real-checkpoint full128px predictions bitwise. Optimizer step
63500 retained; original checkpoint SHA matches. Periodic-eval RNG checks
previously passed in this unchanged environment.

Baseline23117422 submitted from verified staged100epoch state; monitor2838752
active. Scheduler reasonQOSGrpGRES at last check. Duplicate slow file copy
was stopped and replaced by a hard link to the separately copied anchor;
atomic future checkpoint replacement preserves the anchor. Both staged SHA
and original SHA match. The private incomplete copy was removed after
validation; original scientific records were never modified.

Runtime confirmed on H100gh012: epoch101 completed, warmed updates~.057s,
GPU93–97%, loss~.0299. Baseline continuation active under job23117422.

Final inspection confirmed106completed epochs and67310optimizer steps,
continuing beyond63500. An initial inspection of the live replacing last
checkpoint hit ESTALE; a temporary hard-link snapshot pinned the complete
inode for verification. Training remained normal; the temporary link was
removed after collecting the proof.
