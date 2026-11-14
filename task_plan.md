# Task: resume native SwiGLU baseline to160epochs

User authorized14November2026: baseline must also continue, matching K1/K8.
Restore the original native checkpoint100, seed0, fixed recipe; evaluate
FID50k120/140/160. Preserve original100epoch source/checkpoint/FID evidence.

- [x] Verify baseline checkpoint SHA/state and native-math/initialization
      compatibility with the validated periodic-evaluation runtime17b61a5.
- [x] Add a minimal baseline continuation recipe and register source/outputs,
      budget, tests and checkpoint provenance.
- [ ] Preserve original100checkpoint, stage continuation state, validate
      documented Slurm command and submit one priority job.
- [ ] Attach monitor and verify allocation/resume state; publish GitHub/Torch.

ODE K8 remains authorized through160; no architecture or optimizer change.
Baseline runtime bookkeeping may change for periodic FID, with native math
and initial/resumed state checked explicitly.

Compatibility PASS: source65864ad ->17b61a5 preserves native math, initial
state/RNG and real-checkpoint full128px predictions bitwise. Optimizer step
63500 retained; original checkpoint SHA matches. Periodic-eval RNG checks
previously passed in this unchanged environment.
