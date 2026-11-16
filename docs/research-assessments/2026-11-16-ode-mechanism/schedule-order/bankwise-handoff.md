# CPU sampling-batch handoff

The native32,768-source CPUbatch took134seconds at inner4/Heun128.
The optional4096-source path retains exactly the same models, sampler,
source order and complete banks. All four trained-resolution endpoint arrays
match the native reference exactly. Fresh documented entries pass all sixteen
bounded checks. The new source is published/frozen at303d213.

The measured complete new-entry first row is78seconds while the old workers
continue; the old row was134seconds. Inner16/Heun128is259versus530seconds.
These are observed CPUwall times under different concurrent loads.

## Running sources and evidence

- Original: frozenfcbe347, `schedule-order/observer-repo`, queue55052,
  `numerical-full`. Preserve every completed row and any partial output.
- New: frozen303d213, `schedule-order/bankwise-repo`, queue56007,
  `numerical-bankwise`. Eight tasks, twenty unique full-population cells;
  no rows are replaced with old-backend rows in the new comparison.
- `bankwise_audit.py` is read-only. It validates source freezes, process
  advancement, every retained old prefix, all32768endpoint values and all
  pooled/bank scores. Missing replacement cells make retirement ineligible.

From the active repository, use a NEW output name for each audit:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 studies/muon_pd_20261102/coefficient_20261116/bankwise_audit.py --study /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order --output /Users/mr.z/ode-workspace-prep/coefficient-20261116/schedule-order/handoff-audits/prefix-NNN.json
```

`eligible_to_retire_old` must be true before changing the original execution.
The output's `passed` field alone means observed overlapping rows agree; it
does not imply that missing rows or the complete sweep have finished.

## Retirement procedure

1. Re-read both actual queue states and verify the new workers are live and
   advancing. Require a fresh eligible all-prefix audit, not an old receipt.
2. Capture the original queue and current children: UID, PID, PPID, process
   start time and complete command. Each must belong to this user and the
   exact `observer-repo`/`numerical-full` invocation. Never use a broad kill.
3. Briefly suspend only that queue and those verified active children to
   stabilize the old snapshot. Re-run the audit. If any new old row is missing
   or an identity differs, immediately resume our queue/workers and retain them.
4. Only after the stable audit passes, terminate the verified old inference
   children and continue them so the termination signal is delivered. Resume
   their parent to collect actual exits. The native queue must retain its
   failed/unstarted receipt for this intentional retirement; do not rewrite it
   as a successful full run. If the workers already finished, record that fact
   and retire only the verified idle queue rather than inventing child exits.
5. Record all identities, audit hashes, sent signals and actual terminal
   state in a separate intentional-handoff receipt. Confirm old processes are
   absent and new workers still advance. Preserve the old raw directory.

No GPU0-3or foreign job is used. All permitted145cards have resident contexts
at the latest inspection. No old CPUworker is held for a merely waiting route.
Independent full numerical collection follows completion of the new eight
tasks; this routing operation does not itself establish a model advantage.
