# ISRU CIFAR-10 continuity notes

Report2 work; Report1 is frozen at994e660. User requests actual CIFAR-10
training of the existing ISRU design, with continued development.

The transferred response is `W2 * (W1*h / sqrt(1+1.2*(W1*h)^2))`, with two
independent bias-free square matrices. Synthetic initialization uses the
original P-derived `I+0.3P+0.045P^2` for both matrices. The strong Product16
field-fit result is task/budget-specific; preserve all other task outcomes.

The existing local image environment is
`/Users/mr.z/ode-workspace-prep/.venv-image-bridge/bin/python`, registered
as image-training@475428fb. The metrics environment is
`/Users/mr.z/ode-workspace-prep/.venv-image-metrics-v2/bin/python`,
image-metrics@dc729a50. Warm specs must be checked, not rebuilt casually.

Mac mini is an M1 with16GiB unified memory, about51GiB free storage at initial
inspection. Lenovo alias query returned SSH exit255 (connection timeout);
no job was launched there. Mac mini MPS is the current execution route.
No MBP/torch dependency and no external messages.

Keep all new raw data/checkpoints/logs outside Git and give the new benchmark
its own source/settings snapshot. Do not silently replace the old evaluator,
claim the old baseline was adequate, or count protocol preparation as a
verified training launch.

## Registered execution choice
The user selected Mac mini only. The official JiT source supplies the time
law, velocity-weighted clean-prediction loss, LR scaling, warmup/constant
schedule, betas, EMA and last-Euler sampler. The CIFAR model shape,50k budget
and clip1 are explicit adaptations. Full registration is in README/config.
Batch512 hardware calibration: ISRU2.026676s, matched1.980124s,
wide2.159246s; about6.166s/cohort and5.58GB maximum driver allocation.
Batch128/manual-attention calibration was also retained; no quality data
was used to select the execution batch.

Author checks in `oct29-isru-cifar10-author-v1` both exit0:
training66345/122956, metrics71119/270561. ISRU/native SwiGLU derivatives,
active attention, actual real-CIFAR MPS updates/EMA/validation/sampler and
physical resume pass. CPU stream state is bitwise restored; maximum tensor
resume difference2.17e-19. This is not a quality result or a substitute for
the fresh documented pass. Warm environments were not modified.

Documentation edit: one combined patch failed on an absent notes line;
the corrected targeted patch was applied. No experimental source/runtime
was altered by that failed patch.

## Formal launch
Source6bffa70 was published, then frozen in `oct29-isru-cifar10-v1` with
bindings SHA2564df78d20e3eb786e7e01ff7ecb4c81f5835f6726f53dd1354bb8d0e4a5fdb396.
Fresh documented native77532/ccea00 and metric68393/b96d97 each exit0 once.
No invocation mismatch. The metrics environment naming alias refers to the
same spec/interpreter. Full receipts are in readiness.json/private runtime.

Launch77701/d8ab5d exits0; controller95010/trainer95011 remain live after
the launcher exits. Increasing1→5→10→25 updates are verified; all three
25-update loss/gradient rows are finite. Real cohort speed about6.3seconds,
driver allocation6.8GB, disk space about52GB. Both ordinary controls share
the actual ISRU batch/noise/time stream; this is not just a PID check.

MPS itself exposes no portable utilization counter, but an actual Apple
`ioreg AGXAccelerator` probe succeeds with device97–99%. Added read-only
telemetry95058,30s interval, frozen under separate telemetry-source and
source-hashed. It does not alter/restart training or the registered recipe.
The existing owned supervisor still records CPU/RSS/speed/phase and actual
child exit, without automatic restart. No formal FID/quality result yet.
