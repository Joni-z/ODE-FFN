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
