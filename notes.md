# Mechanism execution notes

Research contract: docs/research-assessments/2026-11-16-ode-mechanism/research-question-card.md.
Existing CPU fixed controls are under response-screen-20261115/output.
Keep their tensors, source and data immutable. Recompute on new common banks.
Formal free-coefficient models: alpha2..8 per local solve initialized1;
native recurrence and original parameter gradients must match at initialization.
145 GPUs4-7 currently occupied, including6/7 with0% instantaneous utilization
but resident compute processes. They are not free cards. No GPU allocated.

Both coefficient20k trajectories completed with zero exit and native final
test artifacts. Frozen8c10a8d; queue22510 terminal, analysis watcher22707 active.
State stage1ff0d02 started queue23380 with workers23381/23382, watcher23556.
All work is Mac mini CPU. Warm environment and data remain unchanged.

Frozen-K8 probes at5times x8states x6layers: median restart/increment ratio
Ring8 .139 and Product16 .131, versus tiny Taylor truncation. The refreshed
correction/increment ratios .247/.207 and full-prediction effects are resolved.
This satisfies the predeclared state-flow diagnostic gate, not a claim that
feedback improves already-trained weights. Fixed-weight replacement worsens fit.

Oracle transport probes completed for8old plus2new cases,64sources and160
Heun steps, with320step refinement. Linear sensitivity works locally but its
lambda1 residual is large on these multimodal tasks. Do not use its endpoint
prediction as an explanation of full learned sampling. Window interventions
are measured directly; paired endpoint error is not SW2. These are bounded
diagnostics, not complete distribution-evaluation results.

A state-flow win would initially identify feedback/repeated nonlinearity,
not uniquely ODE structure. Conditional-stage-control.md registers a same-
function initialization release of four midpoint stage coefficients, with
equal response evaluations/P calls. It is not launched before confirmation.
Do not close the full research goal merely because the first four runs finish.
