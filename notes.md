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

All six20k runs and primary evaluations now complete. Fixed/free Taylor:
confirmation SW2 Ring8 .088/.089, Product16 .056/.055. Fixed/free midpoint:
Ring8 .071/.071, Product16 .052/.052. Gamma coefficients moved substantially
(up to1.520 Ring8 and1.972 Product16), so compare feature-weighted reset
effects as well as raw values. Raw sampling/field CSVs preserve precision.

Numerical/symmetry queue25633 was frozen24dbb60 with25tasks, two CPU workers.
State/free-stage full-bank internal16 plus14bounded outer-resolution tests
and7C8-stratified interventions. The symmetry shortcut has float64/float32
native trajectory witnesses; each side has4096independent base points and
32768group-stratified points, not32768iid points. Do not mix absolute scores.

GPU6 readiness stopped after8cases because PID2254862 appeared; do not retry
or share that card until a fresh empty inventory. Its partial witness is not
a completion result. Portable12case data and source remain on145.

Open attribution question: all current designs still share P. A same-work
P_j=P+Delta_j control is registered in operator-sharing-control.md, with3zero
matrix offsets per solve and73,728extra model parameters. It tests autonomous
sharing against a larger nested nonautonomous class, not ODE versus non-ODE.
Full internal-refinement evidence supports overlapping this control with
the independent symmetry check. New evaluation streams2026112000/2100 and
six reference checkpoints are hash-bound before formal training.

Trained-state DOP853 checks complete: median32->64 local-map convergence
rates fixed midpoint1.992/1.992 and free stages0.973/0.971 (Ring8/Product16).
This verifies numerical order, not a generation advantage. CPU full4->16
results observed so far: state Ring8 .071->.074 and Product16 .052->.055.
All18resolution tasks are complete and independently recompute within2.78e-17.
Queue25633 then stopped at fixed-K8's float32 C8 shortcut tolerance;21cases
passed, one failed and three were unstarted. Preserve it. A fresh documented
float64 C8 witness passes1.61e-13; all seven C8cases will be repeated as one
separate float64 bundle. Do not combine prior float32 scores into its ranking.
Sharing doc witness passes both native8->16 continuations and evaluators;
initial outputs exact, common gradients agree within4.34e-19, added gradients
nonzero. Additional73,728params and clipping effects remain explicit.

Frozen8205c5a published before launch: operator-sharing queue27486,
workers27487/27488 both past1500/20k, analysis watcher27667. Float64 C8queue
27668 runs two inference workers. 145inventory later showed4-7idle, but
the matched new trajectories remain on the original CPU backend;0-3are
never used. No GPU training or card reclamation occurred.
