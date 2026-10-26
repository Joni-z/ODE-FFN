# Fixed-network teacher fitting under a known representer

Registered before any new student optimization. This is a learning diagnostic, not distillation proposed as a new method and not ordinary FM architecture validation.

## Question and limits

Earlier exact-population regression leaves an ODE deficit, but that population target is not known to lie in the tested finite ODE class. A frozen same-architecture ODE teacher is known to lie in it: its complete weights and actual runtime buffers are an explicit zero-risk representer. If the cold ODE student still fits this function much worse than generic students under the fixed budget, absence of that particular target from the class is not an explanation. This does not establish an optimization mechanism, show that the true field is representable, or guarantee a useful new architecture.

The second frozen SwiGLU teacher distinguishes matching one learned smooth field from matching a richer ordinary predictor. Show teacher energy, fixed-time derivative energy and existing nonlinear-response measurements; do not assume equal complexity. Teacher choice and student widths are fixed without inspecting new training results.

## Teachers, students and six cells

Teachers: the completed conservative_readout_20261026/output direct ODE K2 and direct SwiGLU128 checkpoints, step5000, Product16. Preserve exact checkpoint hashes, persistent parameters and nonpersistent Fourier buffers. Both use the immutable conservative-readout source-v2 implementation.

Students: cold direct original ODE K2, SwiGLU128 and SwiGLU192, with their unchanged native constructors. All three retain the identical surrounding initialization from the existing build routine; their step0 model tensors and runtime buffers must match their historical native step0 snapshots. ODE and SwiGLU128 are near parameter matched; ODE and SwiGLU192 are near executed-matrix-work matched. The student FFNs, heads, conditioning and normalization are unchanged. No teacher parameter, feature, rotation, Jacobian or optimizer state is copied into a cold student.

Run the complete two-teacher by three-student layout, six new cells. Each starts from seed0 independently; compatibility reruns and teacher restoration are not extra training seeds or independent replications.

## Input replay, cached labels and training

Use the existing Product16 cached training target bank, independent fresh standard Gaussian sources and uniform times. Reproduce the old input generator and draw order exactly: its fixed data-generator state is derived from training seed0 as in the existing runner. This is not an additional model seed. Batch512,5000updates, AdamW3e-4/defaults, weight_decay0, global_clip1, no schedule. The only training-target change is replacing pair velocity by the chosen frozen teacher's output at the same state/time.

To avoid evaluating each teacher three times for the identical stream, precompute one shared sequential input cache and one label cache per teacher. Arrays are float32 memory maps; no training-point or label selection. Record every input/label hash chain and the input RNG states at snapshot rows. All six cells consume the identical ordered inputs and the appropriate identical teacher labels. Verify the final input chain against the historical stream. Labels are produced with eval/no_grad and frozen teacher parameters/buffers, and are independently replayed before interpretation. No test split is opened.

The original uniform-time 20000-row validation field bank and five512-row panels define evaluation inputs. Cache both teachers' actual outputs and full Jacobians on these panels. Retain predictions for every student and checkpoint; teacher risk is evaluated directly against the selected teacher. The analytic oracle and ordinary FM metrics remain secondary context, not selection endpoints or replacement targets.

Save snapshots at0,1000,2500,5000 with full model, optimizer, actual runtime buffers, global RNG and consumed-cache position. Also save the input/label chain at that position. Log train loss, gradient norm, clipping, validation teacher risk, teacher energy, relative teacher risk and actual P trajectories. A replayed source-RNG state is labeled as such rather than pretending that cache consumption sampled new data. Preflight uninterrupted3 versus2+resume3 must match exactly.

## Fixed primary measurements and interpretations

Primary: full uniform-time validation teacher MSE at5000updates, normalized by that teacher's mean squared output. Also report its unnormalized value, the five fixed-time risks, relative Jacobian error, complete learning curves and actual student matrix/parameter costs. Do not select a better checkpoint, teacher, time or student width.

The descriptive learning-difficulty pattern is declared in advance: on the ODE teacher, cold ODE relative risk exceeds.05, while **both** SwiGLU references have relative risk at most.05 and at most half the ODE student's error; their mean fixed-panel risk must also be at most half, with derivative error no worse. If all conditions hold, report a finite-budget ODE fitting difficulty on a function demonstrably in its own class. Otherwise the specific pattern is not established; report every value. The5%/twofold thresholds are practical diagnostic cutoffs, not significance claims.

On the SwiGLU teacher, report all ratios and the cross-teacher interaction descriptively. A poor ODE fit there cannot prove the function lies outside its class. A good fit to its own teacher does not show adequacy for the analytic target. A result shared by all students is a generic learning observation, not an ODE mechanism. Exact teacher-as-student zero risk is a realizability/procedural control, not trained performance.

No cell or secondary row nominates a new generative architecture. No longer-budget, learning-rate, target-scale, teacher-strength, width, head, activation or optimizer rescue follows automatically. An actual architectural hypothesis requires a separate argument and ordinary pair-target FM controls.

## Verification and execution

Before launch verify old teacher forward/Jacobian replay, complete cold initialization, exact teacher representer, no teacher gradients or parameter changes, label-cache equivalence to live teacher training, actual optimizer updates to P/controllers, deterministic checkpoint resume, actual matrix counts and monitoring writes. Freeze source/protocol and bind actual source/data/config/checkpoints before exposing readiness.

Use at most two one-thread Mac CPU workers with actual process/CPU/memory telemetry every5seconds. Stop the queue on a genuine failed cell and preserve its outputs; do not restart because a session observation expires. Lenovo is unassigned while SSH times out; any future migration requires native buffer/forward preflight and an exclusive whole-cell assignment. No GPU is reserved. Preserve all old sources/results and Report1; no commit/push.
