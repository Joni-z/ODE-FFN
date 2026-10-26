# Scope, controls and audit

The pre-outcome protocol fixes two actual step 5000 teachers and three unchanged cold student families. Every formal cell uses training seed 0. The six cells are one teacher-by-student comparison, not independent repeated estimates of each architecture. Validation rows, time panels, internal solves and training checkpoints are measurements of these predictors, not independent trained-model replications.

## What the realizable control proves

The ODE teacher and ODE student have the same finite architecture and inference code. Restoring the teacher's complete weights and actual buffers into a separate instance therefore gives an explicit functionally identical representer. Both teacher-as-student controls reproduce all 20000 validation outputs exactly. These control weights never initialize a formal cold student: all 24 actual student snapshots at step0/1000/2500/5000 are audited, and each step0 state exactly matches its historical cold native reference.

A poor cold fit to the ODE teacher cannot be explained by absence of that function from the class. It can still reflect finite-budget optimization and parameterization. Conversely, a good fit to that teacher does not establish representability or learnability of the analytic population oracle or the frozen SwiGLU teacher. The teacher functions differ in shape as well as construction, so this is not a causal attribution of difficulty to the teacher architecture itself.

## Inputs, labels and replay

The input stream replays the original training procedure: cached Product16 target rows, independent Gaussian source, and uniform time. The historical data generator uses state 100000 derived from training seed 0. The full 5000-step target/source/time hash and final RNG state agree with both old teacher checkpoints. That stream is distinct from the actual model-seed setting; no alternative model seed is run.

One immutable sequential float32 memory map stores each training state/time, and each teacher has one output-label map. All six students consume the same ordered inputs. Each actual training batch also updates independent consumed-input and consumed-label hash chains. The source target/noise/time digest, replay digest and label digest are all retained at snapshot rows. Cache position is the actual training cursor; source RNG states are recorded separately rather than presented as new sampling during replay.

After all cells became terminal, the independent audit regenerated every input with a fresh generator and replayed 10000 teacher label batches bitwise. It recomputed every saved chain and verified actual consumed hashes in all student checkpoints. The cached targets were produced without teacher gradients or changes to weights/buffers. Target caches reduce repeated label computation; they do not alter the regression objective.

The uniform-time validation bank has 20000 rows; each fixed-time panel has 512 rows. They are previously exposed project validation inputs, not a fresh confirmation set. No test targets were opened for selection. The diagnostic does not use generated samples or claim a practical FM/image-generation gain.

## Metrics and fixed rule

Teacher MSE averages squared vector-coordinate error. Relative teacher MSE divides by the fixed teacher's mean squared output on the same bank. Relative MSE 0.00629465 is 0.629465% squared error; its relative RMS would be about 7.93%, so these quantities must not be interchanged. Panel Jacobian error averages squared Frobenius differences; relative Jacobian error divides by the corresponding teacher derivative energy. Native float32 outputs are retained, and metric reductions are recomputed in float64.

The registered severe-learning-difficulty pattern required ODE self-teacher relative MSE above 5%, with both generic references at most 5%, halving its field and mean-panel field errors, and no worse derivative error. The pattern is false: the ODE achieves 0.629%, and SwiGLU128 does not halve its self-teacher MSE. SwiGLU192 does better than half, but cannot replace the other required control. All six endpoints and all time rows are reported. No post-result threshold, teacher, width, checkpoint or training-budget selection replaces this rule.

The much larger gap on the SwiGLU teacher is descriptive. It motivates studying which nonlinear functions are difficult for this finite ODE predictor; it does not prove a function-class separation or identify the optimizer cause. Teacher derivative energy is also descriptive rather than a complete complexity or stiffness measure.

## Initialization, numerical and execution checks

The full preflight verifies 10 historical teacher prediction/Jacobian panels and two complete field banks; all three student initial states/buffers; two exact teacher representers; six live-versus-cached three-update trajectories; and six uninterrupted3 versus2+resume3 trajectories, including optimizer state. Nonzero P/controller updates are observed in the ODE preflight. Executed dense forward/loss-backward MACs are counted through actual Torch matrix operations. ODE totals 83862 parameters / 106944 forward MACs; SwiGLU128 83598 / 81792; SwiGLU192 108624 / 106560. Elementwise work is additional, and these counts do not establish measured latency.

After the first passing preflight, actual consumed-tensor hash bookkeeping was added. The original source and passing receipt remain in preflight-source-v1. A focused stream/checkpoint delta check passes before formal source freezing. No numerical model, label, optimizer or criterion changes. The final frozen source has 9 files plus the bound immutable predictor source; both receipts gate launch.

Formal supervisor session 40507 / PID 1521 ran once and exited0 after 96.33 seconds. All six child cells reached 5000 updates. Five-second monitoring records actual parent/child CPU time and RSS, with at most two student processes. The largest observed process-RSS sum is 1.43 GiB; it is not total machine memory or a true unsampled peak. Concurrent elapsed cell times are not isolated inference benchmarks. Lenovo SSH timed out, so no remote assignment or GPU reservation was made.

Independent session 33667 / PID 1596 exited0 after 18.89 seconds. It verifies 24 complete checkpoints and optimizer counters, replays all 24 full saved fields and 30 native endpoint prediction/Jacobian panels bitwise, and reconstructs all 30 endpoints through a separate float64 forward/derivative implementation with zero measured discrepancy. It retains 48 original-P checkpoint records and independently checks singular spectra plus eigenvalue invariants. No student training is rerun for this audit.

One read-only status command mistakenly used the system Python to import psutil and failed before reading status; it was corrected to a standard-library read. The same live supervisor handle was then polled. No training, preparation or audit process was restarted because of that observation error.

The larger useful-ODELayer objective is not achieved by this diagnostic. Any future architectural claim still requires ordinary pair-target FM, appropriate strong controls, transfer, fresh confirmation and measured cost. Earlier sealed files and submitted Report1 are preserved; no commit or push occurs in this study.
