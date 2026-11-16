# Does the same-weight solver benefit survive the outer sampling mesh?

Registered after the complete24-role extreme-source diagnosis, before new
full-bank resolution observations. That diagnosis conditions its maxima on
Heun128and therefore cannot establish a population sampler effect.

## Fixed question and cohort

Primary question: for Product16RK4-trained weights, does replacing the hidden
RK4solve by midpoint still improve the final distribution at Heun2048on BOTH
original fresh-bank splits? The other three trained weights are controls.

Keep all four20kseed0checkpoints, their native/crossed hidden solvers, both
target-transport streams, all eight4096-source/target banks and256directions.
No training, new random draw, response change, fitted coefficient, checkpoint
selection or ordinary-baseline ranking. All internal solves retain8Pactions.

Outer Heun counts are127,128,129,512,2048.128is a replay anchor to the cached
complete study.127/129check neighboring meshes;512/2048check finer meshes.
2048is the primary finer comparison, not a certified continuous reference:
the actual time embedding has high frequencies and the network is nonlinear.
Do not select a winning count or describe this as convergence from two counts.

## Portable model identity and numerical gates

On Mac, restore the original FP32models and their actual runtime buffers;
require the original4096-source Heun32native anchor to replay exactly. Export
all named parameters AND all named buffers, including nonpersistent time
frequencies. Cast those actual values to FP64; bind every tensor/file hash.
Reconstruct with the original builder and copy every tensor by exact name and
shape. State dictionaries alone are insufficient for nonpersistent buffers.

Run the unchanged native generated_samples Heun algorithm and sliced_w2.
First replay128on all32768sources for native AND crossed. Component endpoint
error versus original CPUFP64cache must be<=1e-7*(1+abs(reference)); pooled
SW2difference must be<=2e-8+1e-7*abs(reference). Check all eight bank metrics
as well. Failures remain explicit and block new-count observations for that
case/split. Retain full endpoints and errors, including rare-source tails.

A bounded CUDAwitness uses the first64sources from each bank, all models and
both splits. The full-bank128gate still applies during formal execution;
passing a bounded witness does not substitute for it. Batch source evaluation
per bank, keeping the native4096shape; concatenate saved endpoints only for
the pooled metric. Preserve all model parameters/buffers and inference RNG.

## Analysis and decision

Save native/crossed endpoints at EVERY count, bank SW2and pooled SW2. Independently
reconstruct using NumPysorted projections from the same targets/directions.
Record paired bank changes in squared SW2, their mean/SD, number improving,
95%conditional percentile intervals and32-contrast-adjusted intervals.
Enumerate all6435multinomial count vectors with exact multiplicity8^8;
the adjusted two-sided tail is.05/(2*32). The32contrasts are4weights x2splits
x4new counts.128is descriptive cached-reference replication, outside that family.
These are evaluation-bank intervals conditional on frozen training, not training
replicates or guaranteed finite-sample coverage. Report pooled changes separately.

Primary survives if both ProductRK4splits pass numerical identity gates and,
at2048, have negative pooled crossed-minus-native squaredSW2change AND a
negative upper endpoint of the adjusted bank-mean interval. Otherwise report
the measured direction/uncertainty and do not promote an intrinsic hidden-ODE
generation benefit. Preserve every other count and model regardless of outcome.
If it survives, test whether the analytic midpoint/RK4modified-field defects
predict the useful complete-field and target-functional change. If it reverses
or becomes unresolved, diagnose outer-time integration before new architecture.

## Resources and implementation boundary

Warm Mac CPU04f11f75exports original tensors; warm145CUDA41450454evaluates.
No package change. Use only freshly empty physical4-7(<500MiB,no compute PID),
one process/card, one CPUthread/process;0-3prohibited. Recheck before/after each
bank and save GPUtelemetry. Never stop or share another user's work.
Publish/freeze the portable adapter and documented bounded invocation, run an
independent agent-follows-doc witness, then launch monitored formal execution.
Final reports and all five native image files remain unchanged.
