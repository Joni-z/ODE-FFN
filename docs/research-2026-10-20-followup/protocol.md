# Frozen-order intervention and controlled operators

Prepared before GPU evaluation on 2026-10-20. This is a diagnostic stage, with no new training, loss, normalization, architecture family, or optimizer. Training seed remains 0. The user authorized further experiments and specifically requested efficient GPU use.

## H1: numerical saturation versus training adaptation

Use the original input-gated K=1,2,4,8 checkpoints at update 5000 on Gaussian, Ring8, and Moons. In each model, freeze all learned parameters and change every ODELayer's evaluation to K=1,2,4,8 or a direct matrix exponential. This gives 60 intervention cells. Evaluate the same-update SiLU and SwiGLU checkpoints as six unchanged controls: 66 cells in total.

“Frozen gates” means frozen gate parameters, not frozen gate values. In the full-network intervention, later hidden states and gate values can change when upstream operators change. A separate local diagnostic captures the original hidden inputs and gate values and compares all polynomial orders against the exponential on these identical inputs.

All polynomial and exponential actions in the intervention use float64, then cast back to the native hidden dtype. Gate calculations and the rest of the network remain float32. Native float32 evaluation is separately reproduced for all 18 checkpoints. Both native versus archived SW2 and native versus the same-order float64 action must agree within 5e-4, or the worker stops rather than attributing numerical drift to the mechanism.

Primary: SW2 at Heun32 using exactly the existing 2048 source/target points and 128 directions. Also report oracle/pair MSE, oracle MSE in ten time bins, generated-endpoint RMS and predicted-velocity RMS relative to each checkpoint's native evaluation, and the local exponential-action discrepancy. Moons has no oracle. No best-checkpoint or best-order selection is used. This initial diagnostic keeps the external solver fixed; it does not establish convergence for every changed vector field.

Read the full trained-order by evaluated-order matrix. Small within-checkpoint K4/K8 effects would support numerical saturation; large within-checkpoint effects combined with similar separately trained endpoints would suggest training adaptation. Report raw effects. An absolute relative SW2 change <=1% is a descriptive practical flag, not a significance or equivalence test. Checkpoints, layers, and probes are correlated, not independent seeds. New test-set inspection makes this exploratory.

## H2: what eigenvalues and spectral norm do not determine

Run small CPU-only matrix examples:

1. A 2D upper-triangular shear with diagonal -0.25 and varying off-diagonal magnitude: fixed eigenvalues, changing norm.
2. Rescale each shear to spectral norm 1: eigenvalues now change, so this does not isolate geometry at fixed spectrum.
3. A 3D nilpotent-chain family with diagonal -0.25 and an off-diagonal scale solved to give spectral norm 1: both eigenvalues and spectral norm are controlled while the chain shape varies.

For hidden durations 0.2,0.6,1.0, measure polynomial gain at K=1,2,4,8, exponential gain, and relative operator error. Record spectral radius, norm, logarithmic norm, stable rank, and non-normality. These are illustrative mathematical controls, not evidence that a new P constraint improves FM. Matrix properties besides the two controls still co-vary.

## Resource plan and monitoring

- One GPU allocation, not one GPU per dataset. Eligible types: A100, L40S, H100, H200, RTX6000.
- Batch exponential actions in chunks of 512. Use one CPU thread per GPU worker.
- Warm up the actual K2 model, then benchmark 1 versus 2 GPU worker processes using identical exact-action forward batches. Compare aggregate examples/second after initialization; use two only if throughput is at least 10% higher. The benchmark covers the expensive exact-action component and is not a training-speed claim.
- Process the three datasets with the selected concurrency on that same GPU. Do not start extra training jobs to fill the card.
- Sample allocated-GPU utilization, memory, and power every 10 seconds; record progress after each cell.
- A detached local monitor checks Slurm and artifact completion every 60 seconds, with a 15-minute no-progress warning. It never resubmits jobs or sends external messages.
- Finish the GPU allocation immediately after the evaluation and checkpoint-integrity audit. CPU-only plotting/report writing happens separately.

## Provenance and failure policy

Freeze code and a manifest before sbatch. Validate all selected dataset files against the actual torch manifest, all 18 checkpoint hashes and configurations, and source hashes before and after the stage. Never edit old training outputs or the previous frozen runtime.

A native reproduction mismatch or source mismatch fails the worker. Nonfinite interventions are recorded explicitly and cause an incomplete stage, rather than being silently omitted. Successful completion requires all 66 finite cells, three dataset completion markers, unchanged checkpoint hashes, and Slurm exit 0:0. H3 (new gate training) remains a separate decision.
