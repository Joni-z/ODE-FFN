# Centered clock-curvature experiment

Registered before inspecting any new training outcome on Lenovo, 2026-10-24.

## Question
Does the quadratic dependence on the input gate contribute beyond the static K2 map and the first derivative with respect to that gate?

For s0=0.6 and delta=s(h)-s0, the professor's layer is exactly

```text
F(h) = [B0 + delta B1 + 0.5 delta^2 P^2] h
B0 = I+s0 P+0.5 s0^2 P^2
B1 = P+s0 P^2.
```

The control `ode_gate_linear` subtracts `0.5 delta^2 P^2 h`, preserving the same P, gate network, active parameter count and matrix MACs. Native initialization is retained; at random initial gates the two internal functions are close but not exactly equal. Both final output projections start at zero, which alone does not imply equal internal functions. This control tests gate curvature, not shared versus independent matrices and not the entire value of P-squared.

## Fixed protocol

- Datasets: existing Gaussian anisotropic 2D and Ring8; copy the original versioned files and verify manifest hashes.
- Seed: 0. Three arms: native original K2, the gate-linearized K2 control, and SwiGLU.
- Width64, depth3, gate hidden64, nominal FFN hidden128; original gate temperature10, shift0.2 and scale0.8.
- Independent standard Gaussian source, linear interpolation, direct velocity regression, uniform time; batch512.
- AdamW learning rate0.0003, weight decay0, gradient clipping1, 5000 updates.
- Validation every500 updates; snapshots at0/1000/2000/3000/4000/5000. Final step5000 is primary; no checkpoint selection for primary claims.
- Validation SW2:2048 samples,128 directions,Heun32. These are progress metrics, not a basis for tiny winner claims.
- Final common-point velocity/Jacobian diagnostic:512 validation-based interpolants at t=.1,.3,.5,.7,.9, equally weighted, float64; identical points for all models. Record native and curvature-removed fields, one layer at a time and all layers together, using s0=.6 fixed in advance.
- Record P spectra and actual gate/activation summaries throughout training. Save raw diagnostic arrays and hashes.

## Sequence

1. Verify native Windows CUDA, model semantics, environment and exact dataset files. Benchmark real training updates on CPU and CUDA, including 1 versus2 CUDA processes; choose useful throughput within memory limits.
2. Train the native and SwiGLU arms on this same environment. Older CPU checkpoints are currently inaccessible through SSH and not present in the local curated export, so do not silently claim the new diagnostic uses those older weights.
3. Diagnose the newly trained native checkpoints. Start the selected gate-linearized training control when the deleted term materially affects the field or derivative. Predeclared practical trigger: aggregate relative field RMS change or aggregate relative Jacobian Frobenius change of at least1% on either target's common panel. This is an engineering screening threshold, not a statistical significance claim. Save the decision before launching the control.
4. If both changes are below1% on both targets, stop the curvature training branch and report the result. The free-B parameterization experiment requires a separately recorded choice and is not automatically launched alongside it.

The diagnostic can show dependence of trained weights; the retrained control addresses whether the quadratic gate relation helps learning. A worse frozen intervention alone cannot establish a superior architecture. All arms must share the selected device, numerical settings and environment for the primary comparison.

## Resource and execution policy

The environment is isolated under D:\Research\ODE-FFN. Do not increase batch or model size solely to fill GPU memory. Benchmark concurrency and record actual utilization/temperature/memory, process liveness and steps every10 seconds. At most2 training processes on the8GB GPU; use fewer if throughput fails to improve or memory is insufficient. CPU diagnostics should use one thread. Stop and mark failed runs on nonfinite losses/gradients or nonzero process exit; preserve logs and snapshots. Background execution must survive the SSH command ending, and status must be verified after launch.

## Outputs

Source snapshot hashes, Python/package lock, environment/GPU information, benchmark, stage decisions, status/telemetry, per-run logs and checkpoints, common-point diagnostics, and a final machine-readable summary. Deployment checks may use tiny temporary data; scientific training remains seed0 and the full original dataset.
