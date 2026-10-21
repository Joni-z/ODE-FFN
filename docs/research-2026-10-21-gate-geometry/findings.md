# Gate derivatives improve local field geometry, with no isolated multimodal mechanism

This diagnostic ran on **145 / hw3242-gpu1**, using one CPU thread, in **8.76 seconds**. No GPU was allocated. The existing seed-0, step-5000 checkpoints were evaluated without training or parameter changes. All 100 panels passed the [audit](audit.json). Execution source is frozen at `fe0bd5d`; later commits contain collection and analysis only.

## What was measured

At an input x and time t, we retained the trained model and recorded every gate value. We then compared the native velocity Jacobian with the derivative obtained while holding those gate values fixed at that anchor. The prediction at the anchor is exactly unchanged. Both derivatives are compared with the analytic target field's Jacobian.

This answers whether the gates' **input dependence contributes a useful local derivative**. It does not compare two trained models, change the anchor's velocity error, or establish a generation benefit. In particular, the anchored derivative must not be described as the derivative of a model that dynamically recomputes detached gates at every neighboring input.

The interpolation panel uses the same 512 validation-based states across all five architectures at each of five declared times. The trajectory panel follows 128 shared initial indices separately through each model, so its states differ by architecture. Times are 0.1, 0.3, 0.5, 0.7 and 0.9. These panels are not pooled with previous random-time validation evaluations.

## 1. Gate contributions generally correct local derivatives

Errors below are mean squared Frobenius distances from the analytic target Jacobian, averaged equally across the five times. Lower is better. Relative reduction is calculated from the aggregate errors, not by averaging per-point percentages.

| Target / input panel | Anchored gates | Native input dependence | Relative reduction | Fraction of points improved |
|---|---:|---:|---:|---:|
| Gaussian / interpolation | 0.033639 | 0.018676 | 44.5% | 85.2% |
| Gaussian / learned trajectory | 0.033006 | 0.018450 | 44.1% | 83.6% |
| Ring8 / interpolation | 3.575498 | 1.485390 | 58.5% | 88.0% |
| Ring8 / learned trajectory | 7.828940 | 5.966779 | 23.8% | 85.5% |

Mean improvement is positive at every evaluated time for both targets and both panels. Thus the trained gates' derivative contributions are not simply uncontrolled amplification; at these anchors they usually move the complete field derivative toward its target. Gaussian also shows this effect, although earlier removal/retraining of gates did not consistently harm Gaussian quality. Local usefulness in an adapted model is different from architectural necessity.

## 2. The data do not isolate uncertainty-driven mode separation

At Ring8 t=0.5 on common interpolation states:

- Low normalized component entropy: 155 points, mean derivative-error reduction 1.424, relative reduction 74.3%.
- Middle entropy: 356 points, mean reduction 1.205, relative reduction 60.9%.
- High entropy: only one point. It cannot support a broad conclusion.

At t=0.9, all 512 interpolation states are low-entropy and locally contracting in the analytic field. Gate input dependence still reduces derivative error from 3.31231 to 0.44414, an 86.6% reduction. At t=0.5 contracting states also show a substantial reduction. Therefore the observed usefulness extends to within-mode contraction; it cannot be attributed exclusively to ambiguous, expanding boundaries.

This weakens the narrow explanation that the original gate's distinctive role is mode separation. A broader explanation involving local field adaptation remains possible. No arbitrary thresholds were changed after inspecting these results; counts and all strata are in [geometry-strata.csv](geometry-strata.csv).

## 3. SwiGLU remains a strong explanation/control

On common interpolation states, with equal time weights:

| Model | Gaussian velocity MSE | Gaussian derivative error | Ring8 velocity MSE | Ring8 derivative error |
|---|---:|---:|---:|---:|
| Input ODE | 0.003722 | 0.018676 | 0.032183 | 1.485390 |
| Fixed ODE | 0.003600 | 0.016966 | 0.076519 | 2.519224 |
| Scalar ODE | 0.003599 | 0.016970 | 0.075688 | 2.493638 |
| SiLU | 0.003627 | 0.016685 | 0.042022 | 1.578452 |
| SwiGLU | 0.003453 | 0.023583 | 0.013891 | 0.571027 |

On Ring8, SwiGLU fits both the velocity and its derivative more closely than the original ODE model on this panel. On Gaussian the derivative and velocity metrics do not give identical rankings. These pointwise field metrics are not endpoint SW2 or evidence of a universal model ordering.

## 4. Case study: one trajectory point dominates a late-time average

At Ring8 t=0.9, input-ODE trajectory source index **79** reaches approximately `(-1.14881, -2.72937)`. Its analytic log-density is -15.1434, normalized component entropy 0.2959, and squared Jacobian error 2686.29. This single point accounts for **94.7%** of that panel's total native derivative error. The largest four errors account for 97.5%.

Consequently, the trajectory panel's late-time mean error of 22.1656 should not be read as typical of all 128 states. The corresponding common interpolation panel has mean error 0.4441. The learned trajectory visits states that have very low target-path density, where the oracle geometry can be difficult. This is a descriptive model-distribution shift, not proof of its cause or proof that a different P constraint would repair it. The example was found after inspecting the diagnostic; all top cases are preserved in [tail-cases.json](tail-cases.json).

## What follows

The evidence supports a limited statement: input-dependent gates contribute useful derivatives in the original trained model. It does not establish that those contributions are unique to ODE parameterization or concentrated in multimodal separation. The fixed/scalar comparison still changes capacity, while SwiGLU already supplies generic input gating.

This diagnostic does not justify starting a new spectral-constraint sweep or adding a mechanism-specific architecture. The remaining distinction is the value of the shared-P polynomial restriction relative to ordinary gating and its optimization effects. Selecting the next training control should follow that distinction; none has been launched automatically.

## Verification and provenance

The five new tests passed locally and on 145. All panel hashes and stratum metrics were rechecked locally; common interpolation states and zero-correction baseline controls were verified. Anchor outputs match exactly. Maximum native/anchored finite-difference relative error is 1.08e-6. Analysis uses float64 with unchanged saved weight values; float32-to-float64 forward differences are recorded per checkpoint in [provenance.json](provenance.json).

Raw arrays and monitoring files:

```text
145: /data2/zz5070/ODE_FFN/gate_geometry_20261021/output/
local: /Users/mr.z/ode-workspace-prep/gate-geometry-20261021/output/
```

The transport archive SHA256 is `bc0d2264083b40830a98b175ac2f869434cf5ebc7c6c4ea40abd04dffb97bfd7`. See [launch receipt](launch-145.json), [protocol](protocol.md), [complete summary](geometry-summary.json), and [all panel metrics](geometry-panels.csv). The completed diagnostic process has exited.
