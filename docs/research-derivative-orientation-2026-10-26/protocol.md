# Directional derivative versus potential-gradient readout

Registered2026-10-26 before the new directional training or crossover outcomes. The parent potential-readout study is complete and hash-bound. No potential variant or failed checkpoint is continued.

For the same complete raw network G(x,t), compare

```text
A[G]=.5*(G+J_G^T x)  # parent potential gradient
B[G]=.5*(G+J_G x)    # new radial directional derivative
A[G]-B[G]=.5*(J_G^T-J_G)x.
```

A has a symmetric full input Jacobian. B generally does not. For G=Ax+b, the two readouts are sym(A)x+b/2 and Ax+b/2; for nonlinear G, neither equality of feasible functions nor equal optimization geometry is assumed. The tangent x in B depends on the input and must not be detached when differentiating B. A direct input-Jacobian formula is

```text
J_B = J_G + .5 * directional_derivative_x(J_G).
```

Use true forward-mode torch.func.jvp for B, and independently reconstruct it from complete raw Jacobian rows. Check cubic and nonsymmetric affine examples, full input derivatives, mixed P/controller parameter derivatives, batch independence, no_grad behavior, exact initial states and actual optimizer updates. Sampling uses no_grad, not inference_mode, which can disable forward AD. Audit actual aten matrix work and isolated deterministic batch512 preflight time; dense G-forward hooks alone are insufficient.

Train only B for unchanged ODE K2, SwiGLU128 and SwiGLU192 on Gaussian2D, Ring8, Product16: nine new cells. Keep width64/depth3, both original solves/SiLU, time FiLM, normalization and vector head. Use seed0,5000updates,batch512,AdamW3e-4/wd0/clip1, identical cached-bank pair-target FM, uniform time and original input streams. Raw initialization, buffers and Torch RNG match the corresponding direct/gradient parent. No analytic supervision, new scale, distillation, solver change or spectral control.

Reuse all21 parent controls, including direct/gradient SwiGLU128/192 and direct SwiGLU384. Verify source/core/bank identity and actual initial states, data RNG/hash chains and recorded endpoint outputs. Reuse is not independent replication. A small helper refactor may allow explicit builder/settings/script/protocol injection; prove its default trainer compatibility against actual parent three-update histories and verify the new CLI before formal launch. Frozen parent source is not modified.

Primary is final full uniform-time validation oracle MSE. Retain all five512-input field/Jacobian panels, symmetric/skew errors, Heun32 SW2 on2048 common starts/targets, actual costs, training/clipping histories, P spectra and10-second process telemetry. No test bank is used for selection.

Register a descriptive2x2 crossover for every backbone/task: raw weights trained under A or B, evaluated under A and B. Evaluate every full field and all panels with complete input Jacobians. Save per-input predictions, the exact A-minus-B raw-Jacobian identity and squared-error polarization for each finite switch. No crossover or parent A is eligible for promotion; these cases distinguish native fit, fixed-weight decoder dependence and co-adaptation. They do not prove a causal benefit of conservativity alone.

Mechanistic reporting: compare B to its direct parent and to native A on each task; report the complete output's skew norm and fixed-weight crossovers. Comparable B performance without an enforced gradient construction weakens a necessity claim for hard conservativity in this setting. Worse B supports importance of the orientation under this parameterization, not a theorem about all conservative/nonconservative architectures. Approximate symmetry learned by B must remain visible.

Only native cold-trained B-ODE can nominate continuation. Require at least5% improvement over its direct parent and at least10% lower primary MSE than every generic control on a nonlinear task, at most5% primary regression against the best generic on other tasks, and Jacobian/SW2 within10% of the best generic on both nonlinear tasks. Include all newly trained B generic models and all parent generic models. Keep actual parameter/matrix/latency tradeoffs explicit. These are practical criteria, not hypothesis truth scores or significance thresholds.

If none qualifies, stop this directional setting without coefficients, alternate directions, tangent detachment, norm/temperature/spectral changes, widened states or longer-training rescue. A pass still needs separately registered continuation/fresh holdout/cost verification. Do not change the old A result or the global useful-architecture requirement.

Float32 native replays must be bitwise; alternative float64 readout values/Jacobians must agree within rtol1e-8/atol1e-9 (predictions use rtol1e-9/atol1e-10). All inputs remain included, with bounded chunks for higher derivatives. Any numeric failure is investigated with preserved files, not a silent threshold change. Two one-thread CPU workers on Mac mini; Lenovo is currently offline. No automatic restart after observation timeouts.

Pre-outcome implementation amendment: the installed native LayerNorm JVP higher derivatives fail finite differences (implementation-errors.md), while native scalar-potential reverse derivatives pass. Use native LayerNorm primal values with explicit mathematically equivalent backward/JVP formulas, retaining exact epsilon and Parameter objects. Verify first/second derivatives and actual mixed gradients before launch. This changes neither the intended B operator nor a model/data/quality setting. Parent scalar-potential finite-difference checks are additionally required; no training has started at this amendment.
