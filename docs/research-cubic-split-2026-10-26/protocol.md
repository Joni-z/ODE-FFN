# Cubic local flow: one bounded ordinary-FM screen

Declared before new outcomes, 26 October 2026. The completed bias-placement seal is `a5a5637e99c2733ee533ce7ea01efec20fc307f2ac0c99c673dbe1c6c2df3ec7`. Existing studies do not demonstrate that a P spectral condition or a missing bias explains the nonlinear gap. This screen asks a different question: whether a nonlinear local flow with an odd saturating response is useful at a measured small-model budget.

## Operator

Take the first P and scalar controller from each cold original FFN. Retain their exact initialization and original duration `a(h)=.2+.8 sigmoid(controller(h)/10)`. Replace the full two-solve/SiLU FFN with one local flow of

```text
dy/dr = P y - y^3                    # elementwise cubic; coefficient fixed to one
```

The duration is selected once at the input. Approximate its endpoint by a linear/nonlinear/linear composition with the same P in both halves:

```text
H_a(z) = z + (a/2) Pz + (a^2/8) P^2 z
C_a(z) = z / sqrt(1 + 2a z^2)
F(h) = H_a(C_a(H_a(h)))
```

There is one symmetric split step, with second-order Taylor linear half-flows and an exact cubic subflow. No projection bias, second P, extra readout, trainable cubic coefficient, new loss, internal recurrence or spectral constraint is added. This changes the local field rather than increasing the order of a stopped SiLU-drift field. At fixed selected a the response has nonlinear odd components; adaptive a need not be even, and whole-backbone inputs are not assumed symmetric.

## Mathematical meaning and limits

With `mu=lambda_max(sym(P))` and `R=||y||^2`, the continuous field satisfies `R' <= 2 mu R - (2/d) R^2`. This establishes a forward boundedness mechanism for the specified field. It does not certify stability of the finite split step, the complete residual network or the learned FM predictor. The implemented map has local error O(a^3) as a tends to zero on bounded states/matrices; actual durations .2–1 are not assumed small. Check consistency independently rather than infer accuracy from its name.

The nonlinear subflow is exactly a known ISRU response with alpha=2a. No activation or splitting novelty is claimed. A standard free-matrix ISRU FFN is indispensable; if it explains an improvement, the result is ordinary nonlinear response, not evidence for the ODE coupling.

## Arms and controls

Five new arms on each of the unchanged three tasks: gaussian_aniso_2d, gmm_ring8_2d and rotated_product_bimodal_16d.

1. `cubic_split`: the map above, one P/controller per FFN (8321 FFN parameters).
2. `silu_split`: same P, controller, two linear half-maps and initialization; replace only the middle C map by SiLU. Parameter and matrix-work matched composition control; it is not asserted to solve the cubic field.
3. `free_isru`: two independent bias-free 64x64 matrices with fixed `C_.6` between them. Initialize both from the same cold `H_.6` matrix. This is an ordinary 8192-parameter activation control, without an adaptive clock or tied generator. Its initial function is not claimed identical to the adaptive candidate.
4. `swiglu64`: ordinary nominal width64, actual hidden42, 8212 FFN parameters, a near-parameter control.
5. `swiglu160`: ordinary nominal width160, actual hidden106, 20628 FFN parameters, a near-forward-matrix-work control (20352 versus candidate20544 MACs per example).

Reuse the three native original-ODE K2 and three native SwiGLU128 direct-readout trajectories from conservative_readout_20261026 after checking source, initial state/buffers, optimizer, actual data streams and native outputs. They remain six historical references. All seven arms have the same surrounding tensors and time-frequency buffers, native residual scale and direct velocity readout. Ordinary baselines retain their standard biases. No teacher output enters training.

## Training, measurements and decision

Keep seed0, 5000 updates, batch512, AdamW3e-4/default moments, weight decay0, clip1, uniform-time independent noise/target pairs and the exact historical generator stream initialized at100000. That generator is a fixed data-stream offset, not another training seed. Use the existing hashed data/banks; no new test split or task-specific analytic initialization.

Primary: uniform-time oracle validation MSE. Report all five time panels, full velocity-Jacobian errors, 32-step common-source sampling SW2, all four snapshots, P/controller/activation trajectories, clipping and actual matrix MACs. Matrix counts exclude pointwise work, so no latency advantage follows automatically. Full run timing is descriptive of this Mac execution.

A candidate is eligible for another experiment only if Product16 MSE is at most .9 times the best ordinary SwiGLU risk, at most .8 times original ODE risk and at most .9 times both silu_split and free_isru risks. Gaussian and Ring8 must each stay within1.05 times the best ordinary SwiGLU risk; mean Jacobian error must stay within1.10 times the best ordinary SwiGLU mean on every task. Sampling must be finite and no worse than1.10 times the best ordinary SwiGLU SW2 on any task. These are descriptive practical gates, not significance tests. Every arm and exception remains visible.

Failure stops this fixed cubic-field screen: no coefficient, duration, solver order, stage count, width, output head or optimizer rescue. Partial improvement over the old ODE alone is not success. This small synthetic screen cannot establish natural-image/JiT gains, ODE-specific necessity or a publication-ready method.

## Preflight, freezing and monitoring

Check analytic input Jacobians and P/controller adjoints with nonconstant clocks; verify nonlinear subflow and whole-map small-step expansions independently of the native forward. Compare untouched historical initial parameters/buffers/optimizer and three-update histories. Verify exact uninterrupted versus resumed trajectories for every new arm. Record actual matrix work with dispatch counting, and verify common non-FFN tensors.

Freeze source hashes, protocol and all input/prior bindings before the single launch. Save model/optimizer/RNG/buffers and consumed-stream digests at0/1000/2500/5000; metrics500, status100 and CPU/RSS monitoring5seconds. Use two one-thread CPU children. An observation timeout never authorizes restarting training. Independently replay all endpoints, saved checkpoints and consumed streams before an architecture claim.
