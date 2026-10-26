# Shared hidden evolution and intermediate supervision

Registered2026-10-26 before formal training outcomes. This is one architecture/learning-mechanism comparison based on the completed literature review, not an extension of the failed prediction-space, spectral, activation or gate variants.

## Operator and hypothesis

Preserve the original scalar K2 ODEFFN, including both frozen-clock solves, tau10, duration .2+.8*sigmoid(logit/10), and the intervening SiLU. Preserve width64, three executed residual updates, affine conditioning on the external time, LayerNorm, and the existing zero-initialized final readout.

Within one network call, x_t and external t stay fixed. Let h0 be the ordinary input embedding. A residual update is the existing TimeConditionedBlock:

```text
h_(k+1) = h_k + a * FFN_theta(LayerNorm(h_k)*(1+scale(t))+shift(t))
a = 1/sqrt(3),   k=0,1,2
```

Independent blocks use different theta_k; shared blocks reuse one theta. For the shared model this is an Euler discretization of a hidden-state field with step a and training horizon3a. No claim is made that a is numerically small, that extra iterations improve prediction, or that the hidden P equals a physical velocity Jacobian.

The same final norm and output head can read every intermediate state. Final-only training uses the last prediction. Deep supervision uses weights(.25,.25,.5) on the three predictions, so its weights sum to1 rather than merely increasing the loss scale. Each readout regresses the same sampled pair velocity. Backpropagation traverses the full rollout; no state or intermediate target is detached. There is no depth embedding, input reinjection, learned step size, new activation, extra head or spectral constraint.

The hypothesis is that a consistent shared update with intermediate supervision can learn useful repeated refinement more efficiently than independently trained updates. It may fail through insufficient capacity, unstable long rollouts or loss interference. Benefits also found with SwiGLU must be labeled generic recurrence/supervision benefits rather than an original-P advantage.

## Fixed14-arm screen per dataset

For each of professor scalar K2 and nominal SwiGLU192 (actual hidden128), run:

1. native independent initialization, final-only loss;
2. native independent initialization, deep supervision;
3. independent blocks cloned from block0, final-only loss;
4. independent blocks cloned from block0, deep supervision;
5. one shared block, final-only loss;
6. one shared block, deep supervision.

Add native independent SwiGLU128 with each loss as cheaper, approximately ODE-parameter-matched practical controls. All14 arms run on Gaussian2D, Ring8 and Product16:42cells total. These are fixed controls, not a hyperparameter search. K2 versus SwiGLU192 approximately matches executed matrix computation; the shared versions also expose their distinct unique-parameter budgets.

Construct one seed0 ODE-K2 native reference. Copy all non-FFN parameters and actual runtime buffers into each family. For cloned/shared models, clone or reuse the first complete block after that copy. Thus cloned and shared versions of one family have identical initial hidden functions and native readouts, also under a nonzero head; their independent parameter responses/optimizer updates need not agree. Native independent references retain stronger unconstrained initializations. All arms receive identical sampled target-index/noise/time streams per dataset.

## Data, training and execution

Use the existing immutable cached Gaussian anisotropic2D, Ring8 and rotated Product16 banks. Keep training seed0,5000updates,batch512,uniform external times, independent standard-Gaussian source, direct pair-velocity prediction, AdamW3e-4/default betas, weight decay0 and global gradient clip1. No analytic oracle is used as a training label.

Use complete same-host arm groups. Initial assignment is Gaussian/Ring8 to Mac mini; Product16 to Lenovo if SSH is available before it starts. If unavailable after the Mac groups finish, assign Product16 locally rather than leave computation idle. Record assignment before launch and never launch the same cell twice. Both hosts use two one-thread CPU workers and10-second process/memory monitoring; no GPU is reserved. Cross-host results are not treated as paired trajectory replications.

Preserve frozen sources, initial/1000/2500/5000 checkpoints, optimizer and generator states, actual nonpersistent buffers, input hash chains, curves, P spectra, and actual gate/feature behavior at each recurrent visit. A unique shared P has several activation contexts; do not report only its last call as the whole trajectory.

## Evaluation and selection

Primary: population-oracle velocity MSE on all common uniform-time validation pairs from the existing validation targets. Also retain fixed-time field and full-network Jacobian error at t=.1,.3,.5,.7,.9. Generate2048 common validation samples using Heun32,128 slice directions, and record SW2 plus target-vs-target floor. All evaluations initially use three executed updates.

For every shared model, record field error at1,2,3,6,12 updates with the original step a. These change the internal horizon. Separately record6 and12 updates with steps3a/6 and3a/12 to hold the horizon fixed. These are frozen-checkpoint diagnostics, not additional training or interchangeable ODE interpretations. Do not select the best iteration count after seeing results and call it a fixed-compute improvement.

At the training depth, an ODE candidate may advance on a quality claim only if the same setting:

- improves primary MSE by at least10% over the best generic control on Ring8 or Product16;
- is within5% of the generic frontier on the other tasks;
- has nonlinear Jacobian error and SW2 within10% of the best generic values;
- has no more than5% higher dense inference MACs than the qualifying generic comparator.

A parameter-efficiency lead instead requires primary MSE within5% and nonlinear Jacobian/SW2 within10% of the generic frontier on every task, with at least25% fewer unique parameters than the strongest qualifying practical dense comparator, and no loss against the same-cost shared SwiGLU setting. A merely generic sharing benefit is reported separately.

For extended-horizon six-update generation, only an arm whose three-update nonlinear field errors are within10% of the generic frontier is eligible for extra sampling. Compare its Heun32 against dense generic Heun64 and shared generic six-update Heun32, recording actual total MACs. An extended-depth lead needs at least10% SW2 improvement on one nonlinear task, at most5% regression on other tasks, and comparable total MACs. It is not established by field accuracy alone. Twelve-update outcomes remain diagnostic in this screen.

Nominees require20000-update continuation, a frozen fresh test,8192-sample Heun32/128 checks and isolated measured throughput/memory before acceptance. Do not alter completed thresholds or sweep steps/widths/temperatures to rescue failures. All outcomes remain visible.

## Relationship to prior work

The conceptual basis is recurrence inside one evaluation and supervision across depth, as studied by [Looped-DiT](https://arxiv.org/html/2609.40305), [LiFT](https://arxiv.org/html/2610.05538) and [DeepFlow](https://arxiv.org/html/2503.14494). This screen does not reproduce their full methods or claim a new recurrence principle. It asks whether the professor's original block is useful within this broader mechanism under the project's established small synthetic protocol.
