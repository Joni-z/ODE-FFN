# Stage5: one controller across the two ODE solves

Registered before the training outcomes. Retain K1, the intermediate SiLU, all existing time conditioning, the optimizer, and the loss. The only architectural axis is how the two solve durations are produced.

## Why this question follows the evidence

Deleting the middle SiLU worsened both scalar/vector Ring8 models, so keep it. Anchored derivative attribution shows that clocks on both solve positions can matter; it does not justify simply discarding the second solve's input dependence. Instead ask whether the same input-dependent duration can drive both solves. This retains a route by which the input changes the second effective operator, but computes one controller rather than two.

Neither this idea nor general gate sharing is claimed as new. [Gated Attention](https://arxiv.org/html/2505.06708v1) treats gate position and sharing as experimentally distinct choices. The authors' [Mamba-2 systems discussion](https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems) separates parameterization from actual runtime. [Mamba-3](https://arxiv.org/html/2603.15569v1), ICLR2026, likewise tests a quality/latency frontier and specific recurrence-derived capabilities. Our inference from these sources is methodological: an ODE-inspired description requires a mathematical operator, competing mechanisms and measured cost. These sequence-model results do not establish our FFN design.

## Exact K1 operator

Let s(h) be the existing64-channel sigmoid controller (MLP hidden32), D=diag(s(h)), and phi=SiLU. Define

```text
z = (I + D P1) h
u = phi(z)
F(h) = (I + D P2) u.
```

The same numerical D is reused, not a detached value: gradients flow from both uses into its controller. P1 and P2 remain distinct learned matrices. This is an internal two-stage frozen-controller map; it is not an exact autonomous ODE flow, a higher-order solver, or the external FM sampling interval.

For Js=ds/dh and S=diag(phi'(z)), exact differentiation gives

```text
J_F = (I+D P2) S (I+D P1)
      + [(I+D P2) S diag(P1 h) + diag(P2 u)] Js.
```

The shared controller therefore still changes both solves' complete derivatives. The controller contribution has rank at most32 per block, compared with a possible sum of two rank32 contributions with independent controllers. This is a restriction of a hidden block's local differential, not a proved bottleneck in the full network. Three blocks, conditioning and normalization prevent direct identification with the target Jacobian.

Reuse also ties the numerical duration applied to two different row bases P1 and P2. It can fail if those solves need incompatible relative rates, or if evaluating the controller after the first transformation supplies information that cannot be economically predicted from h. The tied-weight/recomputed-value control below distinguishes this from a simple parameter-sharing effect. We do not increase controller width if this screen fails.

## Five arms and matched budgets

1. `vector_k1`: previous independent controllers g1(h), g2(u).
2. `vector_tied`: one controller network g evaluated on h and again on u; identical weights, two computed values.
3. `vector_reuse`: one controller evaluation g(h), same duration used twice.
4. `swiglu_budget`: nominal hidden96, actual SwiGLU hidden64.
5. `swiglu`: previous nominal hidden128, actual hidden85.

At dim2, reuse has69282 parameters and67904 forward linear MACs; the budget SwiGLU has69570 parameters and exactly67904 MACs. Full vector has81858 parameters/80192 MACs, full SwiGLU81792/80000. Tied has the same parameter count as reuse and the same forward MACs as the full vector. The controller saving is12576 parameters and12288 MACs per whole model. Elementwise operations, training/optimizer work and measured inference latency are reported separately from MACs.

Construct the full vector reference before tying/deleting its second controller so P1/P2, all non-gate weights, the retained first controller and subsequent RNG states remain paired exactly. Initial second-controller functions deliberately differ. No additional initialization or spectral change is made. A controller reused twice is counted once as parameters, twice if it is actually evaluated twice as compute.

## Training and advancement rule

Use the original v1 Gaussian2D, Ring8 and Gaussian16D files, seed0,5000updates, batch512, width64, depth3 and the existing training settings. CPU2 workers and10-second telemetry; freeze and mirror source before launch. Primary endpoint is final random-time validation oracle MSE; retain common-panel field/Jacobian errors, sample SW2 and trajectories. No test split is read.

A reuse candidate warrants the longer-budget/nonlinear check through either route:

- Quality retained at lower cost: Ring8 oracle error is at most5% above full SwiGLU, with neither Gaussian control worse by more than5%.
- Better fit at matched cost: Ring8 oracle error is at least10% below budget SwiGLU, with neither Gaussian control worse by more than5% versus that budget reference.

Both routes also require Ring8 common-panel Jacobian error and validation sample SW2 not to worsen by more than10% against their respective reference. These are engineering screens, not significance claims. A tied-only gain cannot be credited to compute reuse. Failed reuse stops this branch; do not add a width sweep, new gate function, spectral cap, loss or activation to rescue it.

Passing a screen is not goal completion. It requires a frozen continuation to20000, a nonlinear higher-dimensional check, independent fresh evaluation inputs because the historical test banks are exposed, an actual latency/throughput comparison, and an independently recomputed analysis.
