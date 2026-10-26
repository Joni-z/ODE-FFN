# Interaction of shared hidden dynamics and complete potential prediction

Registered after the exact composition diagnostic, before all new cold outcomes. The earlier direct-readout recurrence setting and the independent-layer potential setting remain stopped. This is one missing architectural factor interaction, not a new scalar potential, partial-adjoint rescue or continuation of those checkpoints.

## Mathematical construction

With fixed outer input/time, one shared residual field advances hidden states:

```text
h0=Encode(x,t)
h_(k+1)=h_k + ds*F_theta(h_k;t), k=0,1,2
G(x,t)=Decode(h3,t)
A(x,t)=grad_x[.5*dot(x,G(x,t))].
```

The step ds is the existing1/sqrt(3), so the fixed internal interval is sqrt(3). This is three Euler updates of a shared autonomous-in-depth field, conditioned on outer t. The original ODEFFN's two scalar K2 solves and middle SiLU parameterize F; SwiGLU uses the same surrounding construction. No internal-depth embedding, extra loss or integration change. Full adjoint propagation remains intact and guarantees the complete A field is conservative; it does not prove numerical refinement improves quality or give local blocks an equivalent guarantee. Neural-ODE parameter sharing and potential readouts are established categories, not claimed novel separately.

## Matched control and Phase A

Native independent layers have different initial hidden features from shared layers. Therefore include cloned independent layers: three distinct copies of the first block, initialized to exactly the shared block. Cloned/shared raw and complete-potential functions are initially identical, including a nonzero-head probe. Shared parameter gradients equal sums of the cloned gradients at tied states. This isolates parameter sharing from merely repeating the same initial block.

Complete the2x2 layout/readout factor for each of ODE K2, SwiGLU128 and192 on Product16:

- New: shared gradient and cloned gradient for all three families (six cells).
- New: shared direct and cloned direct SwiGLU128 (two cells), absent from the earlier recurrence screen.
- Reuse: shared/cloned direct ODE and SwiGLU192 final-only controls (four cells), after actual source, bank, initial-state, stream and output verification.

Keep native direct/A/B controls and wide direct SwiGLU from the verified orientation table as additional practical context. No favorable generic setting is omitted. All new runs are cold, seed0,5000updates,batch512,AdamW3e-4/wd0/clip1, width64, three executed blocks, common cached-bank pair FM and actual uniform-time streams. There is no intermediate prediction supervision.

Use the existing RecurrentDepthNet and generic potential wrapper, preserving actual parameter objects; no duplicate shared state-dict parameters or dormant copies. The shared ODE has two unique P matrices executed three times each, whereas cloned ODE has six unique matrices. Preserve spectra per unique P and all six actual solve visits at every snapshot. Save all normal checkpoints, histories, full field banks, five512-input derivative panels, Heun32 SW2/2048 common starts and actual matrix costs. Two one-thread CPU workers and10-second telemetry.

## Preflight and independent audit

Verify nonzero-head complete shared/cloned functions and tied-gradient addition, exact initial tensors/buffers/RNG, native-wrapper compatibility against actual old potential weights, executed counts, repeated visits, actual3-update trainers/reloads and child repeats. Before interpreting endpoints, regenerate full input RNG/hash streams; replay native fields/Jacobians/samples and all spectra. Independent complete raw-Jacobian reconstruction and finite differences verify the new potential outputs and their derivatives. The new wrapper adds one raw namespace; explicitly map that namespace when reusing old states, never load loosely.

## Phase A eligibility

Only shared-gradient ODE can nominate a further two-task replication, under either predeclared route:

1. Quality lead: at least10% lower field MSE than every generic control and at least5% better than its own shared-direct, cloned-gradient and native-gradient ODE controls.
2. Parameter-efficiency lead: field MSE within5% of the best generic, within5% of cloned-gradient ODE, at least25% fewer parameters than the cheapest generic within that same quality band, and dense inference MACs no more than5% above the least-expensive generic within the band.

Both routes also require Product16 Jacobian error and SW2 within10% of the best generic, finite/audited evidence and all newly trained/reused generic controls included. Report parameter count and matrix work separately. Concurrent training time is not isolated throughput. The quality/parameter routes are prospective selection criteria, not significance tests or retrospective reinterpretation of old results.

Record the within-family sharing ratio under each readout and their ratio-of-ratios descriptively. Do not require a favorable interaction by dropping controls, and distinguish a cloned-initialization gain from a sharing gain. If neither eligibility route passes, stop this exact factor without more tasks, supervision, recurrence depth, step size, potential coefficient, learning rate or training-length rescue. A pass only triggers separately registered Gaussian/Ring replication, then new holdout/cost evidence; it is not an accepted architecture or full JiT image-generation result.
