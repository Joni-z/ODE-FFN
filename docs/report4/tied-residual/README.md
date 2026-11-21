# Eight shared residual updates

We train a feature block that applies the same matrix to the current ISRU response eight times. Each update is `y = y + P q(y) / 8`. The midpoint reference takes four steps, with two applications of P per step. Both blocks span one unit of internal time.

The two runs use Mac mini CPU, seed 0, AdamW at 0.0003, batch size 512 and 20,000 updates. They inherit the midpoint trainer, initial weights, data stream and readout. Model sizes are 56,706 parameters for Ring8 and 58,512 for Product16. Measured generation quality at 20k determines the choice between the two architectures.

For the main table, use the existing confirmation stream 2026111800 with eight banks of 4,096 samples, 256 directions and Heun128. Pool the eight banks per task. Existing midpoint, Free ISRU and SwiGLU scores on these exact arrays can be reused. The native trainer also evaluates Heun32 on a separate test split. It is recorded separately. The experiment compares architectures at the same update count. The total training cost of the ordinary controls differs.

[Training adapter and grid](../../../studies/muon_pd_20261102/tied_residual_20261121/)

We also test a deployment shortcut on the already trained midpoint checkpoints. Four shared residual updates span the same internal duration and use four P applications per local block. We evaluate the full confirmation banks above and time both inference paths on the same 512 inputs with outer Heun32, using alternating order on one CPU thread while the training jobs run. A smaller SW2 together with lower measured latency would be a practical improvement.

The four-update Euler deployment increased pooled SW2 on both tasks. We then tried two midpoint steps on the same weights to retain the trained update rule with the same four-call budget. This follow-up uses the same sample banks and timing procedure.

## Faster midpoint inference

Keep the trained four-step midpoint weights and take two midpoint steps at inference. On the Mac mini this cuts the median wall time for 512 generated samples with 32 outer Heun steps by about 40%. In the larger Heun128 evaluation, the pooled SW2 improves slightly on both tasks.

| Task | Four midpoint steps, SW2 | Two midpoint steps, SW2 | Original time, seconds | Two-step time, seconds |
|---|---:|---:|---:|---:|
| Ring8 | 0.071 | 0.069 | 0.246 | 0.149 |
| Product16 | 0.052 | 0.050 | 0.247 | 0.149 |

The timing columns are medians of four interleaved runs on one CPU thread. The two-step model has a lower bank SW2 in four of eight paired banks on Ring8 and five of eight on Product16. The largest benefit is the reduction in sampling time. SwiGLU192 still has the lowest pooled SW2 on Product16 (0.034).

To use this configuration with a loaded native `state_midpoint4` checkpoint, set the internal step count once before sampling. The local solver computes the step size from the count, so the feature evolution still covers one unit of internal time.

```python
model.eval()
for _, layer in model.ode_layers():
    layer.substeps = 2
```

Full results are in the per-task midpoint2 and euler4 JSON files in this directory. Each file identifies the weights and evaluation samples and retains all eight bank scores and individual timing measurements.
