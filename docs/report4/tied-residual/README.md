# Eight shared residual updates

We train a feature block that applies the same matrix to the current ISRU response eight times. Each update is `y = y + P q(y) / 8`. The midpoint reference takes four steps, with two applications of P per step. Both blocks span one unit of internal time.

The two runs use Mac mini CPU, seed 0, AdamW at 0.0003, batch size 512 and 20,000 updates. They inherit the midpoint trainer, initial weights, data stream and readout. Model sizes are 56,706 parameters for Ring8 and 58,512 for Product16. Measured generation quality at 20k determines the choice between the two architectures.

For the main table, use the existing confirmation stream 2026111800 with eight banks of 4,096 samples, 256 directions and Heun128. Pool the eight banks per task. Existing midpoint, Free ISRU and SwiGLU scores on these exact arrays can be reused. The native trainer also evaluates Heun32 on a separate test split. It is recorded separately. The experiment compares architectures at the same update count. The total training cost of the ordinary controls differs.

[Training adapter and grid](../../../studies/muon_pd_20261102/tied_residual_20261121/)

We also test a deployment shortcut on the already trained midpoint checkpoints. Four shared residual updates span the same internal duration and use four P applications per local block. We evaluate the full confirmation banks above and time both inference paths on the same 512 inputs with outer Heun32, using alternating order on one CPU thread while the training jobs run. A smaller SW2 together with lower measured latency would be a practical improvement.

The four-update Euler deployment increased pooled SW2 on both tasks. We then tried two midpoint steps on the same weights to retain the trained update rule with the same four-call budget. This follow-up uses the same sample banks and timing procedure.
