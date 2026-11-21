# Reduced-state midpoint and solver ablations

**Two-step training and inference makes the best use of the tested midpoint budget on Product16. Four-step midpoint remains the best training choice on Ring8.** Each new variant completed both its 20,000-update checkpoint and its cumulative-compute checkpoint. The table reports pooled confirmation SW2. Lower is better.

| Model | Ring8 at 20k | Ring8 at matched compute | Product16 at 20k | Product16 at matched compute |
|---|---:|---:|---:|---:|
| Four-step midpoint | **0.071** | **0.071** | **0.052** | 0.052 |
| Frozen-response four-step midpoint | 0.084 | 0.151 | 0.056 | 0.057 |
| Two-step midpoint throughout | 0.073 | 0.138 | 0.053 | **0.042** |
| Reduced-state four-step midpoint | 0.111 | 0.128 | 0.077 | 0.101 |

At 20k, refreshing the diagonal improves SW2 on both tasks and both evaluation streams. Two-step training gives nearly the same confirmation SW2 while reducing arithmetic per update by about 43%. On Product16 it improves to 0.042 after 35,049 updates, with lower SW2 than four-step midpoint on seven of eight confirmation banks and on the pooled selection stream (0.048 versus 0.054). The matched Free ISRU and SwiGLU endpoints are 0.043 and 0.047. The original 20k SwiGLU score is still lower at 0.034. [Ordinary controls](../equal-compute/results.json).

On Ring8, continuing the two-step model from 20k to 35,219 steps reduces oracle MSE from 0.011 to 0.007. The eight-sector mass error rises from 0.031 to 0.053 and pooled SW2 rises from 0.073 to 0.138. The selection stream shows the same deterioration.

The 32-dimensional solve uses 46% of the original arithmetic per training update and reduces Mac mini sampling time by 23–26%. The exact reduction itself saves 15–16% compared with applying the same learned U and V at every stage in full coordinates. Its matched-compute SW2 is worse than both refreshed midpoint depths on Product16 and worse than the original four-step midpoint on Ring8. We retain the full-width states for the two recommended configurations. The two-step model cuts CPU sampling time by 37% on Product16. Timing uses the median of five interleaved Heun32 runs of 512 samples with one CPU thread. [Timing records](benchmark.json).

All six trajectories ran on Mac mini. The two checkpoints per trajectory each completed both eight-bank Heun128 streams. Independent NumPy reconstruction of all 12 confirmation SW2 scores agreed within 3e-17. [Full results and checkpoint hashes](results.json).

This study asks which parts of the four-step midpoint block account for its performance, and tests a smaller state for the internal solve. Ring8 and Product16 use the established seed-0, batch-512, 20,000-update protocol. The four-step midpoint reference is reused.

- Frozen-response midpoint keeps four steps and evaluates its diagonal at the entrance of each solve. This isolates the effect of refreshing the response during the solve.
- Two-step midpoint recomputes the response during training and sampling with two internal steps. This tests whether the four-step training budget is needed.
- Reduced-state midpoint learns two maps of shapes `32 × 64` and `64 × 32` in place of each `64 × 64` matrix. The response is applied after projecting into a learned coordinate system. The solver takes four explicit midpoint steps.

For the reduced-state model, the full drift is `U q(V y)` with the same ISRU response `q`. Write `y = x + U a` and solve `a' = q(V x + V U a)`, starting from zero. Explicit midpoint commutes with this affine map step by step. Matrix work consists of one input and output projection, one small matrix product per solve, and eight narrow matrix-vector products per token. At width 64 and rank 32 this costs 12,288 dense MACs per token plus one shared `VU` contraction, compared with 32,768 for the full midpoint solve. The rank-32 model can only move the state within the column space of U. Two inputs with the same Vx receive the same local update. The directions in the kernel of V therefore pass through that local solve unchanged. The next solve has its own U and V. Initialization rescales U to match the expected squared Frobenius norm of the unfactorized linear generator under the original initialization.

The primary outcome is pooled SW2 on the original eight confirmation banks with outer Heun128. The original selection stream is recorded alongside it. Each variant also continues from its 20,000-update checkpoint to the measured cumulative training FLOPs of the native four-step model. [Operation counts and update targets](budget.json).

[Grid and implementation](../../../studies/muon_pd_20261102/reduced_state_20261122/)


## Reproduce on Mac mini

Use the existing synthetic Python environment and data in `/Users/mr.z/ode-workspace-prep`. Run `check.py` in the study directory to verify the affine reduction against the full-width formula, including derivatives. Run `monitor.py <work-directory>` to train and evaluate all six 20k models. `matched_compute.py <work-directory>` measures the budgets. Passing a cell ID as the second argument continues that cell from 20k and evaluates the matched endpoint. `collect.py` and `benchmark.py` consume the resulting checkpoints.
