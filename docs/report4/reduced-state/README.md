# Reduced-state midpoint and solver ablations

This study asks which parts of the four-step midpoint block account for its performance, and tests a smaller state for the internal solve. Ring8 and Product16 use the established seed-0, batch-512, 20,000-update protocol. The four-step midpoint reference is reused.

- Frozen-response midpoint keeps four steps and evaluates its diagonal at the entrance. This isolates the effect of refreshing the response during the solve.
- Two-step midpoint recomputes the response during training and sampling with two internal steps. This tests whether the four-step training budget is needed.
- Reduced-state midpoint learns two maps of shapes `32 × 64` and `64 × 32` in place of each `64 × 64` matrix. The response is applied after projecting into a learned coordinate system. The solver takes four explicit midpoint steps.

For the reduced-state model, the full drift is `U q(V y)` with the same ISRU response `q`. Write `y = x + U a` and solve `a' = q(V x + V U a)`, starting from zero. Explicit midpoint commutes with this affine map step by step. Matrix work consists of one input and output projection, one small matrix product per solve, and eight narrow matrix-vector products per token. At width 64 and rank 32 this costs 12,288 dense MACs per token plus one shared `VU` contraction, compared with 32,768 for the full midpoint solve. The rank-32 model can only move the state within the column space of U. Its initialization rescales U to match the expected squared norm of the unfactorized linear drift under the original initialization.

The primary outcome is pooled SW2 on the original eight confirmation banks with outer Heun128. The original selection stream is recorded alongside it. Each variant also continues from its 20,000-update checkpoint to the measured cumulative training FLOPs of the native four-step model. [Operation counts and update targets](budget.json).

[Grid and implementation](../../../studies/muon_pd_20261102/reduced_state_20261122/)
