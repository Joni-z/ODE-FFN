# Post-training sample-count check

The original 2,048-sample evaluation is complete and remains the registered
primary analysis. Ring8's validation-target versus test-target SW2 is 0.196125
with the original test projection directions. This is an empirical reference
distance, not a lower bound or a quantity to subtract from model scores.
It motivates checking how much conclusions depend on the finite evaluation sets.

Before viewing larger-sample scores, fix this follow-up:

- Keep all 27 previously evaluated checkpoints (15 fixed endpoints and the
  distinct validation-selected endpoints). Do not select any new checkpoint.
- Use 8,192 generated points and the first 8,192 targets from each existing
  validation/test split. No new training and no additional training seed.
- Keep the existing 128 test projection directions and Heun32 solver.
- Generate standard-normal source points using the original test source generator,
  with its first 2,048 points exactly matching the archived sources.
- Verify the first 2,048 generated points reproduce archived outputs within
  1e-4 maximum coordinate difference and SW2 within 1e-5. Small batched-matmul
  rounding differences are allowed; a mismatch stops evaluation.
- Evaluate each same generated set against both validation and test target sets.
  Report both and their mutual empirical reference distance.

This is a post-hoc measurement check. It does not replace the original endpoint
rule, establish population rankings, or justify choosing the more favorable
sample count. Both resolutions must remain visible.
