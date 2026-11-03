# Product16 closes the common-bank comparison

Both registered task cohorts are now complete. Ring8 retains a fitting lead
for ISRU-D with sampling close to selected Swi64; Product16 favors the ordinary
references. This makes the fitting/sampling relationship the next question.

All entries below use the fixed20k checkpoint and Heun128. Sampling means and
SD refer to the same eight common4,096-sample banks within a task. Native field
MSE is independently recomputed from its original validation arrays.

| Task | Model and optimizer | Oracle MSE | SW2 mean | Bank SD |
|---|---|---:|---:|---:|
| gmm_ring8_2d | ISRU-D / adamw 0.0003 | 0.009773 | 0.171104 | 0.047210 |
| gmm_ring8_2d | Free ISRU / adamw 0.0003 | 0.009998 | 0.179305 | 0.047281 |
| gmm_ring8_2d | Swi64 / adamw 0.0003 | 0.013717 | 0.194535 | 0.065031 |
| gmm_ring8_2d | Swi64 / muon 0.02 | 0.011279 | 0.171453 | 0.053104 |
| gmm_ring8_2d | Swi192 / muon 0.02 | 0.011231 | 0.183991 | 0.057236 |
| rotated_product_bimodal_16d | ISRU-D / adamw 0.0003 | 0.016059 | 0.089003 | 0.006955 |
| rotated_product_bimodal_16d | Free ISRU / adamw 0.0003 | 0.015356 | 0.085964 | 0.005464 |
| rotated_product_bimodal_16d | Swi64 / adamw 0.0003 | 0.020159 | 0.085821 | 0.003689 |
| rotated_product_bimodal_16d | Swi64 / adamw 0.001 | 0.018781 | 0.086536 | 0.003913 |
| rotated_product_bimodal_16d | Swi192 / adamw 0.0003 | 0.015668 | 0.083505 | 0.004632 |

## What the Product16 control adds

ISRU-D mean SW2 is0.089003, versus Free ISRU0.085964, same-rate AdamW
Swi64 0.085821, selected Swi64 0.086536 and selected Swi192 0.083505.
The paired ISRU-D-minus-Free difference is+0.003039 (SD0.003018); the
candidate wins one of eight banks. The difference against selected Swi192
is+0.005498, again with one bank win. The table retains the same-optimizer
comparison separately from the selected reference.

ISRU-D still has lower field error than same-rate Swi64 (0.016059 versus
0.020159) while sampling is worse. Free ISRU and Swi192 have lower field error
as well. The Ring8 field lead therefore does not transfer to every ordinary
reference on Product16.

## Measurement and next decision

All480 saved generated arrays independently reproduce the double-precision
SW2 calculation within5.56e-17. All samples are finite. All ten native final
anchors reproduce original generated/target arrays bitwise. Thirty physical
checkpoint records and source/cache identities are verified. The original25
source files per host are unchanged after execution. No new training runs.

The earlier [Ring8-only snapshot](../cycle14-results-ring8/analysis-report.md)
and its post-outcome pooled precision diagnostic remain intact. Their pending
Product16 status describes that earlier reporting time. This complete cohort
retains the original primary bank estimator and all three checkpoints.

Next, compare the three Ring8 models at15k/20k on a common oracle transport
path, including time/direction profiles and frozen oracle-switch interventions.
The [cycle15 registration](../cycle15.md) fixes this finite question. No new
architecture or image run is selected. The completed Report2 PDF/source remain
the submitted reporting snapshot; these findings belong to the next research
update.
