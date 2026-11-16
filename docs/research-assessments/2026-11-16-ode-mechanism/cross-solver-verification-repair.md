# Verification input-binding correction

All eight full native crossed evaluations completed with actual exit 0. The
original verification queue stopped after the first confirmation check because
verify_cross_solver.py compared confirmation field predictions with a selection-
only FP64 reference. The input states differ. Its precision assertion correctly
failed; the cross-solver samples and native restoration anchors are unaffected.

Keep the original queue terminal (overall exit 1), logs, failed verifier output,
and all eight native output folders unchanged. No samples or training are rerun.
The corrected check computes an FP64 discrete field on each evaluation's ACTUAL
asset split, with source-binding, asset, checkpoint, inference solver, field_x
and field_t hashes. It retains the original 3e-5 absolute/relative precision gate
on all 4,096 inputs. Independent NumPy then reconstructs all field, SW2 and case
metrics. A wrapper rechecks hashes of every original sample/output before/after.

The repaired command audits existing full outputs; it is not new generation.
The original source freeze 9bab560 and its failed terminal remain in the record.
