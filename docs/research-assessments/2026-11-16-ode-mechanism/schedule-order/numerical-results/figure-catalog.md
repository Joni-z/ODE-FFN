# Figure catalog

## resolution-comparison.png / .pdf

Purpose: test whether the pair ordering depends on trained internal resolution
or outer integration. Left panels use internal 4/8/16with Heun128 fixed; right panels
use Heun128/256/512 with internal 4 fixed. Markers are pooled32,768-source scores;
lines connect observed points. Shared P has no internal-8 marker. Its sparse
refinement is reused only after checkpoint and array hash matching.

Observe: Ring8 palindrome stays below alternating; Product16 has the opposite
ordering and both schedules worsen under internal refinement. This rejects a
task-general cancellation gain and preserves the Ring8within-pair effect.
All curves are fixed-weight interventions, with different inference work.

## paired-resolution-effects.png / .pdf

Purpose: retain all bank-level differences behind the pooled comparison.
Dots are eight common-bank differences, black markers their mean, blue
segments marginal 95%empirical bank-bootstrap intervals. Negative favors
palindrome. These intervals apply to bank means, not to pooled SW2.

Observe: Ring8 means remain negative and Product16 means positive across every
setting. Appendix contains the ten-contrast correction and sign-test results.
This figure quantifies sampling variation at the fixed trained checkpoints.

Both actual PNGfiles were visually inspected; both PDFfiles were exported
from the same Matplotlib figures. Input hashes and full statistics are bound
in comparison.json and native-COMPLETE.json. No schematic or invented curve
is used.
