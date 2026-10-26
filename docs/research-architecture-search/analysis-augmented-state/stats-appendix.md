# Verification and scope

The fixed Phase A used six arms, three tasks, seed 0 and 5000 updates. Data,
pair-velocity objective, uniform time sampler, output head, optimizer settings,
Heun32 sampler and hash-bound non-FFN initialization were preserved. Both clocks
retain `.2+.8*sigmoid(logit/10)`. The new family's `legacy` option allows its own
constant-clock initializer; it does not mean random initial clocks.

The frozen runtime contains 604 files: the actual Stage23 595-file parent plus
nine additions and four reviewed replacements. Reviewed code/protocol, all
source bytes and exact trees are bound before launch. Formal training ran once
from the immutable ready command at approximately 2026-10-24 19:23:33.73 UTC.
Queue PID 64832, caffeinate 64840 and supervisor 64841 were captured by root.
An independent live observation at 19:25:15 UTC records workers 64908/64912,
real 10-second telemetry and 5.1165 GiB available memory. Training used two
one-thread CPU workers and reserved no GPU.

The v1 child completed 12/12 at 19:26:33.630607 UTC, Product16 completed 6/6
at 19:28:10.729547, and the queue completed at 19:28:14.993098. The independent
terminal observation found the queue gone. No task was restarted. The main
analyzer ran once after all 18 cells were terminal, session 94474, exit 0.
All conclusions below use complete audited outcomes.

| Check | Verified result |
|---|---:|
| Fresh native checkpoint audit | 108 |
| Fresh five-time saved arrays | 90 |
| Reloaded final checkpoint field/Jacobian replays | 18 |
| Maximum field/Jacobian replay discrepancy | 0 / 0 |
| Exact SwiGLU128/SiLU128 historical saved states | 36 |
| Exact cross-arm checkpoint data-RNG comparisons | 90 |
| Eligible historical SwiGLU96/GLU128 checkpoints / arrays | 36 / 30 |
| Fresh initialization/reference checks | 18 |
| Training-curve rows / internal layer records | 198 / 792 |
| P/factor checkpoint records / full-spectrum entries | 432 / 34,560 |

SwiGLU128 reproduces Stage23 and SiLU128 reproduces Stage19's `ffn_silu` saved
model, optimizer, buffers and RNG states. These are deterministic compatibility
reruns, not independent repetitions. Historical SwiGLU96/GLU128 are separately
bound by source, checkpoint, config, dataset, endpoint-array, native summary and
comparison hashes. Their baseline-reference hashes alone were not treated as
outcome provenance. The generic frontier takes the best eligible value for each
metric, including these historical controls.

The independent checker reproduces all 24 raw endpoint rows and the registered
threshold decisions without importing the analyzer's screen function. It
verifies all 343 main-analysis input bindings and the exact source trees:
604 current, 595 Stage23, and the Stage19 SiLU source. No formal process remains.
The decision is `STOP_FIXED_STATE_ALLOCATION`, with no practical leads and no
augmentation-allocation evidence. No fresh holdout, Phase B timing, transfer or
counterfactual intervention was selected.

Preflight attempt 3 is authoritative: all 72 tests, four full-width float32/float64
initialization cases, 18 legacy comparisons over 17 distinct effective settings,
three full-trainer 3 versus 2+resume3 identities, two generic trainer identities,
all references/configs, actual MAC/call counts, native hidden captures and real
monitoring pass. The duplicate legacy setting is the constant-paired vector map.
Native zero-head kernels pair by masking. Under a nonzero head, common responses
match within declared precision tolerances while extra V2 responses are live;
full parameter kernels are not claimed equal.

Attempt 1 stopped on a harness-only overstrict float32 difference-of-large-norms
identity. The reviewed full energy identity is enforced in float64; individual
float32 output, input-Jacobian and common-parameter tolerances were not loosened.
Attempt 2 stopped on a three-entry `F.pad` tuple in the independent reconstruction.
The frozen helper already used the correct four-entry tuple. Both failed scripts
and logs remain archived; neither changed model code or started formal training.
Dependency metadata preceded the final ready receipt, which remains unchanged.

Seventy-two saved-checkpoint native probe forwards verify actual logged inputs,
activations, full padded P statistics and probe-mean VDU statistics. Matrix-entry
and factor evolution is recorded without interpreting independently sorted
eigenvalues as tracked modes. The new layers remain outside legacy `ode_layers`,
so their legacy anchored Jacobians equal native Jacobians and are not derivative
ablations. Native complete-field Jacobian evaluation remains valid.

Three figures were visually inspected. Only label placement in the cost/extra-path
figure was regenerated from the audited CSVs; the original PNG/PDF and the
standalone plotting script are retained. No numerical result or analyzer audit
was rerun. Report1 remains unchanged and all work is local/unpushed.
