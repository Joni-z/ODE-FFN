# Fixed independent-coefficient screen

Register this source before fitted candidate outcomes. One independent64×64coefficient per original K2 solve; no frame, new activation, normalization, loss, time sampler, cap, optimizer, spectral penalty, order, duration range or head change. The controls and real-arithmetic equivalence are explicit in theory.md. Earlier failed families remain stopped.

## Cohort and execution

Train five arms on each unchanged thin/thick clean128D law: `jet_c` (quadratic C), `affine_c` (.3*s*C), `free_c` (.5*s*r*C), `direct_b` (independent B) and `swiglu258`. Ten cold cells, seed0, width64, depth3, original scalar controller hidden64, batch512,5000updates, stream100000, AdamW3e-4, wd0, clip1. Reuse sealed cycle07 original K2/Swi192 controls on both laws; do not relabel them as new training. Reuse exactly the same target/source/time draws and native regression targets.

Two monitored single-thread Mac CPU workers,600s maximum per cell, terminal child codes and sampled combined RSS recorded. Check Lenovo reachability; no MBP/torch or idle GPU reservation. Save steps0/1000/2500/5000, all field/Jacobian/sample diagnostics, raw P and independent coefficient states, clocks and update histories. Parent loss/evaluation/checkpoint/Heun32 routines remain unchanged. Preserve failed snapshots/logs; never restart a live job because observation expired.

## Measured costs before outcomes

Whole-model parameter/forward/training-forward-backward dense MACs:

| Arm | Parameters | Inference MACs | Training MACs |
|---|---:|---:|---:|
| jet_c |122886|145856|427264|
| affine_c |122886|145856|427264|
| free_c |123276|146240|428416|
| direct_b |122886|121280|353536|
| Swi192, sealed |123072|120896|352384|
| Swi258, fresh |148680|146240|428416|

Costs are executed aten dense operations for one input, excluding pointwise work/optimizer work. Actual cell wall time and sampled memory remain separate. Swi258 is the smallest integer nominal width rounding to the actual channel count that covers the free-gate work; the previous Swi257 was slightly below the present C-arm budget. Direct B is explicitly cheaper, so C-versus-B is not a matched-kernel runtime claim. No width choice uses quality outcomes.

## Initialization and preflight

C-family matrices start at zero; free-head affine tensors copy the original head without extra RNG. Common original tensors/buffers/RNG must match; values and all inherited whole-model gradients must match bitwise under an active nonzero output-head probe for jet/affine/free. New C tangents need not match affine; the free common-C tangent matches the quadratic branch at its copied head. Direct B starts at the rounded native P@P: float64 functional equivalence checked to1e-10, native active complete value discrepancy must stay below1e-6absolute/1e-5relative. Its P derivatives and initial output-head kernel are deliberately not claimed identical; record the actual differences. Swi surrounding tensors/buffers pair, while its hidden functions differ.

Verify independent live-clock/full input/P/coefficient derivatives to1e-10, commutator identity and third-order small-duration error against RK4. Both signs of C are permitted. All ten actual trainer arms must pass exact three-update/resume checks with native buffers and data RNG. Verify calibration counts before ready is written. No fitted candidate quality is used to choose an initializer or modify a threshold.

## Fixed endpoint gates

Only `jet_c` and `direct_b` are proposed nominees. At update5000 each must:

1. Improve thin full and tangent oracle MSE by at least10% against **both** Swi192 and Swi258.
2. Stay within5%of the better Swi on thick full and tangent oracle MSE.
3. Stay within10%of the better Swi for full Jacobian error, tangent-output Jacobian error, full SW2 and tangent SW2 on **both** laws.
4. Reduce the geometric mean of thin/thick tangent oracle MSE by at least5%relative to **both** affine_c and free_c. This is the fixed coefficient-mechanism continuation rule, not a significance test.
5. Avoid nonfinite training/diagnostics and pass complete native/source/stream replay.

Tangent-output Jacobian error is the squared Frobenius error after projecting only its output onto the saved2D target basis, averaged over all existing five-time/64-anchor panels. It retains input-normal sensitivities; it is not the2×2tangent-input block. It is derived from unchanged native model/oracle Jacobian arrays before outcome selection. Full Jacobian metrics are still retained.

If both nominees pass, compare their geometric means of tangent errors normalized to the same Swi frontier. Prefer direct B when its score is within5%of jet C, because it executes fewer matrix products; otherwise retain the lower score. Report all outcomes. A C-versus-B difference concerns optimization coordinates, not function-class capacity or higher numerical order. A win by an ordinary control is credited to that control and blocks the claimed coefficient mechanism.

Failure closes this fixed independent-coefficient/initialization/budget setting without changing degree, rank, clock coefficient, centre, initialization, temperature, width, head, sampler or objective. A pass only registers confirmation on unchanged older Product/dependent-sign tasks and independent evaluation/cost checks. It cannot complete the useful-successor goal or justify an image-generation claim.

## Audit and publication

Independently replay40new/16reference checkpoints,50new/20reference Jacobian panels,10new/4reference sample batches,10000paired stream batches and192P/coefficient records. Compare every native field/panel/sample array bitwise where the established replay supports it; retain inherited scalar tolerance rtol1e-7/atol1e-9. Independently construct dense local polynomial outputs/derivatives at saved32-row probes for every coefficient checkpoint, including free controls and both clocks. Use float64 atol1e-10/rtol1e-8 for algebra. Verify every binding/prior seal, preserve formal output hashes, inspect figures and archive the complete executed source/terminal receipts. Publish curated results under English author identity; raw checkpoints/data/logs remain outside Git and Report1 remains unchanged.
