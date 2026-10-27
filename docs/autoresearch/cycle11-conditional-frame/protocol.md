# Registered single conditional-frame screen

This is one new operator-conditioning hypothesis, not a new head, width, clock range, time sampler, solver-accuracy rescue or combination of tricks. Keep cycle07's immutable thin/thick128D data, clean head, cap.05, hidden width64, depth3, scalar clock tau10/range(.2,1), two solves and middle SiLU. All earlier stopped settings stay stopped.

## Exactly eight new cold trajectories and four sealed references

For each law train conditional-frame K2, global-frame K2, conditional-frame K1 and Swi257. Reuse final and intermediate original K2/Swi192 clean trajectories from sealed cycle07. The three frame arms all add one63-by64 weight and63bias per layer, zero initialized without RNG draws. All inherited tensors/buffers match the original. Global-frame features are ones rather than input features, retaining the same tensors/inference work while making the frame absorbable into P. This is intentionally a redundant parameterization control, not an equal-rank initial-kernel claim.

Use seed0, stream100000, batch512,5000updates, AdamW3e-4/wd0/clip1, snapshots0/1000/2500/5000. Use unchanged native drawing, targets, evaluation, checkpoint/resume and Heun32 generation. Measured cold128D costs are:

| Model | Parameters | Dense inference MACs | Dense forward+backward MACs |
|---|---:|---:|---:|
| Conditional-frame K2 |122880|145472|426112|
| Global-frame K2 |122880|145472|401920|
| Conditional-frame K1 |122880|120896|352384|
| Swi192, sealed parameter control |123072|120896|352384|
| Swi257, compute control |148098|145664|426688|

Swi257 is the smallest nominal size whose rounded internal width makes its dense work at least the candidate's; it is chosen arithmetically before outcomes, not by a quality search. The global frame has no feature gradient, so its backward work is lower. It is a parameter/inference control, not a matched training-work or initial-kernel control. Pointwise/frame overhead and actual runtime are reported separately; no exact wall-time match is claimed.

Before formal training, verify Q orthogonality, similarity eigenvalues/SVD, explicit dense-polynomial and ordinary-gated equality, full state/parameter derivatives, commutator differential and local-response witness. At zero frames require bitwise original K2 values and all inherited state/parameter gradients on an active-head probe; RNG/common buffers remain exact. Verify native exact resume in every new arm and replay all reused references. Additional frame tangents and the K1 hidden/head differences are retained.

Publish full-bank field/tangent/normal errors, all five complete-Jacobian panels, full/tangent SW2, normal sample energy, training curves/clipping, P spectra at every snapshot and true frame/clock diagnostics. At final weights zero only the frame parameters in a copy of each conditional-frame K2/K1 model, and record full-bank output displacement and error changes. This intervention is descriptive, never a nominee or a best-checkpoint selection.

## Fixed acceptance and stop

The sole candidate is conditional-frame K2. Require at5000updates: thin field MSE<=.9 times **each** Swi control; thick field MSE<=1.05 times the better Swi; on both laws tangent MSE, complete Jacobian error, full/tangent SW2<=1.10 times the better Swi. Also require field MSE<=.95 times both global-frame K2 and conditional-frame K1 on both laws. An improvement over the original alone does not qualify. Keep all comparisons and signed outcomes, without selecting a head, time bin or favorable checkpoint.

Failure closes this one frame/initialization/order setting. No extra reflection, altered frame scale, width, clock range, temperature, internal frame reevaluation or new loss follows as a rescue. A passing context screen only schedules an independently registered confirmation on unchanged older Product/dependent-sign laws with adequate ordinary controls; it does not establish an ODE-specific mechanism, image result or completed successor.

Two one-thread Mac CPU workers, per-cell600s ceiling, process/RSS sampling every5s and status every100updates; no GPU reservation. Lenovo is optional only after connectivity and a separate preflight, MBP/torch are excluded. Preserve every failed source/log; immutable hashes and prior seals must match before/after. Independently replay all32new checkpoints,40new Jacobian panels,8sample batches,10000stream hashes and144new P snapshots; also verify every reused reference. Publish code/curated evidence, keep raw arrays/checkpoints outside Git, and seal before final interpretation/commit.
