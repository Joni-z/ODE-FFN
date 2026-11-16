# Unchanged-gate extension of the first-layer sampling pilot

Registered before new1024-step observations. The original256/512 pilot has
three unresolved model/grid comparisons because some256-step snapshot output
errors exceed their registered limits. Every512-step snapshot qualifies, and
endpoint/G shifts are small. Retain that complete original result unchanged.

Run a uniform512/1024 comparison across all six focal models, both Heun128/129
grids and the same64sources, not only the three unresolved comparisons. Keep
the same first layer, modes, weights, targets, directions and all numerical
gates from hidden-sampling-protocol.md. The only new numerical setting is the
inner resolution1024. No model, layer, input or tolerance is selected by score.

Reuse the original512 and native arm arrays exactly, including all DOP snapshots,
after source/checkpoint/data hashes and native trajectory replay are checked.
Copy them into the new independent result directory with source bindings and
explicit reused flags. Compute every1024 reference/normal/tangential/ray arm
at both grids and retain its own five actual-trajectory snapshot batches.
Keep both ordinary controls per task as bound native results. Previous witness
arrays may be reused only by the corresponding one-source-per-bank witness;
formal runs must reuse the complete64-source formal arrays.

The numerical and directional tests remain exactly the same, now between512
and1024: pointwise endpoint limit1e-7+0.01times the fine-vs-native displacement;
bank/pooledGchange<=0.005; all DOP reference/local/output snapshot gates; and
SW2(native,full_fine)>1e-7+100*SW2(full_coarse,full_fine). The six-case prediction
still requires G_T>0,G_T>G_N,G_T>G_ray at both fixed outer grids after all gates
pass. Failed or mixed outcomes remain visible. No continuous outer-convergence
or architecture-performance claim follows from qualification.

Use two one-thread MacCPU workers in the unchanged04f11f75 environment. A
fresh exact documented witness must exercise both reused arrays and new1024
arms before formal dispatch. Preserve the original runtime and outputs; never
replace a failed256 result with its refined counterpart. This numerical study
does not replace the independent equal-dimension mechanism control.
