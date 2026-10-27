# Fixed paired-geometry prediction-head compatibility screen

Register source and protocol before data preparation, preflight and formal outcomes. Read [theory](theory.md) for the mathematical motivation, raw-input bottleneck bounds, primary sources and endpoint caveat. This tests unchanged architectures in a different controlled context; it is not a new ODELayer or a rescue of cycles04–05.

## Exactly18 trajectories

Cross thin/thick128D embedded Ring8, three unchanged native FFNs (professor scalarK2, Swi128 and Swi192) and clean/shifted/direct heads. All native hidden widths64, three blocks, pre-LayerNorm, time FiLM, residual scales and final LayerNorm remain fixed. Swi128 nearly matches the ODE parameter count; Swi192 nearly matches its inference/training dense MACs. Measured raw-network128D costs:

The original FFN retains two independent bias-free P matrices and input-conditioned scalar clocks per block, tau10, range(.2,1), K2, and its middle SiLU. No new constraint or flow operator is introduced.

| Family | Parameters | Inference MACs | Forward+backward dense MACs |
|---|---:|---:|---:|
| Professor K2 |98310|121280|353536|
| Swi128 |98046|96128|278080|
| Swi192 |123072|120896|352384|

Heads add only the declared pointwise decoding. They share one capped pair target and the same frozen evaluation target. No matrix constraint, changed FFN, new head parameter, covariance input or hidden state supervision is allowed. All earlier data, evaluation and stopped settings remain immutable; no new outcome is pooled with their uncapped velocity errors.

## Data, training and evaluation

Use one fixed random orthonormal128-by2 basis, never exposed to models. Keep the radius3/variance.04 latent Ring8 law. Thin normal variance is0; thick is.04. Generate paired clean latent samples and normal draws, then cast targets to float32. Freeze65536 training targets per law,20000 full-field validation pairs, five64-anchor complete-Jacobian panels at t=.1/.3/.5/.7/.9,32 probe anchors,2048 sampling sources and two independent2048 target batches,128 unit sliced-W2 directions. Stream identifiers use SeedSequence([0,20261028,role]); the role IDs are fixed in runnable source. These are additional benchmark banks, not replacement data.

All model seeds0; all18 arms consume the same training stream100000 with batch512. AdamW3e-4,wd0,clip1,5000updates,snapshots0/1000/2500/5000. Targets are `(a/max(a,.05))*(Y-Z)`. No oracle supervision, component labels, basis, warm start or learned skip. Pair identical raw state/buffers/RNG across heads; copy common surrounding tensors across FFNs through the native builder. Record initialization and nonzero-head gradient distinctions rather than claiming equal head tangents.

Report full-bank decoded-field MSE and its tangent/normal decomposition, clean posterior-mean MSE, every time panel's full Jacobian error, training curves/clipping, actual parameters/dense work/time, and every learned original P snapshot. Compare observed errors with the **population** bottleneck bounds without asserting finite-bank errors must exceed them exactly. Heun32 uses the common capped field; retain target-target SW2, oracle-sampler SW2, tangent-space SW2 and normal sample energy. Plot samples after projection to the latent plane, explicitly as evaluation only.

## Preflight and replay

Verify Q orthogonality, paired latent samples and measured float32 subspace residuals. Verify the factorized128D oracle against an independently written dense eight-component Gaussian-conditioning oracle and complete AD Jacobian at interior times for both laws; test its endpoint extension and tangent/normal bounds. Check all head values/derivatives/targets, finite cap behavior, raw initial pairing, nonzero-head differences, exact resume for every arm and measured costs. Bind new sources/data plus all six earlier runtime seals. Run only after all checks pass.

After training independently replay all72checkpoints, reconstruct10000batch digests and every arm's snapshot RNG/Adam counters, verify90complete-Jacobian panels and18Heun32 sample batches, and reconstruct144learned P spectra (six ODE cells times four snapshots times six matrices). Preserve raw spectra, clocks and all failed checks outside Git. A mere result.json is insufficient evidence.

## Fixed interpretation and stop

The primary compatibility contrast is the ratio of ODE to the better Swi reference **within each law/head** at5000updates. Nominate no architecture from this screen. A potentially informative ODE/head interaction requires the unchanged ODE to be at least10% better than **both** ordinary controls under a skip head on thin data, stay within5% on thick data under that same head, and keep tangent-field MSE, Jacobian error and SW2 within10% of the better control on both laws. Even this is only a context-specific lead: it cannot override failures on the older Product/dependent-sign laws or complete the successor objective.

If no skip head meets that fixed condition, close this compatibility screen with no head/cap/time-sampler/basis/dimension/width/initialization rescue. Generic clean/shifted-head gains establish only the predicted bottleneck/parameterization effect. Do not turn them into a hidden-ODE claim. Report all18 cells, including the full-dimensional negative geometry control; no best-head or favorable checkpoint replaces the registered table.

Use two one-thread Mac CPU workers, process/RSS monitoring every5seconds and per-cell status every100updates;600second cell ceiling. Expected private storage is below2GiB. Mac mini has76GiB available at registration. Lenovo may be used only if reachable and separately preflighted; MBP/torch are excluded. No GPU is reserved for this CPU-sized screen. Freeze source before outcomes; no older report or run is modified.
