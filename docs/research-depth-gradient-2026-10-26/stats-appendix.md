# Units, uncertainty and verification

## Comparison units

There are six saved shared trajectories, four snapshots each, two successive training-law banks per snapshot, and eight batches of512 per bank. Training seed is0. The banks, groups and snapshots are repeated measurements of those fixed learned states; they are not independent training replicates. Bank-mean summaries project the mean gradient, rather than average the individual energy fractions. Initialization has exactly zero block gradients behind the zero head, so ratios and unit directions are undefined there.

Reported effects are exact finite-bank descriptive ratios, cosine alignments, and the prospective threshold decisions. No p-values, confidence intervals or training-population significance claims are justified by treating these repeated measurements as replicates. The figure's batch points show finite-bank variability only. Bank A→B transfer concerns first-order training-objective descent, not held-out generation quality or the result of finite Adam updates. Native Euclidean parameter coordinates are part of the definition.

## Driver checks

- 24 learned states and24 complete input-Jacobian probes.
- 384 matched shared/cloned loss and parameter-gradient cases.
- 12 final loss-direction central differences, with maximum error4.61e-10 under the recorded max(1, absolute derivative) normalization.
- Maximum saved function/input-J discrepancy0; maximum block-gradient sum discrepancy1.39e-17; common-gradient discrepancy0.
- One Torch thread;21.918s runtime; sampled peak process RSS433.09MiB; minimum available memory4.768GiB. Telemetry snapshots are not a continuous peak-memory guarantee.

## Independent NumPy replay

The [analysis script](independent_analysis.py) does not import synthetic/depth_gradients.py. It uses explicit sum/difference formulas and direct dot products rather than the original depth-basis implementation. Torch is used only to read checkpoint metadata for provenance.

It verifies2304 batch-group summaries,288 bank-mean-group summaries,144 transfers,8 P selections,384 shared-gradient sums and12 analytic finite directions. It also binds24 checkpoints,36 frozen source files and27 original inputs, reconstructs64 actual batch hash-chain links, and matches state/velocity interpolation bitwise. Every saved original output file is hashed before and after analysis. Maximum CSV discrepancy is5.107e-15.

The NumPy replay validates saved finite-difference estimates against analytic directions reconstructed from raw gradient arrays. The driver performed the actual perturbed-model evaluations. Neither the independent script nor this report claims that common-gradient or complete input-J computations were independently re-evaluated a second time.

Runtime root:

```text
/Users/mr.z/ode-workspace-prep/architecture-search/depth_gradients_20261026
```

Original data are under output/; independent-analysis/ contains independent-audit.json and display-summary.json. The former binds all original output hashes and the analysis source hash. No output was overwritten by the independent analysis. Its initial serialization-only failure is recorded in notes.md; the corrected replay completed normally.

## Decision convention

All five declared inequalities must hold at both late snapshots; none is a significance test. Only the P-retention inequality fails. Positive secondary checks are retained, without relaxing the rule. A separate amplitude/direction factorization was added descriptively after outcomes and has no eligibility role.
