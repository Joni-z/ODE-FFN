# Statistical and numerical definitions

## Unit and fixed comparison

The unit is an independent Gaussian initial point at fixed trained weights,
not a layer, feature dimension or optimization update. Four prebound banks
contain 256 points each; prepare_density.py draws these Gaussian points
independently, without orbit replication. The same point IDs pair all models.
Target samples and projection directions are byte-identical to the preserved
reference population. Formal training uses seed 0 and 20k updates. Both focal
releases use native AdamW 0.0003, batch 512 and the unchanged clipping rule.
The ordinary references retain their recorded settings: Ring8 SwiGLU64/Muon 0.02,
Product16 SwiGLU192/AdamW 0.0003, Free ISRU/AdamW 0.0003.

## Paired intervals

The primary family is new-minus-shared over two tasks and two fields. The
secondary family is new-minus-anchored over the same cells. All eight are
reported together.10,000 RNG0 source resamples retain within-bank stratification
and use identical path indices for every column/model of a contrast. Each
99.375% percentile interval uses quantiles 0.003125/0.996875, a nominal
Bonferroni allocation across the eight comparisons. The table also retains
the previous accepted source resolutions. Positive differences favor the
reference. All eight contrasts retain their signs under the selected
previous/fine comparisons; an interval containing zero remains unresolved.

| Dataset | Field | Reference | New minus reference | Nominal adjusted interval | Previous-resolution mean | Refinement change |
|---|---|---|---:|---|---:|---:|
| Ring8 | original | Shared P | 0.007 | [-0.017, 0.029] | 0.007 | -3.11e-05 |
| Ring8 | original | Anchored | 0.008 | [-0.014, 0.031] | 0.008 | -1.99e-04 |
| Ring8 | projected | Shared P | 0.000 | [-0.026, 0.020] | 0.000 | -1.19e-05 |
| Ring8 | projected | Anchored | -0.001 | [-0.016, 0.010] | -0.001 | -1.20e-04 |
| Product16 | original | Shared P | 0.009 | [-0.151, 0.180] | 0.009 | -2.37e-07 |
| Product16 | original | Anchored | 0.256 | [0.102, 0.421] | 0.256 | 1.61e-04 |
| Product16 | projected | Shared P | 0.090 | [-0.051, 0.238] | 0.090 | 6.89e-05 |
| Product16 | projected | Anchored | 0.294 | [0.145, 0.455] | 0.294 | 2.00e-04 |

## Source qualification

Every model/field keeps 1,024 points. Original thresholds per dimension are:
chain identity 0.001, local-logdet agreement 0.001, log-ratio refinement 0.002,
relative endpoint refinement 0.001, with no nonpositive local step determinant.
Unresolved points alone are refined at 4096/8192/16384 steps. Actual new-model
resolution counts are:

| Dataset / field |2048|4096|8192|Unresolved|
|---|---:|---:|---:|---:|
|Ring8 original|991|32|1|0|
|Ring8 projected|1013|9|2|0|
|Product16 original|953|71|0|0|
|Product16 projected|965|59|0|0|

The Product16formal case was computed entirely on GPU and independently
verified before CPU adaptive refinement. All9retained CPU formal rows match
the GPU case within 1.39e-12; no host/point splicing occurred. The original
CPU formal queue's intentional-15exit is retained in its supersession receipt.
The later adaptive GPUallocation attempts never started a population child;
the original CPU process resumed its exact state and completed successfully.
No numerical gate or source set changed.

Independent NumPy reconstructs all new source merges within 3.82e-14.
A count-weight bootstrap and fsum moments reproduce 20 means/SDs and all 8 paired
intervals within 6.11e-16. The original eight-model analysis and geometry also
replay unchanged. [Independent statistics](statistics-independent.json),
[execution completion](COMPLETE.json).

## Claim register

- Keep: the native coordinate intervention changes Product16 joint KL.
  Evidence: new-minus-anchored intervals above, same function class and recipe.
  Scope: these matched trained functions; no causal percentage is assigned to
  the separately measured mean-step amplification.
- Weaken: shared P provides a general distribution-matching advantage.
  Evidence: all new-minus-shared density intervals include zero, and original
  anchored stages had lower Product16KL. Its primary SW2signal remains a
  separate outcome. Next check: one matched stage-order intervention.
- Keep as a descriptive observation: the new Product16components are broader.
  Evidence: qualified endpoints and identical target calibration. The moment
  plug-in values are not exact finite-sample KLbounds or a complete mechanism.

No confirmation time-bin profiles are exported. Full raw checkpoints, arrays
and logs remain outside Git under contrast-observations; curated values,
source hashes, settings and figure bindings are published here.
