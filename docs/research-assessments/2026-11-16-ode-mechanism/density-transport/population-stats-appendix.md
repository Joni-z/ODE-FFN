# Population-density statistics: Product16 completed

- Unit: one independent Gaussian initial point at fixed trained weights.
- Four common banks of 256 points; 1,024 points per model/field. All four models,
  original/projected fields and all original source identifiers are retained.
- Numerical integration starts at 1,024/2,048 steps. Only unresolved sources
  continue to 4,096/8,192/16,384 under the same five registered gates.
- Every Product16 cell qualifies. Subset selection, density identities, bank assignments
  and all adaptive merges are independently reconstructed without Torch.
- Primary estimator: mean terminal signed density-source integral. Independent
  divergence/endpoint-density reconstruction is a numerical control.
- Paired contrasts use Stage P minus Shared P on common initial points.
- Bootstrap: 10,000 resamples, RNG 0, stratified within the four source banks.
  Both models and resolution columns retain identical sampled path indices.
- Two Product16contrasts are reported from the predeclared four-contrast family.
  98.75% marginal intervals use a Bonferroni family allocation. Missing Ring8
  contrasts are not silently removed from the family.
- MC SE is sample SD divided by sqrt(1,024). Figure error bars show one MC SE;
  they are not the adjusted paired intervals.
- No trimming, clipping of negative observations, favorable checkpoint choice
  or exclusion of difficult paths is used.
- Conditional component moments and pairwise sign statistics are exploratory
  frozen-geometry measurements. The 1,024 native target draws calibrate their
  finite-sample floor. A population moment-based KL bound is derived separately;
  its plug-in estimate is not claimed to be an exact finite-sample bound.
- No confirmation-bank time profiles have been exported or inspected for
  window selection. The density explanation's sign is rejected before any
  proposed oracle-window intervention.

Inputs and exact numbers: [statistics](product-density-statistics.json),
[geometry](product-geometry.json), [audit](product-adaptive-independent.json).
