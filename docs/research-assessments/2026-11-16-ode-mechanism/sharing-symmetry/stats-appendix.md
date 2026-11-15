# Comparison units and paired effects

Training seed remains 0; this follow-up adds no training. The repeated
evaluation unit is one independently generated 4096-base-point bank, with
all eight transformations attached. There are four paired banks, conditional
on the fixed checkpoints and 256 fixed slicing directions. Orbit members
within a bank are correlated and are never counted as independent repeats.

Differences are stage minus shared; positive favors sharing. The table
multiplies differences by 1,000 so three decimals retain useful detail.

| Task | Field | Mean bank difference x1,000 | SD x1,000 | Approx.95% interval x1,000 | Shared wins |
|---|---|---:|---:|---|---:|
| Ring8 | Original | 23.889 | 0.664 | [22.833,24.945] |4/4|
| Ring8 | Projected | -2.694 | 0.802 | [-3.971,-1.417] |0/4|
| Product16 | Original | 8.260 | 1.360 | [6.096,10.424] |4/4|
| Product16 | Projected | 7.061 | 1.471 | [4.720,9.403] |4/4|

Intervals use paired Student-t means with 3 degrees of freedom and assume
a regular distribution of bank differences. Four observations cannot
validate that assumption. The distribution-free two-sided exact sign test
has raw p=0.125 for each contrast; Holm correction over these four comparisons
gives 0.500. No conventional significance claim follows from these tests.
The observed reversal and retained direction describe these fixed checkpoints.

[comparison.json](comparison.json) records all four differences per contrast,
pooled effects separately, and the original/projected mean±SD for each model.
The pooled statistic is nonlinear and is not the mean of bank scores.
Projection's eight per-model effects are descriptive additional interventions;
they are not another confirmatory testing family.

The exact MSE decomposition concerns a group-orbit average at fixed
field points. Its numerical residual is checked separately. SW2gap changes
are not an additive mediation fraction. Product16 removes only reflections
on axes0,7,15; its selected eight-bin balance does not test all 65536 sign
patterns or the full symmetry group.

The source/target stream differs from earlier iid evaluation cohorts.
Report comparisons only within their bound arrays and sampling protocol.
CPU/CUDA actual-entry witnesses are engineering checks, explicitly marked
witness=true, and are excluded from all scientific tables.
