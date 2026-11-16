# Population-density figure catalog

## terminal-density.png / .pdf

Purpose: show all eight models and both fields on their complete common-source
density evaluation. Points are mean log-density ratios; intervals are 95%
within-bank bootstrap intervals for each fixed model. These individual-model
intervals are descriptive; use the predeclared paired contrasts to compare
Shared P and Stage P. Notice the near-tie on Ring8 and the opposite Product16
density ordering to the earlier large-bank SW2 result.

## paired-density-and-discovery.png / .pdf

Purpose: make the four hypothesis tests and the unselected discovery profile
explicit. Left: Stage-minus-Shared means and 98.75% paired intervals; positive
would favor sharing. Both Ring8 intervals include zero and both Product16
intervals favor Stage P. Right: pointwise 95% intervals on the eight signed
bin contributions, using only the 256 sources of Product16 discovery bank0.
Those time-bin intervals are exploratory, not eight independent hypothesis
tests. No window is selected because the primary explanation has the wrong
sign. Confirmation-bank profiles are not used for selection.

Full-figure data/source bindings: figures/population-figure-bindings.json.
Both figures were rendered and inspected for labels, signs, units and clipping.

## product16-density-width.png / .pdf

Purpose: compare qualified joint density errors with two measurable features
of the same generated endpoints. All three panels use the projected Product16
fields and the same 1,024 Gaussian sources per frozen model.

Left: estimated joint KL; bars show mean and one Monte Carlo SE. Use the
separate adjusted paired intervals to assess the Shared/Stage comparison.
Middle: posterior-responsibility conditional variance relative to target
variance; the ideal value is 1. Points avoid implying a zero-based bar ratio.
Right: pairwise posterior-sign covariance RMS; the dashed line is the finite
native target sample value, not a population dependence threshold.

Notice that Shared P has both higher KL and broader components than Stage P.
The pairwise dependence statistic is near the native target sampling floor
for every model. Width is a concrete diagnostic direction; neither that
correlation nor this pairwise statistic establishes the full causal mechanism.

Sources: product-density-statistics.json and product-geometry.json; SHA256
bindings are in figures/product16-density-width-bindings.json. The image was
rendered and visually checked for labels, units, reference lines and clipping.
The source is studies/muon_pd_20261102/density_transport_20261116/plot_product_geometry.py.
