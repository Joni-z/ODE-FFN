# Population-density figure catalog

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
