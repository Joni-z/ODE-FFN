# Statistical and numerical scope

- **Primary metric:** SW2, root mean squared sorted-projection distance, lower
  better. Fixed 2,048 source and target points, 128 fixed directions.
- **Training units:** one seed (0) for each model/dataset/order. Checkpoints,
  orders, layers, solver resolutions and projection directions are correlated.
  They are not independent training replicates.
- **Design:** 12 ODE checkpoints with five evaluation orders (60), six unchanged
  baseline checkpoints (6), 18 native reproductions. Post hoc Ring8 robustness:
  12 model/order combinations at four resolutions (48).
- **Effects:** paired differences and percent changes relative to the same
  checkpoint's native/same-order evaluation. All 66 and 48 cells are retained.
  Dataset-specific results are reported instead of averaging different targets.
- **Precision controls:** operator actions in float64, gate/outer network in
  float32. Native/reference and same-order/action SW2 tolerance 5e-4.
  Maximum native/reference drift 4.47e-8. The solver follow-up reproduces all
  corresponding Heun32 scores and endpoint arrays exactly.
- **Local probes:** 16 native inputs per layer, six ODELayers per network.
  Means summarize these measured inputs/layers. Neither layer averages nor
  maxima over finite probes are global bounds or confidence intervals.
- **Intervals and inference:** no training-seed standard deviation, CI, p-value,
  normality test or population-level significance test is estimable from one
  seed. None is manufactured by treating directions or checkpoints as seeds.
  A bootstrap over fixed samples would quantify evaluation-sample variability,
  not training variability; it is not used here.
- **Multiplicity:** no inferential tests, hence no multiple-test correction.
  All comparisons are exploratory/descriptive; no selected-best-order claim.
  The predeclared <=1% relative-SW2 flag is not an equivalence margin validated
  statistically.
- **Solver interpretation:** Heun steps use two velocity evaluations each.
  Results at 32, 64, 128 and 256 steps are a finite-resolution check, not proof
  of convergence. Post hoc selection of Ring8 is documented in the protocol.
- **Data reuse:** test sets were repeatedly inspected across stages. This is
  exploratory mechanism analysis, not a pristine final benchmark.
- **CPU operators:** exact formulas plus numerical matrix exponentials and SVD.
  Five matched family members are constructed examples, not random samples;
  statistical inference is inappropriate.
- **Performance telemetry:** ten-second GPU observations are time samples
  including initialization and phase boundaries, not a controlled throughput
  study. The earlier 0.16/0.29-second concurrency benchmark is a scheduling
  heuristic; its +8% observed throughput is too brief for a robust speed claim.
- **Generalization:** no attention, two-dimensional data, direct velocity target.
  No inference about JiT image quality, ImageNet, or FID is supported.
