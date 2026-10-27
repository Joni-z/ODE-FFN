# Statistical and numerical scope

- Unit: a fixed saved model/checkpoint and its next native512-row batch. The
  22models share paired data streams; their four checkpoints are repeated
  observations. There are88probes, not88independent replicates.
- Model initialization remains seed0. No new training or label resampling.
- All11unique clean families from cycles07/11/14are retained; no best-model,
  best-checkpoint or best-batch selection. Final batch5001was not trained on.
- Descriptive ranges are minima/maxima over the registered finite cohort.
  They are not confidence intervals or estimates of training-batch variance.
  No p-values, standard errors, significance or multiple-testing claims are
  made because this is a deterministic identity/local route audit.
- Loss fractions divide each projected squared loss by that batch's complete
  paired loss. All use ambient128normalization, so the three energies add.
  They are not per-normal-dimension field MSE, SW2 or path-error shares.
- Gradient vectors add; gradient energies require all three cross terms.
  Their separate squared norms are not causal gradient fractions. Parameter
  coordinates differ between C/B/frame/Swi, so absolute norms cannot rank
  architecture capacity.
- Adam effect size: ||d_full-d_removed||/||d_full|| within a parameter group;
  cosine describes direction. Both use the same stored moments and promoted
  float64 arithmetic. No fitted step or altered historical moment sequence.
- Empty P groups and exactly zero hidden cold gradients have undefined ratios.
  Near-zero cold readout directions are numerical artifacts and are not
  interpreted, even if a nonzero roundoff denominator produces a numeric cosine.
- The native field replay is bitwise on1.76Mrows. Independent gradient gates
  are atol1e-10/rtol1e-8; projector atol1e-9. These gates did not change after
  outcomes. Max gradient/projector differences1.26e-15/9.06e-12.
- Historical training-window clipping fractions and this single next-batch
  probe answer different questions. Final no-clipping does not show that
  late training generally avoided clipping.
