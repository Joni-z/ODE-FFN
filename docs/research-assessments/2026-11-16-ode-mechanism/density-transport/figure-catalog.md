# Figures

- [field-source.png](figures/field-source.png): compare componentwise oracle
  velocity MSE with instantaneous density-source RMS. Filled circles are
  original fields, open squares projected fields; thin lines connect the same
  model. Faint markers are the four field banks. Purpose: check the local-norm
  explanation. Observation: neither local norm gives the known Product16
  sampling ordering. Consequence: measure signed path accumulation rather
  than promoting the local norm to a generative metric.
- [numerical-gates.png](figures/numerical-gates.png): all 16 cells before/after
  targeted refinement. Annotations divide each error by its unchanged limit;
  <=1 passes. Colors show log10(error/limit), with <=0 passing. Row brackets
  give final step counts. Purpose: distinguish formula implementation from
  resolution qualification. Observation: ten 512-step cases fail refinement
  despite small same-resolution chain errors, then pass at 1024. Consequence:
  the formal experiment uses common finer steps and repeats the gates.

PNG and PDF versions are provided. Both PNG figures were visually inspected
for labels, legends, clipping, marker meaning and the distinction between
annotation ratios and color values. These are diagnostic and numerical plots,
not population-density rankings.
