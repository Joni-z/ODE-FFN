# Report2 descriptive analysis

Question: does the existing free-ISRU result survive stronger controls and
transfer to the current image model? Compare fixed outcomes within their
registered benchmark, never across different task dimensions or FID banks.

The synthetic Product16 field-fit lead is real in its recorded budget and
does not imply a uniform task or generation win. The new image run is still
live; the latest completed generation evaluation is25k. Its poor EMA samples
coexist with much better raw-weight validation and more textured raw images.
Both effects must be retained before an architecture verdict.

| Updates | ISRU EMA FID | SwiGLU matched | SwiGLU wide |
|---|---:|---:|---:|
|5,000|445.9173|462.8621|452.0683|
|10,000|450.5974|508.9656|375.2641|
|25,000|252.9649|262.9314|268.7425|

Every row uses1024generated/5000reference images. Final50k evaluation uses
a different generated count and is not available; do not compare a future
5000-generated score directly with these as a pure training improvement.

## Claim candidates

- Claim: ISRU improves Product16 field fitting at the recorded5k budget.
  - Evidence: `../data/synthetic.csv` and the original comparison CSV.
  - Allowed wording:52.5% lower field MSE than Swi160 with fewer parameters.
  - Forbidden stronger wording: universal improvement, converged advantage,
    ODE-specific necessity, or equally strong generation improvement.
  - Next check: shared longer-budget Product16/dependent-sign comparison.
  - Decision: keep with task/budget scope.
- Claim: ISRU's25k image EMA FID is lower than both controls.
  - Evidence: all three25k rows, with5k/10k rows retained and independently
    recomputed from feature files.
  - Allowed wording: a descriptive endpoint observation on a poor interim model.
  - Forbidden stronger wording: effective image method or architecture winner.
  - Next check: finish50k and assess ordinary-baseline quality on the same counts.
  - Decision: weaken to an interim observation.
- Claim: parameter EMA lags the raw predictor in this run.
  - Evidence: fixed validation curves, fixed source0–3 raw/EMA grids, decay.
  - Allowed wording: a substantial observed early loss gap that narrows by40k.
  - Forbidden stronger wording: EMA explains all FID or raw FID is already known.
  - Next check: separately tagged raw/EMA quality evaluation, with identical
    sampling and counts, preserving registered results.
  - Decision: keep as an evaluation diagnosis and prospective hypothesis.
- Claim: matrix conditioning alone explains image quality.
  - Evidence: W1/W2 histories show moderate norm growth and high effective rank,
    without a spectral intervention or full-model causal decomposition.
  - Allowed wording: learned anisotropy, no demonstrated penalty benefit.
  - Forbidden stronger wording: spectral instability caused the poor FID or
    the nonlinear subflow guarantees whole-model stability.
  - Decision: discard the causal claim.

The selected conditional-frame case provides another positive synthetic
direction result while preserving its K1 and thick-transfer limitations.
Other post-Report1 work remains indexed and is not erased by this selection.
