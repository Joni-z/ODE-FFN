# Units and interpretation

Field inputs are four independent banks of 64 points at each of five
prescribed times, not a uniform continuous-time sample. Global MSE and source
RMS aggregate all 1280 points. The scatter's faint markers display each bank;
[comparison.json](comparison.json) and [field-scores.csv](field-scores.csv)
provide bank means and SDs. A pooled RMS need not equal the mean bank RMS.
The models retain their original training settings, including Ring8
SwiGLU64/Muon0.02 and Product16 SwiGLU192/AdamW0.0003. Shared versus stage
uses the matched within-family configurations.

The eight trajectory sources are a deterministic numerical qualification
subset of the newly bound Gaussian source arrays. All model/field cells use
the same sources within task. There are no repeated-training observations,
and no population KL or significance claim is made from this numerical pilot.
The thresholds test numerical accuracy; they are not effect-size margins.

All 48 initial and ten added trajectory rows pass independent artifact checks.
All 16 final cells meet their registered numerical gates. Six stop at 512,
ten at 1024; do not compare performance at their differing pilot resolutions.
The formal comparison will use common 1024/2048 resolutions and retain each
source, including difficult points. A failed per-point numerical gate cannot
be repaired by excluding a source or loosening the tolerance.

The exact continuous-flow log-density ratio has expected value
KL(rho_1||p_1). A finite Monte Carlo average can be negative, and the actual
Heun map's density is not automatically the continuous-flow density. The
observer records both continuous integrals and discrete local log determinants
to expose this distinction. Positive determinants on observed steps do not
prove global invertibility of a coarse discrete map.

Pointwise density-source norms and field MSE are descriptive intermediate
measurements. Different errors can cancel along a model path, and the path
itself changes between models. No correlation test or causal mediation
fraction is inferred from the scatter. The planned formal paired density
contrasts will be assessed on the larger independent Gaussian source cohort.
