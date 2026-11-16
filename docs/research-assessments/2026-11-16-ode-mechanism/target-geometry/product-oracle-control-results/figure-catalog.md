# Figure catalog

## exact-oracle-and-sensitivity.png / .pdf
- Purpose: independently validate the oracle and expose target sensitivity variation.
- Data: quantile-summary.json, hash-bound exact-oracle.npz outside Git, all256sources.
- Caption: left shows absolute native-path errors versus the exact quantile flow; right shows descriptive median/99th/max over repeated source-coordinate pairs.
- Observation: second-order endpoint convergence and a large sensitivity tail coexist.
- Interpretation: oracle integration is checked independently; a uniformly useful finite linearization cannot be assumed.
- Limit: target sensitivity is not a demonstrated cause of learned-model error.

## correct-law-failed-linear-prediction.png / .pdf
- Purpose: falsify the implication from linear endpoint prediction failure to incorrect generation.
- Data: independent-summary.json, exact actual-law marginal-preserving control.
- Caption: four registered multipliers,256sources; population marginals stay exact by the Gaussian rotation pushforward; these are paired endpoint energies, not SW2orFID.
- Observation: every tested finite-scale fraction is negative despite correct marginals.
- Interpretation: distribution relevance must be tested separately from one reference coupling.
- Limit: the control does not prove trained errors are density preserving and is not a candidate architecture.
