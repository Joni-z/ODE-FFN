# Figure catalog

- product16-heldout (PNG/PDF): final test field, derivative, latent marginal distance and well-membership diagnostics. Purpose: test whether a vector-clock field gain over the scalar parent reaches the practical SwiGLU reference. Observe that it does not. Bars are descriptive values with zero baselines and no inferential error bars.
- product16-learning (PNG/PDF): all recorded validation oracle errors. Purpose: expose the whole learning comparison instead of a selected checkpoint. The baseline lead persists at the final endpoint. Trajectory points are not independent replicates.
- clock-rank-floor (PNG/PDF): scalar actual derivative error and the conditional rank-six best-correction floor, plus their ratio at every time. Purpose: prevent claiming that the scalar rank count alone explains its entire residual. This is conditional on the trained anchored base, and the unrestricted rank-six relaxation need not be attainable by the network.

Data sources are the recorded product16 heldout summary, rank summary, configs and complete metrics.jsonl files, copied under the stage identifier. Log scales in the learning/error panels are labeled. All four baselines remain visible.


## Posterior-region and empirical-target follow-up figures

- figures/posterior-region-errors.pdf: splits the exposed20000-step field errors into expanding diagonal, contracting diagonal and off-diagonal contributions. Contributions exactly sum to the full squared Frobenius error. It prevents attributing every failure to rare expanding boundaries; contracting errors also matter. Data: posterior-regions.json; deterministic cases and interpretation: ../posterior-regions-readout.md.
- figures/empirical-target-difference.pdf: exact100000-atom empirical teacher versus population teacher on128 fixed validation anchors per time. Shows field/Jacobian differences and median posterior ESS, with logarithmic scales and no inferential error bars. The late mismatch requires matched-law controls before a noise-isolation experiment; it does not explain model differences by itself. Data: empirical-target-summary.json, independently checked by empirical-target-autograd-check.json.
