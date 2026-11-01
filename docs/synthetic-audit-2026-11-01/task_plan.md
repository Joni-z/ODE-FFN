# Synthetic experiment audit — 1 November 2026

## Goal
Determine which models were actually trained, whether the FFN controls are fair, and which architecture claims the existing synthetic results support. No new training or checkpoint mutation.

## Plan
- [x] Inventory historical source, frozen runs, configurations and result files.
- [x] Obtain a fresh read-only integrity review from primary artifacts.
- [x] Recompute reported comparisons and verify source/model/evaluation bindings.
- [x] Write and verify a scoped verdict and concrete prerequisites for any next experiment.

## Constraints
Preserve Report1, Report2 LaTeX/PDF, active training code and all runs. Keep the Lenovo baseline paused. Use seed0 only if any future experiment is proposed. Publish verified audit documentation to GitHub.

## Status
Fresh review completed with WARN. Numerical replay passes; all local links resolve and protected source/report files are unchanged. Audit ready for GitHub publication; no experiment has been launched.

## Errors
- Initial inventory searched removed active synthetic/configs/scripts directories; source is preserved in the research archive and prior checkout.
- Optional .obsidian directory was absent; no project binding was found at that path.

- First replay reached all 20 exact endpoints, then failed in the new audit script because the half-mixture CSV also includes reused reference laws. Restricted that selector to law=half_mixture; the corrected complete replay passes. Historical experiment files were unaffected.

## Decision
Preserve the Product16 fixed-budget finding. General ODE, activation-only, convergence and image-transfer claims are not established. The next experimental decision should address budget adequacy and initialization/activation attribution, not another unregistered design.
