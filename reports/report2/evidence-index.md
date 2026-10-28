# Report2 evidence index

All research after the finalized Report1 revision is in scope. The entries
below organize the main evidence; the complete Git-derived area inventory
is retained in [scope.json](scope.json). Inclusion here does not mean every
experiment belongs in the three-page main report or meets an effectiveness
criterion.

| Evidence area | Principal sources | Report2 use and current limit |
|---|---|---|
| Positive ISRU design | [Initial study](../../docs/research-cubic-split-2026-10-26/analysis-report.md), [current follow-up](../../docs/autoresearch/isru-followup/README.md) | Product16 field MSE 0.037159 versus Swi160 0.078178 is a positive finite-budget result. The winning free-matrix response, tied cubic candidate and baseline must remain distinct. Training adequacy and attribution are unresolved. |
| ISRU task boundaries | [Dependent-sign study](../../docs/research-joint-dependence-2026-10-26/analysis-report.md), [half-mixture control](../../docs/autoresearch/cycle02-full-support/analysis.md) | Retain both reversals: ISRU/Swi160 MSE 0.085073/0.039790 and 0.163557/0.113254. They constrain generalization and rule out literal component count alone as a sufficient explanation; they do not erase the Product16 positive result. |
| Input-dependent P directions | [Frame study](../../docs/autoresearch/cycle11-conditional-frame/analysis.md) | Thin-data field MSE 0.000685 versus Swi192 0.003095, with visible eight-cluster recovery. Thick-data transfer and a second-order-specific benefit are not established. This is another positive design observation with explicit boundaries. |
| Other synthetic designs and controls | [Architecture-search review](../../docs/research-architecture-search/next-direction-evidence.md), [autoresearch index](../../docs/autoresearch/README.md) | Preserve fixed-budget outcomes and their controls. Do not treat implementation/audit success as model effectiveness, or failed fixed screens as universal impossibility results. |
| Mathematical structure and prediction objective | [Cubic-flow derivation](../../docs/research-cubic-split-2026-10-26/prior-and-theory.md), [denoiser dynamics](../../docs/autoresearch/cycle22-denoiser-dynamics/analysis-report.md), [real-state optimizer coordinates](../../docs/autoresearch/p-coordinate-steps/analysis.md) | Select only mathematics needed to explain the method and its limits. Local flow identities, optimizer-coordinate equivalence and conditional-mean dynamics do not by themselves prove a generation benefit. |
| Image exploration and its validity limit | [Image-budget study](../../docs/autoresearch/image-budget-continuation/analysis-report.md), [clock comparison](../../docs/autoresearch/cold-clock-source/results/analysis-report.md), [validity correction](../../docs/autoresearch/baseline-validity-review.md) | Report as limited small-model exploration. The weak, unvalidated baseline prevents architecture-effectiveness claims from FID172–200 rankings. Keep the correction visible. |
| Ordinary controls and initialization | [Cold clock/SiLU controls](../../docs/autoresearch/image-clock-training/analysis-report.md), [matched SwiGLU initialization](../../docs/autoresearch/image-swiglu-initialization/results/analysis-report.md) | These expose parameterization and initialization confounds in that screen. They do not replace validation of a competent image baseline. |
| Current and next-two-week work | [ISRU plan](../../docs/autoresearch/isru-followup/task_plan.md), [Report2 plan](task_plan.md) | Existing-state recovery, training adequacy and a focused follow-up remain work in progress. State hypotheses prospectively and update this row only after actual evidence exists. |

## Working narrative

Start from the project question, then explain the useful synthetic design
signals and what the controls changed in their interpretation. The current
ISRU lead is a candidate main thread, with the validity/convergence gap stated
directly. Include other completed work only when it changes the argument or
explains a decision. End with specific hypotheses for the next two weeks.
This is a provisional organization, not a frozen Report2 conclusion.
