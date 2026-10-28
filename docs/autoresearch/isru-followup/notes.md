# ISRU evidence and continuity notes

## Latest user direction — October 29, 2026
The user corrected the claim that synthetic work had no positive design result,
asked why ISRU was not being pursued, and required maintaining Markdown so
research progress and decisions survive context loss. ISRU evidence is now
the immediate focus. Baseline adequacy remains a prerequisite for claims.

## Completed history
1. Cubic local-flow screen: the tied/adaptive cubic split improves the old
   ODE but loses to strong ordinary controls. The independent `free_isru`
   control reaches Product16 oracle MSE 0.037159 versus Swi160 0.078178.
2. Cold dependent-sign transfer: the same ISRU model reaches 0.085073 versus
   Swi160 0.039790. This narrows its task scope, not the existence of the
   Product16 improvement.
3. Half-mixture control: ISRU 0.163557 versus Swi160 0.113254, while all
   65,536 component locations retain positive probability. Literal component
   count alone is not a sufficient explanation.
4. Subsequent research moved to other operator families. It did not settle
   the positive ISRU mechanism or provide an adequate convergence comparison.

## Exact implementation and paths
- [FreeISRU and cubic_flow](../../research-cubic-split-2026-10-26/cubic_core.py).
- [Original protocol](../../research-cubic-split-2026-10-26/protocol.md),
  [comparison CSV](../../research-cubic-split-2026-10-26/comparison.csv),
  [training CSV](../../research-cubic-split-2026-10-26/training-metrics.csv).
- [Transfer protocol](../../research-joint-dependence-2026-10-26/protocol.md),
  [comparison CSV](../../research-joint-dependence-2026-10-26/comparison.csv),
  [training CSV](../../research-joint-dependence-2026-10-26/training-metrics.csv).
- [Half-mixture comparison](../cycle02-full-support/results/comparison.csv).
- The study imports its historical native source from
  `/Users/mr.z/ode-workspace-prep/architecture-search/conservative_readout_20261026/source-v2`.
  The exact ISRU runtime/checkpoint path is a pending recovery task; do not
  invent it or launch a replacement run to avoid locating the old one.
- Working checkout: `/Users/mr.z/ODE-FFN-autoresearch`, branch
  `autoresearch/oct27-ode`. Canonical remote: `Joni-z/ODE-FFN`, main.

## Recorded training facts and unknowns
The initial/transfer experiments use 5k updates, batch 512, uniform-time paired
velocity supervision, AdamW3e-4 and clip 1. ISRU's last Product16 clipping
fraction is 0.980, while Swi160's is 0.000. Both oracle errors fall substantially
between 2.5k and 5k. These facts do not establish convergence or identify a
clipping cause. The fixed budgets and old gate decisions are historical facts,
not proof that the whole ISRU direction is invalid.

## Interpretation to preserve
There is a real positive, task-specific ISRU result. Do not replace it with
the original professor ODE's worse scores, or describe all synthetic work as
unsuccessful. Also do not inflate that result into a general generation or
ODE-specific success. Current priorities are training adequacy, fair controls
and understanding the positive case before adding a new component.

No new training ran during this Markdown update. All old reports, raw runs,
sealed results and submitted Report1 remain unchanged.
