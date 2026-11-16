# Figure catalog

## hidden-map-refinement.png / .pdf
- Purpose: establish scale and refinement of the trained hidden-map difference.
- Data: complete independently verified six-model aggregates, all36layers and2880states.
- Caption:4/8/16/32midpoint substeps, two checked DOP853references; equal original checkpoint and entrance states.
- Observation: compare the map's actual approach to its continuous reference.
- Interpretation: establishes a computational difference, not its generative utility.

## normal-and-tangential-feature-error.png / .pdf
- Purpose: distinguish potential-normal from tangential feature error.
- Data: pooled squared norms at native4substeps from every state/layer.
- Caption: projection onto q at the reference endpoint;1/64is a geometric isotropic benchmark only.
- Observation: read both shares, retaining the direction component.
- Interpretation: motivates a direct downstream intervention rather than inferring generation from energy.

## hidden-energy-budget.png / .pdf
- Purpose: distinguish changes of path work from numerical balance defects.
- Data: RMSof complete signed per-state energy/work discrepancies.
- Caption: RMSbars are not additive; exact signed closure includes the reference's measured residual.
- Observation: inspect whether work and balance terms are large relative to the resulting gap.
- Interpretation: a defect norm alone does not explain the map or full network.
