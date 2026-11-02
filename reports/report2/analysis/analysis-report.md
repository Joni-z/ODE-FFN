# ODEFFN retrospective analysis

The question is whether a specific internal ODE constraint improves the learned
field and generation at an adequate ordinary-model budget. This synthesis
replaces the earlier image/ISRU-centered analysis. Source paths, protocols and
all smaller attempts are retained in [the ledger](../attempts.md).

## Findings that change the argument

| Observation | What it supports | What it does not support |
|---|---|---|
| Spectral caps reduce sensitivity more consistently than they improve SW2; input-gated P can have a smaller norm but a larger full Jacobian. | Measure gate derivatives and the complete map, alongside P. | Flattening P's singular values is a justified next treatment. |
| Exact-flow replacement of frozen K2 worsens Ring8; static free maps fit better than a tied polynomial. | Solver accuracy and learned function quality differ; parameterization matters. | Cold exponential training must fail, or the optimizer alone causes the gap. |
| Complete-gradient ODE improves Product16 MSE .325527→.037320. Gradient Swi192 reaches .033826 at near-equal dense work. | A full transposed derivative can be useful; the ODE line has conditional positive observations. | A general original-ODE advantage, or a free improvement at unchanged compute. |
| Conditional-frame K2 improves the thin target, but K1 is similar and thick sampling loses. | Input-dependent directions can help this geometry. | A second-order benefit or broad transfer. |
| Tied cubic improves Product16; free ISRU does better but reverses on dependent signs/mixtures. | A particular response and initialization are useful controls. | The winning free model establishes the value of generator tying. |
| Analytic labels, affine corrections and frozen head refits do not close the direct ODE gap; local teacher fits also show a gap. | Several simple explanations/repairs are insufficient at the tested budget. | A representational lower bound or identification of the sole mechanism. |
| The ideal conditional mean is not trajectory-invariant. | Designs must state which physical or hidden dynamics they represent. | An arbitrary hidden ODE follows automatically from the sampling ODE. |
| Image baselines remain inadequate. | A competent baseline is needed before transfer claims. | Any image ranking is evidence for or against ODEFFN. |

## Decisions

Retain complete-gradient readout and input-frame observations as conditional
positive evidence in the original-model investigation. Retain free ISRU as a
control, with its independent matrices and task boundaries explicit. Do not
present exact integration, proof verification or deployment speed as learning
success. Do not select a new constraint on P from anisotropy alone.

The report proposes two discriminating questions: whether thin-data direction
adaptation survives adequate controls/training, and whether tying earns anything
beyond initialization and optimizer coordinates. Neither is launched here.
Earlier failed settings keep their original status. The next experiment is a
researcher decision, not an automatic sweep of further tricks.

This pass reuses preserved evidence. It verifies the selected source-table
bindings and presentation, without new training or a fresh replay of every
historical result. The November1 audit separately binds the synthetic model
identity and selected checkpoints.
