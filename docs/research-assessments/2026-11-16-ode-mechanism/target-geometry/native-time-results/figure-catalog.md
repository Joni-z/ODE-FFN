# Figure catalog

## native-grid-convergence.png / .pdf
- Purpose: show why nearby/dyadic checks do not substitute for fine-grid qualification.
- Data: summary.json convergence_to_finest, recomputed from hash-bound native arrays.
- Caption: both tasks, all six models,256sources/model, categorical step axis; finest endpoint is an internal reference and its constructed zero is omitted; no error bars.
- Observation: fine-grid residuals shrink; two cases need8192under the component gate.
- Interpretation: native endpoints are ready for the refined sensitivity test.
- Check: do not interpret this as SW2, an analytic convergence proof or hidden-solver refinement.

## near-equal-work-grid-sensitivity.png / .pdf
- Purpose: isolate outer-time grid sensitivity at fixed checkpoints/inputs.
- Data: the1000/1024/1032contrasts in summary.json, all8banks retained.
- Caption: small dots are bank RMSs; diamonds pool squared error over all components/sources; no independent training repeats.
- Observation: nearby counts can differ appreciably; the finest reference is absent here.
- Interpretation: near-equal work does not establish equal continuous trajectories.
- Check: no accuracy ranking or distribution-distance claim follows from this figure.
