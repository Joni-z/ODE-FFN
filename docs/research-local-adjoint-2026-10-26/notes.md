# Prototype notes

The first four-test run found a one-dimensional input shape problem: the reused batched adjoint helper introduced a leading singleton axis for a bare feature vector. The wrapper now promotes a single vector to a batch and restores its shape; batched computation and old helper sources are unchanged. Three other derivative/full-model checks passed in that first run. No quality experiment or profiling had started.

The candidate's local gradient identity is a known pullback construction. It does not inherit the full-model potential guarantee or the previously observed whole-field training gain. Generic tied-ridge controls and ordinary SwiGLU remain essential. The current task is correctness/cost validation before choosing any cold experiment.

The fixed resource screen and all actual pipeline checks passed before a separate seven-arm Phase-A protocol was executed. Formal5000-update cells all completed, and the complete independent replay passes. Candidate error.190290 improves original.325527 and omitted.343629, but loses to near-parameter ordinary tied128.179585, tied192.135191 and both SwiGLUs. All quality/parameter and derivative/sampling continuation conditions fail. No new task or tuned variant is launched.

Both report figures were inspected. The endpoint's log-axis tick labels crowded SW2; only the Jacobian/SW2 axis formatter was changed to ordinary numeric ticks and figures regenerated. The original model/field/analysis data were unchanged.
