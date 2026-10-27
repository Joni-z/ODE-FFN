# Audit correction; no formal experiment changes

The first independent audit stopped on a conditional-generator spectrum at the first cold snapshot. The formal diagnostic evaluates the two saved visit anchors as a two-row projection, whereas the first auditor projected all32 anchors and then sliced the first two. These mathematically identical float32 operations use different matrix kernels and differed by up to2.68e-7 in the resulting spectra.

Audit-v2 reproduces the diagnostic's two-row projection before independently constructing K and computing its NumPy singular values. It retains the original spectrum tolerances (1e-8 relative,1e-9 absolute). Full32-anchor projections remain in the separate activation-response analysis. The failed auditor source/log remain at private `audit/failed_audit_source.py` and `audit.log`. No model, checkpoint, metric, formal output, source snapshot or scientific selection rule was changed or rerun.
