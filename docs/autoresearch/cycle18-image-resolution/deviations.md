# Execution deviation

All four sampler children finished with exit 0. The first independent audit
then exited 1 before creating its output directory or doing numerical work:
`scipy.linalg` was imported at module scope, although sampling replay needs
only the lean training environment and SciPy is declared only in the metrics
environment. The original controller consequently stopped; it did not retry.

The original frozen source, terminal failure, launch records and log remain
unchanged. `audit_v2.py` moves only the SciPy import into `independent_fid`.
It and the byte-identical study module have separate runtime hashes. All
existing sampling files are separately bound before recovery.

`recover_audit.py` explicitly runs the corrected sampling audit, original
metric producer and corrected feature audit once, with their registered
ceilings. It does not launch training or sampling. No package, native model,
source bank, step count, metric formula or numerical tolerance changes.
