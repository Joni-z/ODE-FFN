# Anchored gate geometry diagnostic

Completed on 145 (`hw3242-gpu1`) with one CPU thread in 8.76 seconds. Existing checkpoints only; no training and no GPU allocation.

- [Findings](findings.md)
- [Protocol fixed before execution](protocol.md)
- [Collection audit](audit.json)
- [Panel metrics](geometry-panels.csv) and [stratified metrics](geometry-strata.csv)
- [Full summary](geometry-summary.json), [case studies](tail-cases.json), [provenance](provenance.json)
- [Launch receipt](launch-145.json), [execution notes](notes.md)

Runtime and raw arrays are preserved at `/data2/zz5070/ODE_FFN/gate_geometry_20261021/` on 145. The source that executed is commit `fe0bd5d`. The diagnostic writes `status.json`, `monitor.json` and `telemetry.jsonl` in its output directory.

Reproduction scripts: [check_gate_geometry.py](../../scripts/check_gate_geometry.py) and [analyze_gate_geometry.py](../../scripts/analyze_gate_geometry.py).
