# ISRU CIFAR-10 — running

October29,2026 launch snapshot. **Formal training is running on Mac mini MPS**,
from the cold seed0 start. No image-quality conclusion is available yet.
The authoritative current process state is in the private runtime logs;
[launch-status.json](launch-status.json) is a timestamped publication snapshot,
not a live dashboard.

- Frozen training/source registration: `6bffa70ef99649ec4eef421763dea1519bb19a93`.
- Runtime: `/Users/mr.z/ode-workspace-prep/autoresearch/oct29-isru-cifar10-v1`.
- Controller PID95010; trainer PID95011; Apple telemetry PID95058.
- All three arms reached25 updates; the loss/gradient records are finite.
- Real throughput is about6.3s for one update of each arm, with batch512
  and about6.8GB MPS driver allocation. Device-wide Apple GPU samples are
  97–99%; utilization is not a per-process attribution.
- Existing supervisor records process/memory/throughput every30s. An added
  read-only [Apple GPU monitor](mac_gpu_monitor.py) records `ioreg` counters
  every30s in `output/apple-gpu-monitor*.json*`. Its frozen source is in
  `telemetry-source/`, independently hashed. The live training source,
  source bindings, recipe and process were not edited or restarted.
- Ordinary controls and ISRU use the same data stream. The first2k sample
  grids are expected after roughly3–4hours, first5k FID after about9hours.
  Full50k budget is roughly four days including evaluation, subject to
  actual machine throughput.

[Readiness evidence](readiness.json) retains the fresh agent's exact command
and actual terminal exits. Both checks passed once with exit0. The naming
alias “image-metrics” versus ledger “image-metrics-v2” refers to the same
dc729a50 spec and v2 interpreter; no rebuild or invocation correction was
needed. [Hardware calibration](hardware-calibration.json) includes both
tried execution settings and contains no quality results.

Next inspect the first validation curve at500 updates and fixed raw/EMA
grids at2k. Retain both ordinary-control outcomes and all W1/W2 diagnostics.
Do not treat this successful launch as convergence, baseline competence,
or an ISRU image benefit. All work belongs to Report2; Report1 stays frozen.
