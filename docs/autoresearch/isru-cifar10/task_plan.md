# ISRU CIFAR-10 transfer — Report2

## User authorization
On October 29, 2026 the user explicitly requested taking the existing positive
ISRU design to CIFAR-10 and continuing the research. This selects a new,
separately registered transfer study. It does not establish that ISRU is the
best model on every synthetic task, and does not reopen Report1.

## Goal
Run the existing free-matrix ISRU response in unconditional image JiT with
ordinary SwiGLU controls and a credible shared training setup. Establish
baseline competence before interpreting relative quality. No new ODE trick.

## Actions
- [x] Read the existing ISRU implementation, records and user corrections.
- [x] Check allowed machines: Mac mini has MPS and16GiB system memory;
  Lenovo SSH currently times out. Do not wait for MBP or torch.
- [x] Check the primary JiT training/sampling recipe and warm environment.
- [x] Measure a larger shared backbone's real MPS memory/throughput and
  select a feasible batch before freezing the formal protocol.
- [x] Specify the unchanged ISRU operator, near-budget SwiGLU and stronger
  ordinary control, common initialization scope, training/EMA/budget,
  sampling/evaluation and interpretation limits.
- [x] Implement the entrypoint and monitoring; keep old sources/data sealed.
- [x] Commit and publish source; freeze an exclusive runtime.
- [x] Pass seeded kernel/gradient/resume/metric checks and a fresh documented
  invocation before formal launch.
- [x] Launch on an available authorized machine, verify actual progress and
  monitor throughput, memory and GPU use where observable.
- [ ] Keep Report2 evidence and progress updated; evaluate only at specified
  checkpoints, preserving both candidate and ordinary-control outcomes.

## Decisions so far
The old three-layer,64-wide,20k CPU image screen does not meet the baseline
requirement. The new setup must address capacity and training adequacy rather
than simply add ISRU to that screen. Hardware calibration is not a quality
search; record every tried shape and timing. Final settings will be registered
before quality training. No new training has started at this plan's creation.

The user subsequently selected Mac mini only. Batch512/SDPA/foreach Adam
hardware calibration passes; about6.17s per three-arm update and5.6GB peak
driver memory. Shared50k cap means roughly86hours plus evaluation. No
convergence or baseline-competence claim is implied by that budget.
The author real-kernel/gradient/physical-resume witness passes; fresh frozen
documented checks are next. No packages installed and no historical source
or Report1 edited.

## Running
Registration6bffa70 was published and frozen before formal training.
Fresh native77532/ccea00 and metric68393/b96d97 both exit0 once; hashes bind
in `documented-pass.json`. Launch77701/d8ab5d exits0, controller95010 and
trainer95011 are detached and live. All three arms reached25 updates with
finite losses/gradients. Monitor and `caffeinate` are active. Separate
Apple device telemetry95058 records97–99% utilization; the frozen training
source was not changed. See [status.md](status.md) and the timestamped
[launch snapshot](launch-status.json). Next:500-update validation/checkpoint,
then2k fixed-start grids; image effectiveness remains unresolved.
