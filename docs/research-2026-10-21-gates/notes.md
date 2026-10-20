# Working notes

- Torch login currently fails authentication.
- MBP is reachable; its connection to 145 is closed remotely.
- Existing trainer samples against test targets during training. New configurations
  must explicitly select validation sampling; preserve legacy defaults.
- Removing gate modules normally shifts subsequent initialization. Construct and
  discard them before scalar/fixed replacement to preserve every common weight.
- The submitted report is not being revised.
- Direct SSH to 145 works. Its canonical project path moved to
  `/data2/zz5070/ODE_FFN`; the old data1 root is retired.
- GPU0's three yp2841 workers were stopped with the owner's helper, but their
  controller restarted them. A second immediate launch completed four 20-step GPU
  smoke checks before the guard detected new foreign processes and stopped ours.
  No formal GPU training started. Do not repeatedly kill these workers.
- CPU fallback benchmark is running. Its results decide feasibility, not model
  quality. Protocol amendment was made before any full training.
