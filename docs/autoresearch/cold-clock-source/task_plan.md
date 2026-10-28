# Cold clock-source comparison

## Goal
Test whether making the local clock depend only on objective time yields a
useful conditional mixing restriction beyond the completed fixed-clock model.
Require improvement against a matched-initial-function feature-clock control
and all stronger retained ordinary/ODE controls. Do not treat the existing
AdaLN time conditioning as absent.

## Actions
- [x] Close the real-state P/W coordinate check and select a different
  functional question rather than another fixed-core interpretation.
- [x] Check the old time-profile stop rule: it closes that frozen calendar,
  not every cold-trained time-clock model. Preserve it without new calibration.
- [x] Verify that the actual JiT FFN receives c(t)=t_embed(t)+y_embed(0),
  independent of image features. Register exactly two cold arms and the gates.
- [x] Implement the feature-clock and time-clock arms with identical initial
  parameter states and only a changed clock input source.
- [x] Verify active full derivatives, native replays, exact resume, conditional
  matrix equivalence and measured parameter/forward/backward costs.
- [x] Pass fresh native/metric documented invocations and launch the two
  monitored one-thread CPU trajectories on Mac mini.
- [ ] Complete prediction, matrix/clock, sample and feature/FID audits; publish
  all outcomes and stop or retain the setting under the original rule.

## Status
Protocol and mathematical scope are fixed before new training outcomes.
Both actual documented checks pass, native33463/b15aae and metric62554/73fffe,
exit0 once with no discrepancies. Controller96978 is live with owned
feature/time training PIDs84850/84851. See progress.json for a snapshot.
Both arms use model seed0 and original data stream. No new FID or model gain
is available. The strict17-control/endpoints comparison remains pending.
The stopped matched-Swi initializer and frozen calendar remain stopped.
