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
- [ ] Implement the feature-clock and time-clock arms with identical initial
  parameter states and only a changed clock input source.
- [ ] Verify active full derivatives, native replays, exact resume, conditional
  matrix equivalence and measured parameter/forward/backward costs.
- [ ] Pass fresh native/metric documented invocations and launch the two
  monitored one-thread CPU trajectories on Mac mini.
- [ ] Complete prediction, matrix/clock, sample and feature/FID audits; publish
  all outcomes and stop or retain the setting under the original rule.

## Status
Protocol and mathematical scope are fixed before new training outcomes.
Implementation and execution have not started. No new model gain is claimed.
The stopped matched-Swi initializer and frozen calendar remain stopped.
