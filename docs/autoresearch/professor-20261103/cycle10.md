# Cycle10 registration: radial growth of a direct input diagonal

Registered after the completed cycle8 constraint rejection and cycle9 feature
audit. Test only the functional change in [radial-diagonal.md](radial-diagonal.md).
No new P constraint, whitening, loss, optimizer or activation between solves.

## Twelve fixed20k cells

Each task has six arms: native raw K1, radial D, static d=h/sqrt(2), and native
SwiGLU64/128/192. Repeat all four existing native references with exact archived
state/sample checks; the two new arms retain raw P/common initialization and
the same vanilla Muon rate (.02 Ring8, .006 Product16). Ordinary rates remain
their earlier validation-selected values, not rates tuned on these new outcomes.

Only the D generation function changes. Both new arms have the raw model's
58,512parameters at Product16, with no learned controller or additional weights.
Radial D adds a reduction/square-root/scaling; record actual wall cost separately
from dense matrix MACs. All use seed0, batch512, native20k FM-pair training and
unchanged validation/test banks, Heun32 and logging. No strength/rate sweep.

Raw and Swi64 anchors launch first per host, two workers per GPU0. Then radial
and static arms, then the two wider ordinary references. No decision until all
twelve finish and replay. Complete native8->16 resume checks and exact parent8
checks precede launch; final native20k anchors must reproduce their archived
model/optimizer/buffer/RNG states and saved arrays.

## What would count as progress?

Use final validation oracle MSE for provisional parent decisions, preserving
the program's5% rule. Additionally report radial versus static and every
ordinary control: a static gain is not evidence for input-dependent radial
behavior. If radial does not improve both the native raw and static control
by at least5% on validation, do not select the radial mechanism for continuation.
A one-task lead remains task-specific. Test SW2 is separate and never selects
a rate, checkpoint or model; a worse sampling score is a tradeoff, not acceptance.

This20k screen is not asserted converged. A qualifying validation lead needs
its own matched80k confirmation against all relevant controls before an FFN
advantage is considered. Existing80k raw/Swi evidence stays visible. Do not
start image transfer or a hyperparameter rescue on a weak-parent improvement.

Before interpreting numerical regularity, verify the full normalized q
Jacobian, its bound, raw-direction/parity identity, same zero-anchor quadratic
coefficient and equality to the static control on unit RMS. Complete functions
and initial hidden states differ away from that surface; neither initialization
nor the function class is claimed globally identical.

- [x] Implement a D-only adapter/grid and pass CPU derivative/source witnesses.
- [ ] Publish/deploy, pass fresh documented native CUDA and actual-entry checks.
- [ ] Run baseline-first monitored queues, complete/replay all12, verify eight
      exact native20k anchors, and report all amplitude/ordinary controls.
- [ ] Continue or reject from the complete registered comparison; no automatic
      extra trick, temperature, rate, K2 or image rescue.
