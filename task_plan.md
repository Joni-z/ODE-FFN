# ODE mechanism investigation

Goal remains active: determine what continuous ODE structure contributes to
flow-matching models, using mathematical predictions and falsifiable training
controls. Seed0 only. Scores use three decimals. Preserve final Reports1/2 and
native image code. Mac mini freely available;145 physical 0-3 prohibited,
4-7 only when<500 MiB and no compute PID. Never reclaim/share a job.
Publish verified code and evidence promptly to GitHub main as Zhizhe Zhang.

## Completed evidence

- [x] Fixed/free Taylor coefficients, state feedback, shared/released operators,
  coordinate and matched stage-order controls. No general cross-task advantage.
- [x] Full refined transport calibration and exact actual-law counterexample:
  source-coupled endpoint error does not establish distribution quality.
- [x] Hidden energy/readout/reference studies on all 2,880 states. All-layer256
  references fail late Ring8 layers; this failure remains recorded.
- [x] Complete equal-dimension null: all six tangent-minus-null adjusted
  intervals negative. Potential-normal specificity resolves only on Ring8.
- [x] Uniform512/1024 first-layer sampling extension: all 12comparisons qualify;
  changes to target SW2 are small and not consistently beneficial. Preserve the
  original256 pilot failures. Do not turn reproduction G into a quality score.
- [x] Parameter-free midpoint bias prediction: all 2,880 states qualify;
  trained 4-step total prediction B=.831--.971, both total/normal adjusted
  intervals positive in6/6. Fine64->128->256 residuals decrease in6/6.

## P0: causal test of solver faithfulness

1. [x] Register rk4-control-protocol.md/execution.md. Same f=PISRU(y), width 64,
   depth 3, interval [0,1], tensors/RNG, seed 0 and 20k updates. Change midpoint4 to
   classical RK4 two steps; both use8 P applications. Official numerical step copied.
2. [x] Author checks: independent algebra/gradients/order/Pcalls, native
  8->16resume, real evaluator entries and exact original-midpoint step0 tensors,
   buffers/data_rng. No quality conclusion from short checks.
3. [x] First frozen documented pass exits 0, all native checks pass. It flags
   gradient-check wording broader than the actual directional test. Preserve the
   receipt, correct only wording, and repeat from a new frozen doc/output path.
   Corrected frozen9452a4f invocation exits 0 with no divergences.
4. [x] Launch two one-thread Mac CPU 20k training trajectories with a persistent
   monitor; verify live command identity. Poll same handles, never restart on timeout.
5. [x] Require actual native exits 0,20k checkpoint, standard final test and
   ADAPTER_COMPLETE. Run both fixed common-bank splits using native evaluator.
6. [ ] Measure same-input/all-six-layer native-to-continuous error for trained
   midpoint/RK4, using independent qualified references; compare to sampleSW2.
7. [ ] Evaluate registered outcomes, ordinary controls and measured cost.
   No image promotion from hidden accuracy alone. Do not add another activation.

## Current state and turn classification

This goal turn is progress: published the matched-solver study, passed fresh
frozen readiness, and completed both 20k trajectories plus all four common-bank
evaluations. Supervisor session 40324 and six native commands have actual exit 0.
Independent metric verification and full trained-fidelity observations remain.
The observer author check passes four cases/six layers/eight inputs; all references
qualify and independent reconstruction is below 6.2e-12. Freeze a fresh documented
observer pass before formal 80-input measurements.145 cards 4-7 remain occupied;
none used. Mac CPU environment 04f11f75 is unchanged.

Author readiness: /Users/mr.z/ode-workspace-prep/coefficient-20261116/rk4-control/author-check/PASS.json.
Historical journals retained under docs/history/2026-11-16-before-rk4-control/.

## Retained execution issues

An unused guard was edited while initial solver-bias author verifiers ran;
those outputs are retained but not used for readiness. Stable-source v2checks,
then the separate frozen documented pass, passed before formal observations.
The full solver-bias native/verifier commands subsequently all exited0.
No study process is restarted due to a tool observation timeout.
