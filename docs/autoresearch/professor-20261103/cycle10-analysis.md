# Cycle10 analysis contract

Runtime7e394e2 is frozen. Later scalar-log readers and figures are analysis-only
and live outside native runtime worktrees. They cannot change model predictions,
training, the validation oracle or the test sampler.

1. Require six completed cells/host, no pending/failed workers.
2. Run native audit_completed.py with --variant-module
   autoresearch_20261103/cycle10/trial10.py, explicit experiment id,
   --endpoint-step20000 and --expected-count6 on each original CUDA host.
3. Run cycle10/anchors.py for four exact20k references/host, without --only-ready.
   Its earlier partial receipts are progress only, never closure evidence.
4. Collect native endpoint wall times separately from quality replay. Read final
   samples with the existing cycle6/sample_readouts.py, explicit cycle10 id,
   endpoint20000 and expected-count6. Primary scores are unchanged.
5. Use the scalar-log collector to retain every logged layer, denominator and
   response; bind metric/config hashes. It passed a real six-cell16-step native
   log check before final use. These are observed native features, not causal
   interventions or global predictor bounds.
6. Curate all twelve endpoints with hash-bound parents and eight anchor
   receipts. Append their unique IDs to the external trial ledger exactly once.
7. Render all six models/task, learning and P/response histories, every registered
   Product16 axis0/7/15 example and the same Ring display indices. No favorable
   sample subset or checkpoint is chosen.
8. Report radial against raw, static and each ordinary width. The general
   provisional parent ledger rule is not mechanism acceptance. Radial can be
   selected for longer confirmation only if its validation gain is at least5%
   against both raw and static. Sampling remains a separate tradeoff readout.
   The existing80k ordinary frontier remains visible.

No training continuation is selected from partial results. No image transfer,
rate/strength rescue, P constraint or second mechanism is bundled into this test.
