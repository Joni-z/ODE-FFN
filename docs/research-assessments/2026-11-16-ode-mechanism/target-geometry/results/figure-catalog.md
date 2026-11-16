# Figure catalogue

## jacobian-versus-sampling.png / .pdf

Purpose: compare the prespecified whole-output response metric with generation
on all twelve checkpoints. Data: independently reconstructed summary.json,
confirmation scores bound in native-rows.json. One point per checkpoint; no
error bars or independent-training inference. Notice whether the focal pair
orders the same way on both axes. This determines whether the proposed average
response-error explanation survives; it does not assign causality.

## time-resolved-jacobian.png / .pdf

Purpose: expose every registered time slice behind the equal-time summary.
Data: summary.json/by_time. Each point averages128 common interpolated states;
the five times reuse the same pairs. No error bars or independent-time inference.
Notice where errors differ, while retaining the primary average decision.
This distinguishes an overall failed explanation from a possible new,
explicitly registered trajectory/time-dependent question.
