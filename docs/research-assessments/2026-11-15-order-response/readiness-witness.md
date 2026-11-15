# Readiness doc witness v2: passed

Input: /Users/mr.z/ODE-FFN-upstream/docs/research-assessments/2026-11-15-order-response/readiness.md

Both documented shell blocks executed verbatim on SSH145 using physical GPU4 and warm py312, at clean HEAD9f72bd34daa242aa1a8812b5a490e94e7cee970c. The second ran only after first exit0 and PASS.json verification.

## First command
- Exit0; /data2/zz5070/ODE_FFN/order-response-20261115/preflight/documented-v2/kernel.json passed.
- anchors.json: four exact native K1 sampling replays.
- PASS.json: four completed cells, every resume16_exact and evaluator_roundtrip_exact true.
- Full diagonal Jacobian flags true for dim2, false for dim16; do not claim full16D Jacobian coverage.

## Second command
- Exit0; /data2/zz5070/ODE_FFN/order-response-20261115/preflight/diagnostic-v2/COMPLETE.json passed=true,witness=true.
- all_parameters_buffers_unchanged=true; orders_restored=8.
- Term reconstruction uses exact torch.equal assertion in diagnostics.py:26, executed successfully on native term observations.
- terms.json, truncation.json, banks.json, COMPLETE.json written.
- Checkpoint hash82ef961b1c6478c9b9d16462455a7022c136d69c3e2b29dc3b2a17a04d311eda matches PASS.json first case.

## Post-state
Remote HEAD unchanged and clean. GPU4 idle at2MiB/0%. No source edits, package changes, formal20k training, GPU0-3 use, or prior-output deletion. Previous report.md and failed execution logs retained. V2 logs: check_order.v2.execution.log, diagnostics.v2.execution.log.
