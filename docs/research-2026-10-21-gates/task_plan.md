# Gate necessity study

- [x] Review Report 1's proposed follow-up and existing trainer.
- [x] Freeze the question, controls and evaluation rule in protocol.md.
- [x] Implement fixed and learned scalar durations with paired common weights.
- [x] Verify legacy behavior, gradients, checkpoint loading and split isolation.
- [x] Freeze configs and sources; benchmark concurrency and launch monitoring.
- [x] Collect results and separate observations from explanations.

Status: All 15 CPU runs and 27 endpoint evaluations completed and audited.
Training source commit: 844a396.
GPU workers respawned after both reclamation attempts; our guard stopped the GPU
smoke attempt. The CPU pilot measured 69.98 versus 138.04 updates/s for one versus
two workers. All formal comparisons use the CPU backend. A local collector also
monitored completion and audited endpoint samples after retrieval.
The 2,048-point results and figures are available under results/.
The 27-cell frozen-checkpoint 8,192-sample check also completed; all saved sample
hashes and both target-set scores were audited. Native sample prefixes reproduced
exactly. See findings.md for the analysis and the next control proposed for a
research-direction decision.
Submitted Report 1 is immutable for this study.
