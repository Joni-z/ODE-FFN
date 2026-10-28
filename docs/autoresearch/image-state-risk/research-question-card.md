# Research question

Question: does K1's lower image FID coincide with a reversal of conditional
prediction error on shared generated states, and is the available empirical
training-bank conditional mean suitable for interpreting that reversal?

Type: exploratory diagnosis, before selecting another architecture.

Hypothesis: an error ranking on paired states may change on sampler states.
This is not inferred from the FID ranking; it is the missing premise to test.

Evidence: ER-20261029-image-mismatch-01 and ER-20261029-empirical-separation-01
in notes.md; the complete rejected rotation screen remains contradictory
evidence against selecting exact feature flow merely for its numerical meaning.

Missing: common-state predictions, finite-bank reference moments and a
train/validation comparison that exposes a possible empirical-reference mismatch.

Support/falsification: apply the fixed reversal and reference-validity rules
in protocol.md, retaining every ordinary control, domain and time.

Minimal action: run one bounded read-only100-batch comparison and its full
independent replay. Decision: run the registered diagnostic after native preflight;
do not select neural training or claim a new mechanism from its outcome alone.
