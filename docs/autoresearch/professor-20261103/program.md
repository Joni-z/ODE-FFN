# Task-specific autoresearch protocol

This adapts Karpathy autoresearch at228791fb499afffb54b46200aca536f79142f117;
upstream README.md and program.md are unchanged. Its bundled LM/val_bpb task is
not used. Reuse the established synthetic FM trainer and evaluation. Each cycle has one isolated numerical adapter under
studies/muon_pd_20261102/autoresearch_20261103/: trial.py for cycle1 and
cycle2/trial2.py, cycle3/trial3.py, cycle4/trial4.py and cycle5/trial5.py for their registered cycles. The copied trainer/evaluator and official Muon stay
frozen. Registrations, numerical witnesses and analysis scripts are separately
versioned; no new training loop is introduced.

The adaptation intentionally retains20k updates rather than replacing historical
comparisons with a five-minute cutoff; log actual elapsed time and parameter/MAC
counts. Each host keeps its own task/device pairing. No language data, new packages,
new loss, altered sampler, test-driven checkpoint selection, or all-network Muon.

Cycle loop: evidence -> explicit hypothesis -> commit one-axis candidate -> native
checks -> baseline first ->20k training -> replay -> record keep/discard/crash ->
next evidence-driven cycle. Complete registered comparisons before choosing a new
family. Never discard negative evidence or reset shared Git history. Raw results.tsv
stays outside Git as upstream requests; publish a separate curated decision ledger
and all endpoint metrics so the user can read them on GitHub. Keep data/checkpoint
hashes and native source commits. A failed numerical trial is recorded; no silent
retry. Infrastructure failures preserve their logs and require explicit diagnosis.

Primary quality: final validation oracle MSE. Test SW2 is always reported separately;
it cannot select models or checkpoints. A validation improvement with worse sampling
is a tradeoff, not overall acceptance. Provisional keep requires >=5% validation
improvement over the matching parent at the same optimizer/rate. Test SW2 remains a
reported readout, not a candidate-selection rule; worsened sampling must be called a
tradeoff. Ordinary SwiGLU controls still determine whether there is an FFN advantage. Report all outcomes,
including borderline effects, without significance tests over checkpoint repetitions.

Continue without asking the user whether to proceed; stop immediately on an explicit
user stop. Do not automatically widen to image training or unrelated architecture
search. Fresh native witnesses and actual process/resource monitoring precede each
new numerical runtime.
