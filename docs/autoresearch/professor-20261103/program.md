# Task-specific autoresearch protocol

This adapts Karpathy autoresearch at228791fb499afffb54b46200aca536f79142f117;
upstream README.md and program.md are unchanged. Its bundled LM/val_bpb task is
not used. Reuse the established synthetic FM trainer and evaluation. Each cycle has one isolated numerical adapter under
studies/muon_pd_20261102/autoresearch_20261103/: trial.py for cycle1 and
cycle2/trial2.py, cycle3/trial3.py, cycle4/trial4.py and cycle5/trial5.py for their registered cycles. The copied trainer/evaluator and official Muon stay
frozen. Registrations, numerical witnesses and analysis scripts are separately
versioned; no new training loop is introduced.

Cycles1–5 intentionally retain20k updates rather than replacing historical
comparisons with a five-minute cutoff. Registered cycle6 alone uses80k for every
arm, verifying exact20k numerical prefixes and retaining matched ordinary
controls; it tests budget/convergence without a new model. Log actual elapsed time and parameter/MAC
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

Cycle7 is an explicitly registered read-only intervention diagnostic, not a new
training cycle. Its54 frozen cases preserve the native evaluator and six fixed
checkpoints. Record sensitivity outcomes separately from training keep/discard;
no new training-model claim can follow from this diagnostic alone.

Registered cycle8 returns to20k training: an explicit optimizer adapter applies
a weak norm-preserving spectral relaxation every10native updates, with one
registered10k-update relaxation time and paired scalar-contraction/native arms.
The copied trainer, loss, model builders and official Muon are unchanged; the
post-step constraint and its total-update logging are intentionally isolated.
No strength/rate/gate sweep is authorized automatically by a negative outcome.

Registered cycle9 is a read-only native-feature/finite-series accounting
follow-up to cycle8, not a new architecture or training trial. It preserves the
earlier no-whitening decision and every primary quality metric. No matrix
inverse, feature transform, gradient update or test-driven selection is used.

Registered cycle10is a separate D-function question: input-radial versus static
scaling, with raw and three native ordinary references. Vanilla Muon/native loss
return unchanged; no rejected P constraint is combined. Fixed20k screen, with
any continuation requiring its own matched-budget registration.

Registered cycle11 completes the rank32-controller by output-transform
comparison from cycles1/2: one missing factorized-linear/affine-output cell at
the three existing optimizer/rate settings pertask, plus native references.
K2 and the native trainer/official optimizers remain unchanged. This is not
continuation of rejected radial or historical K1/DP range settings.
