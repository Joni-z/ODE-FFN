# Proof audit: objective-level denoiser dynamics

Substantive result: no open issue in U0–U8 after two notation clarifications.
Assurance: **native same-family / provisional**. This is not cross-family
semantic acceptance, formal proof certification, theorem novelty or empirical
evidence for an architecture. Numeric evidence and citations were outside
the reviewer's scope.

## Source and coverage

The [derivation](derivation.md) and [obligation skeleton](PROOF_SKELETON.md)
define nine source units. One fresh read-only review covered all nine and
extracted 35 deterministic microclaim IDs. The same review session rechecked
all units after revision. The [coverage receipt](results/coverage-and-verdict.json)
records both rounds. Coverage proves execution over declared source units;
it does not prove that every possible obligation was discovered.

Reviewer routing selected the native codex backend, requesting gpt-6-astra
with ultra effort. The tool did not return an independently attested actual
model identity, so that field remains null. No stronger independence claim
is made. HTML rendering is disabled for this CLI-oriented package; Markdown
and a standalone audit-summary PDF are the reusable formats.

## Before and after

| Issue | Initial finding | Revision | Recheck |
|---|---|---|---|
| R1 | Conditional expectation notation is an almost-everywhere class, while derivatives are pointwise | Explicit smooth Bayes-ratio versions of m and D | Resolved; no theorem scope change |
| R2 | Undefined p_t notation and a vector/scalar domain ambiguity | State uniqueness under the law of Z_t and y in R^d,r>0 | Resolved; no mathematical change |
| Dependency | U6's FM-coordinate application also uses U4 | Add U4 to skeleton row | Verified acyclic |

Initial issues were cosmetic/minor, with no fatal or critical finding. The
reviewer checked derivative domination and posterior positivity, all PDE
signs/scalings, both exact Gaussian excess risks, the precision third-moment
expansion and both realizable posterior examples. Degeneracy checks include
point masses, singular Gaussian priors, two-point laws, coincident components,
small positive mixture weights and excluded endpoint regimes.

## Trace and remaining limits

The complete [round1 report](results/raw-round1.md) and [round2 report](results/raw-round2.md)
are preserved verbatim. The reviewer did not edit the proof, run experiments
or use numerical outputs as proof evidence. The executor applied the two
notation changes. The runtime retains its original registered mathematical
snapshot; the revised public derivation changes no formula, prior class or
numerical criterion.

The separate [prior application receipt](results/assumption-discharge.json)
checks the actual three benchmark-law files. All numerical gates and both
finite-difference steps pass. These deterministic checks are accepted only
within their computational scope; they do not upgrade semantic assurance.
No architecture, PDE regularizer or changed training target is selected.
