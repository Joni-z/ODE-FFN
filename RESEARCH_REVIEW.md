# Current evidence and one next experiment

The independent review finds no justified new ODELayer candidate. It supports
one bounded test of whether the seven existing image models retain their
quality–cost ordering at a larger fixed training budget. This is not an
architectural breakthrough or a positive result selected by review.

The [briefing](docs/research-next-design-review-2026-10-28/briefing.md) lists
the original problem and primary inputs. A fresh reviewer checked the reports,
CSV tables, learning curves, generated examples, trainer structure and LiFT's
methods. Requested route was native gpt-6-astra/ultra; actual model identity
was not returned. Review assurance remains same-family/provisional. Full
prompt, supplements, response and metadata are preserved privately under
`.aris/traces/research-review/2026-10-28_next-design/`.

## Findings that change the next decision

The frozen Taylor interventions failed, but did not compare independently
trained solver-aware models. This distinction does not itself justify another
Taylor experiment: no task-specific reason has been established for its cost.

The input-frame model's thin-data improvement is real and is largely present
at K1. A useful first-order parameterization need not be rejected just because
K2 fails to explain its gain. This does not rescue the frame setting: thick-data
generation independently fails. A narrow wording correction to the cycle11
analysis is necessary: 77.9% is the full-field-MSE reduction versus Swi192;
78.2% is the tangent-MSE reduction versus the better tangent reference. They
are not two full-field-error percentages. Original numerical results and
stopping decisions remain unchanged.

Ordinary function containment is not optimization equivalence. The fixed-clock
image model is a two-matrix SiLU function, yet cold SiLU64 has worse FID.
The paired synthetic coordinate studies also do not identify a universal
P-parameterization failure. No initialization or optimizer repair follows.

Most importantly, every image regression trajectory is still improving.
For K2/Swi129/SiLU193, the last 1,000 updates reduce weighted validation MSE by
2.05% / 2.23% / 2.63%. These are regression curves, not generation-quality curves;
K2 does not show a favorable catch-up slope. The existing ordering is valid
at 5k, but its stability with more training has not been measured.

## The selected empirical premise test

The [registered protocol](docs/autoresearch/image-budget-continuation/protocol.md)
continues all seven unchanged models from 5k to 20k, with generation/FID at
fixed 10k and 20k, with 20k primary. All optimizer/RNG/buffer state, loss, data,
sampler, seed0 and shape/work controls are retained. The extra budget applies
to all arms, not only an ODE candidate. A fresh documented-entrypoint pass and
seven physical-resume checks gate launch.

A prospective follow-up lead requires 5% lower screening FID than both relevant
ordinary controls, with each fitting metric no more than 5% worse. This is a
new allocation rule for the 20k comparison, not an amendment to earlier failed
conditions. Continuous results and all references remain visible even if it
fails. A qualifying lead would require separate confirmation, not an immediate
ODE-mechanism claim. If no lead emerges, there is no automatic 40k extension.

The possible decisions are concrete: an ordinary frontier redirects attention
away from Taylor order; a K1 operating point favors the simpler first-order
parameterization; a K2 operating point makes its training parameterization and
cost worth investigating with K1 retained. None alone proves that accurately
integrating a hidden field improves the outer FM objective.

## Routes deliberately left unselected

Local recurrence failures do not refute LiFT. Its depth-specific targets use
a detached initial prediction and a continuously conditioned recurrent core,
with normalized input reinjection. That differs from the local same-target
recurrence screen. Implementing the full mechanism now would test an existing
whole-backbone training design and change several factors before establishing
the local-block reference. [LiFT, Sections 3.1–3.2](https://arxiv.org/html/2610.05538v1)
The paper is a 2026 preprint, not evidence of an ODELayer advantage here.

Another generic identity, frame/coefficient/Taylor variant, or enlarged
benchmark is not the next action. Existing negative results stay scoped to
their settings, and the submitted Report1 remains untouched.

## Review outcome and limitations

The reviewer recommends the budget comparison for protocol/preflight only.
It did not replay physical checkpoints, certify launch, establish novelty,
or demonstrate an effective new architecture. The executor accepts that
recommendation and the scoping corrections, while retaining every original
experiment decision. Full-state readiness and empirical results remain
separate obligations. No top-venue contribution is presently established.
