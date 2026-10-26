# Focused literature review: what does it mean to ODE-ize a model?

## Objective

Explain concrete architecture-level ways of making a model ODE-like, grounded in verified primary literature, and assess their relation to the original ODELayer and diffusion/flow generation. Emphasize recent strong evidence and practical mechanisms rather than a long list of loosely related papers.

## Scope

- Distinguish external sampling time, internal network depth, and sequence-state time.
- Include continuous-depth networks, repeated/shared blocks, learned integrators, and explicit/structured flow maps.
- Use older foundational work only to establish what each mechanism means; prioritize relevant recent advances and controlled comparisons.
- Separate architecture changes from training/sampling changes and from theoretical reinterpretations of an unchanged architecture.
- Search official arXiv, OpenReview/proceedings and ACL/PMLR records; log queries and source limitations. This is a focused narrative review, not an exhaustive systematic review.
- Read papers rather than relying on code. No new experiments or training launches are part of this request.

## Phases

- [x] Scope the review and inspect relevant existing project notes.
- [x] Search and screen primary literature across the defined mechanisms.
- [x] Read methods and relevant ablations; verify exact bibliographic details and practical limitations.
- [x] Write a short synthesis and create a conceptual architecture diagram.
- [x] Audit source support and save the review for an intuitive answer.

## Status

Complete. The review and corrected diagram are saved. Main recommendations are LiFT, Looped-DiT and DeepFlow, with explicit preprint/published distinctions, component/budget limits and adverse results. Older papers establish categories rather than being presented as frontier recommendations. No training was launched.

## Verification and access limitations

- Titles, dates/venues and methods are checked against the primary records listed in notes.md and review.md.
- LiFT is explicitly a discrete depth-conditioned recurrence, not an inner numerical solver. Its B/2 failures, operating-point dependence and analytic-FLOP/hardware distinction are retained.
- Looped-DiT's matched-cost table and component ablations are separated; the complete method is not attributed to weight sharing alone.
- DeepFlow's uniform-time component ablation is kept separate from its lognormal-time compute table.
- The packaged DOI verifier could not run because requests is absent from the existing environment. Bibliographic verification uses the opened primary records; no automated verification success is claimed and no dependency was installed.
- The AI-generated figure was inspected; its initial decreasing-time label was corrected. Original files are preserved, and the corrected copy is included in figures/.
