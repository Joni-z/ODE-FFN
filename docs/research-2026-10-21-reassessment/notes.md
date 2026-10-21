# Working notes

This review responds to the user's concern that experiment volume has outrun a clear research direction. Local starting commit: 652087e. New training is paused. Primary source verification uses the available web tool; only closely relevant sources will be retained.

## Conclusions retained

- The frozen-duration linear ODE interpretation is valid; a state-gated nonlinear ODE and the external FM ODE have additional/different second-order terms.
- Fixed/scalar equality is a function-class identity, not evidence of optimizer equivalence.
- The K2 matrix parameterization induces a Sylvester differential; it can help or hinder optimization, and no benefit is proved.
- Exact posterior-covariance identities specify target geometry for the current independent Gaussian path. Gaussian/Ring8 have required expansion as well as contraction.
- The gate-to-posterior-direction link is a hypothesis, not an observed mechanism. No pointwise hidden-to-output correspondence is recoverable from the existing aggregated diagnostics alone.
- Current image registry `ode` is ODEOnlyFFN, not OriginalODEFFN; historical checkpoints need execution-time source/config audit before cross-domain interpretation.
- The suggested narrow SiLU run is paused along with other new architecture experiments.

## Verification

`verify_math.py` completed on CPU in the local test environment. Max checked algebra/oracle/autograd discrepancy: 2.842170943040401e-14. No optimization, no generated checkpoint samples, no GPU allocation.

All local Markdown evidence links resolve. Submitted Report 1 PDF SHA256 remains `165d46476362f006db6124f0035bcb6964cb163e509f52eb49adbf4de5839ecf`.

## Search and source coverage

Focused queries covered FM target regularity, posterior covariance, flow conditioning, JiT prediction parameterization, input-dependent continuous-time gating, MeanFlow, and multimodal FM. Final source scope/versions and exact inspected sections are in literature.md. This was not a systematic review. Search snippets and third-party summaries were discovery aids only, not the basis of the final scientific claims. No public query included private checkpoint data or unpublished result tables.

The new research-note literature is separate from the submitted report; no references were added to Report 1.

## Workflow attribution

Used scientific-critical-thinking to organize the distinction between claims, observations, confounds and hypotheses, and planning-with-files to preserve the audit. The scientific-critical-thinking skill requests attribution when used for a report. Its source is Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel and Aubrey M. Brueckner (2026), [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065); author/title metadata was checked against the arXiv record. This is workflow attribution only, not an ODE-FFN research reference or support for any mathematical/performance claim.
