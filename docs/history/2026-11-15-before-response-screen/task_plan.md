# Order-by-response study, 15 November 2026

Authorized: complete the K1/K8 x ISRU/tanh comparison on Ring8 and Product16.
Use 145 physical GPUs 4-7 only; seed 0. Reuse existing K1 and ordinary controls.
Preserve finalized reports, native trainers, old runs and image baseline.

- [x] Verify resources and read source-bound K1 protocol/results.
- [x] Register four K8 runs and bind historical K1 checkpoints/evaluations.
- [x] Implement order-only adapter, shared evaluator bridge and term diagnostics.
- [x] Verify explicit Taylor powers, full entrance-diagonal Jacobian (dim2), unchanged initialization and native8->16 continuation.
- [x] Fresh documented witness; push/freeze source; launch four monitored pipelines.
- [x] Complete native/C8 sampling,20k order truncation and layer-term diagnostics.
- [x] Independently recompute metrics; compare response-by-order effects and ordinary controls.
- [x] Publish findings and next decision; record open research questions.

GPU4 Ring8/ISRU-K8; GPU5 Ring8/tanh-K8;
GPU6 Product16/ISRU-K8; GPU7 Product16/tanh-K8.
Same20k updates, batch512, AdamW3e-4, eight common4096sample banks and Heun128.
Equal parameters and updates; K8 uses eight P applications per internal solve.
No new optimizer sweep, scaling, spectral caps or rotation augmentation.

Current: four pipelines completed at frozen9f72bd3, all exits zero, GPUs idle.
NumPy recomputation passed (max sampled-metric difference2.8e-17).
Figures inspected. Findings and evidence published to GitHub in2639792.
The four-run study is complete; the proposed rotation-training control and
an ODE-specific quality/cost advantage remain open. Source sync is performed
through GitHub; the145 experiment worktree remains frozen at9f72bd3.

Readiness1 failed at helper import: historical check.py shadowed response
check.py. Explicit-path imports fix the wrapper; attempt2 uses new output
paths and preserves the original failure. Both documented checks then passed;
four real K1 sampling replays, four exact continuations/evaluator round trips,
and unchanged parameters through frozen-order diagnosis were verified.

## Proposed next work after the professor follow-up

Planning discussion only; no new jobs are authorized by this note.
- [x] Recheck the three professor-comment findings and the completed K1/K8 response comparison.
- [x] Prioritize the existing CelebA checkpoint diagnosis over additional response/order sweeps.
- [ ] Measure native SwiGLU versus widened ISRU-D K1/K8 errors by noise time and layer responses; cap the proposed diagnosis at one GPU-hour.
- [x] Record the Tanh-D K8 proposal, subsequently superseded by the requested response-family comparison.
- [ ] Review the response comparison, implement and verify it under an execution decision; no run is launched here.

Ring8 rotation augmentation remains an optional small control; it is not a
prerequisite for answering the professor or evidence of an image-model gain.
The earlier ISRU-midpoint plan remains documented in
docs/research-assessments/2026-11-14-next-architecture/execution-plan.md but is
not selected as the default next experiment. The user correctly challenged
the reason to privilege ISRU after the matched tanh results. A fixed-response
feedback comparison could isolate one variable, but that alone does not make
it the research priority. Neither a post-hoc improvement nor a post-hoc
failure proves the potential of a newly trained operator.

Current proposal: docs/research-assessments/2026-11-14-next-architecture/response-screen-proposal.md.
The earlier tanh-only image transfer is superseded. Compare representative
responses under the same recipe before selecting up to two image candidates.

## Response-family selection revision

The user requests comparative selection rather than choosing tanh after one
ISRU contrast. Prepare the bounded comparison before another image run.
- [x] Recheck existing response evidence and standard function definitions.
- [x] Register representative response families, scale matching and K1/K8 controls.
- [x] Specify fresh selection/confirmation measurements and image-transfer decisions.
- [x] Verify the response formulas and update the superseded tanh-only proposal.
- [x] Publish the revised plan and record that training remains unlaunched.

No new trainer, optimizer sweep, learned response scale or formal job in this
planning turn. Preserve all completed results and finalized reports.

Planning deliverables: response-screen-proposal.md, eight-family float64
normalization/derivative checks and inspected curve plot in the existing
next-architecture folder. All96 report files match their protected hashes.
No model trainer or scheduler source changed; formal runs remain proposed.
