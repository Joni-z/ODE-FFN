# Report 1: evidence map

This is a progress-report draft, not a validated image-architecture claim.
The assistant checked source artifacts and rendered pages on 20 October 2026;
no human verification or submission approval is claimed. The only named author
and PDF author metadata are **Zhizhe Zhang**. The NeuroMotion report supplied
layout and organization only; its scientific content and other authors were not
transferred.

Main text occupies pages 1–3. Page 4 contains references and a numeric comparison;
pages 5–6 contain supporting visualizations. The final main-text section gives
uncompleted hypotheses for the next two weeks. H1/H2 from the earlier report
are now completed diagnostics, not still-future promises.

## Source binding

`evidence_manifest.json` records source hashes. Earlier evidence was assembled at
commit `84ff784`; frozen scientific code came from
`1dd6a86daed8521bea71c5faf3df12107cff7d07`. The follow-up uses the relocated manifest,
whose scientific input hashes match the original. Curated figures can be rebuilt
without GPUs using the commands in README.md.

| Claim / display | Source and locator | Bound on interpretation |
|---|---|---|
| ODELayer, two layers, duration range, capped operator | `synthetic/models.py`: OriginalODELayer/OriginalODEFFN | Hidden dynamics; P is not the full generative Jacobian |
| Taylor remainder and full gated Jacobian | Direct differentiation of the implemented finite polynomial; `docs/research-2026-10-20-followup/mathematical-notes.md` | Standard identities, not new theory or a performance theorem |
| Targets and splits | `synthetic/data.py`, `synthetic/geometry.py`, `configs/synthetic/data_v1.json` | Chosen settings; generation seed 20260923 differs from training seed 0 |
| Width, blocks, optimizer, endpoint and budgets | `docs/research-2026-10-20/protocol.md`, `experiment-manifest.json`, `configs/synthetic/train_v1.json` | MAC proxy excludes backward/nonlinear/optimizer costs |
| SW2 and oracle protocol | `synthetic/train.py`: sliced_w2, fixed_eval_data, generated_samples | Finite-sample projected metric; Heun uses two network evaluations per step |
| E1: baseline table | `docs/research-2026-10-20/stage-summary.json`: diagnostics labels silu_5000, swiglu_5000, gate_k2_input at 5000; swiglu_budget at 6572 | Single fixed endpoints, no best-checkpoint selection |
| Endpoint sensitivity | Same summary: Ring8 silu_5000 at 5000 vs silu_budget at 5024 | Correlated checkpoints, not separate seeds |
| E2: complete order matrix | `docs/research-2026-10-20-followup/frozen-order-summary.json`: all 66 rows | Frozen parameters; downstream gates recompute |
| Native and precision controls | `collection-audit.json`, `native-145.json`, same-order rows in frozen summary | Maximum native/reference SW2 drift 4.47e-8; tolerance 5e-4 |
| Low-order exact effects | `exact-effects.json`, `exact-effects-table.md`, all K1/K2 rows | Six target/checkpoint comparisons, not six independent seeds |
| K4/K8 adaptation versus saturation | Ring8 rows order_k4_input at eval4/8 and order_k8_input at eval8/4; all K8 exact rows | Initial <=1% flag is descriptive, not an equivalence test |
| Local fixed-input accuracy, Figure 4 | `probes-145.json`: six layers × 16 native probes, by evaluation order | Local averages; not global bounds or full-network interventions |
| Endpoint examples, Figure 2 | `data/ring8-frozen-examples.csv`; full arrays identified by hashes in frozen summary | First 1,000 actual target/source indices; not selected visually |
| Solver check, Figure 3 | `solver-check-summary.json`, `solver-check-audit.json`; 48 cells | Post hoc Ring8 check; finite resolutions, no convergence theorem |
| Ring8 K4 oracle/SW2 mismatch | Frozen summary, order_k4_input eval4 vs exact | Error timing/propagation is not causally isolated |
| E3: spectral cap, Figure 6a | `docs/research-2026-10-19/diagnostics/{dataset}.json`: free/cap at K1/K2; mean full_layer_jacobian_norm | Six comparisons; norm averaged over inputs/layers, not a global bound |
| E3: time gate, Figure 6b | Stage summary: gate_k{1,2}_{input,time}; gate_correction_norm and full_layer_jacobian_norm | Retraining changes gate inputs and learned operators |
| P trajectories, Figure 5 | Stage summary: curves for four input-gated orders; six-layer means of effective sigma_max/stable_rank | Figure shows K2; text ranges span all orders; stable rank is not algebraic rank |
| Constructed operator family, Figure 5 | `controlled-operators.json`: matched_eigenvalues_and_norm; `scripts/controlled_operators.py` | Same eigenvalues and largest singular value, not all singular values; illustrative examples |
| Additional K8 compute | Stage summary: forward_macs for order_k4_input vs order_k8_input | 63.7% forward-linear-MAC increase, not measured wall time |
| Training / diagnostic counts | Stage completion audit: 30 trajectories, 54 diagnostics | Includes baseline-prefix technical runs; not independent replications |
| Collection / resources | `collection-audit.json`, `solver-check-audit.json`, full GPU CSV traces | GPU samples include startup; short concurrency benchmark is not robust speed evidence |

Follow-up filenames without a directory prefix are under
`docs/research-2026-10-20-followup/`. The analysis bundle includes an exact table,
claim candidates, a statistical appendix and a figure catalog. Raw archives,
checkpoints, datasets and complete generated arrays remain outside Git.

## Literature

The primary records were opened on 20 October 2026 during report preparation to
verify titles, authors, identifiers and the limited background statements:

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747): velocity-field regression.
- [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720): JiT and clean-image prediction.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202): gated FFN variants.

No external benchmark results are reproduced or claimed here. The main report
explicitly limits conclusions to seed-0 synthetic velocity networks without attention.

## Final verification

Verify all hashes in `evidence_manifest.json`, compile LaTeX, inspect extracted
text for author and page boundaries, and render every page. Figure review includes
axis labels, log-axis minor tick suppression, common scatter limits, and correct
separation of trained order from evaluation order. Numerical values are generated
from source JSON; intervals and p-values are deliberately absent because there
are no independent training-seed replications.
