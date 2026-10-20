# Report 1: evidence map

This is a research progress report, not a claim of a validated architectural improvement. The report is an English draft for author review. The assistant checked the evidence below on 2026-10-20; no human verification or submission approval is claimed.

## Scope and sources

The evidence revision is `84ff784`. Paths below are relative to the repository root. `evidence_manifest.json` records SHA256 hashes; `data/report_values.json` contains the numeric values used in the figures. No training was launched while preparing this report.

| Report claim | Source and locator | Interpretation limit |
|---|---|---|
| Original ODELayer, two solves, gate range, and capped operator | `synthetic/models.py`: `OriginalODELayer`, `operator`, `OriginalODEFFN` | Hidden dynamics, not the Jacobian of the generative velocity field |
| Taylor series and complete layer derivative | Direct algebra from `OriginalODELayer.forward`: differentiate each term with the gate evaluated once | Standard identities, not new theoretical results; for time gates, take the hidden derivative at fixed external time |
| Target parameters | `synthetic/data.py`: `suite`; `synthetic/geometry.py`: `GeometricTarget`, `available_suite` | Chosen settings, not an exact paper benchmark reproduction |
| Dataset split sizes and seed | `configs/synthetic/data_v1.json` | Generation seed is 20260923, distinct from training seed 0 |
| Model widths, optimizer, seed, steps, sample sizes, and budgets | `docs/research-2026-10-20/protocol.md`, `experiment-manifest.json`, and `configs/synthetic/train_v1.json` | Approximately matched parameters; forward linear MACs are not GPU runtime |
| 30 trajectories and 54 diagnostics | `docs/research-2026-10-20/stage-summary.json`: `progress`, `diagnostics`, `curves`; `completion-audit.json` | Includes baseline prefix technical reruns; do not count stages/checkpoints as independent replications |
| SW2 implementation and fixed target reference | `synthetic/train.py`: `sliced_w2`, `fixed_eval_data`, `evaluate` | Finite-sample projected distance; target-versus-target values are not hard lower bounds |
| Experiment 1: cap and Jacobian | `docs/research-2026-10-19/diagnostics/{dataset}.json`: `runs`, filter `ode_original` / `ode_spectral_cap` and `orders=1,2`; `sampling.32.sw2`; mean of `layers[*].full_layer_jacobian_norm.mean` | All six lower Jacobian norms, only three lower SW2 scores; separately trained models |
| Experiment 2: fixed-update and MAC comparisons | Stage summary: labels `silu_5000`, `swiglu_5000`, `gate_k2_input` at step 5000; `swiglu_budget` at step 6572; `parameters`, `forward_macs`, `sampling.32.sw2` | Fixed endpoints, no best-checkpoint selection |
| Endpoint instability | Stage summary: diagnostics for Gaussian `swiglu_5000` at 5000 versus `swiglu_budget` at 5036, and Ring8 `silu_5000` at 5000 versus `silu_budget` at 5024 | Correlated checkpoints; no significance claim |
| Experiment 3: gate changes | Stage summary: `gate_k{1,2}_{input,time}`, step 5000; `sampling.32.sw2`, `layers[*].gate_correction_norm.mean`, `full_layer_jacobian_norm.mean` | Initial weights/count match, but gate inputs and optimization differ |
| Ring8 case study | Earlier Ring8 diagnostics: `final_oracle_mse`, `time_bins[0,7].oracle_mse`, `sampling.{32,256}.sw2` for `ffn`, `swiglu`, `ode_original` K2 | Measured on interpolated held-out points, not generated trajectories; error timing is not a causal explanation |
| Experiment 4: order and exponential action | Stage summary: labels `gate_k1_input`, `gate_k2_input`, `order_k4_input`, `order_k8_input`; `forward_macs`, `sampling.32.sw2`, mean `layers[*].relative_action_truncation_error.mean` | Each order uses its own trained operator; K8 is not exact exponentiation |
| Experiment 5: learned P | Stage summary: `curves` for the four input-gated orders; mean `records[*].ode_layers[*].effective.{sigma_max,stable_rank}` | Six-layer averages, not uncertainty estimates; stable rank is not algebraic rank |
| Image limitations | `docs/legacy-overview.md`, `docs/research-2026-10-20/stage-report.md`, synthetic implementation | No new ImageNet/FID result is claimed |
| Historical manifest correction | `docs/research-2026-10-20/data-content-audit.json` and the accompanying stage report | Preserve the frozen annotation and separate correction; do not rewrite the historical record |

## Visualizations

Run `python reports/report1/make_figures.py` from the repository with NumPy and Matplotlib installed.

1. `01-data-and-comparison.pdf`: actual target examples plus fixed-update and budget comparison. The CSV contains the first 1,000 `x1` rows from each actual torch test archive, exported without resampling. These are target points, not model-generated examples.
2. `02-spectral-cap.pdf`: arrows connect independently trained free/capped models with the same order. Each dataset has its own axes.
3. `03-gate-intervention.pdf`: time-minus-input differences, for both generation and full-layer sensitivity.
4. `04-order.pdf`: generation and exponential-action discrepancy across separately trained orders.
5. `05-p-trajectories.pdf`: descriptive six-layer means over training, with distinct markers and colors.
6. `06-ring8-case.pdf`: time-resolved oracle error, long-baseline endpoint variation, and solver resolution. The dotted line is a fixed ODE endpoint.

All training results use seed 0. There are no independent-seed error bars, p-values, or claims of statistical significance. The illustrative test-point selection is deterministic and does not select examples based on model performance.

## Literature checks

The following primary records were opened on 2026-10-20 to verify titles, authors, identifiers, and the limited background claims cited in the report:

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747): simulation-free regression of velocity fields.
- [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720): JiT and direct clean-image prediction.
- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202): gated feed-forward variants.

The report does not reproduce results from those papers or claim that the synthetic protocol exactly follows any of them.

## Proposed work

H1, H2, and H3 are plans for discussion, not completed experiments. H1 is the first priority; H2 separates operator scale and geometry; H3 is conditional on the chosen direction. New training, if selected, should retain seed 0 and the baselines, and fix an endpoint policy before looking at new outcomes. This report does not authorize itself to launch those experiments.
