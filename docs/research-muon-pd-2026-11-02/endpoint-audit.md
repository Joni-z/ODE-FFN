# Completed grid: endpoint audit

All36 registered seed-0 cells reached20,000updates:22 on145 and14 onLenovo. Both queues finished without failures; no new training was launched. The records include prescribed final test sampling and adapter completion markers.

## What is checked

Every cell matches the registered grid, seed, batch size and20k endpoint. Saved configuration hashes match provenance. Each run has41 metric records and the final sampling result. File hashes for metrics, saved examples, spectra and provenance are retained in `endpoints/source-records.json`. These checks establish file/configuration consistency. Independent saved-example metric recomputation, checkpoint replay and full trajectory analysis remain pending.

## Preliminary descriptive findings

On Product16, scalar ODE field MSE is0.102262 with AdamW3e-4 and0.033431 with AdamW1e-3. Muon0.006 gives0.028314;Muon0.02 gives0.023849. Thus the higher-rate AdamW control explains much of the improvement over the original default; Muon0.02 still reduces field error by28.7% relative to the stronger tested AdamW. The wider SwiGLU AdamW3e-4 control remains better at0.015668 (ODE error is52.2% higher). Its test SW2 is also lower:0.079206 versus0.088098. This supports an optimization improvement over original ODE, not an ODEFFN advantage.

On Ring8, scalar Muon0.006 improves field MSE from0.015426 to0.013401, but wider SwiGLU Muon0.02 reaches0.011231. Scalar Muon0.02 improves sampling distance while worsening field error versus the lower Muon rate; the two readouts cannot be collapsed into one winner. Gaussian scalar Muon does not improve field MSE (0.003037 versus0.002871), although its recorded sampling distance is lower.

For structure at the common AdamW3e-4 setting, PD has higher field MSE than DP on all three tasks. Product16 PD=0.051505 andDP=0.042745 both improve on default scalar ODE but remain worse than scalar AdamW1e-3 and ordinary SwiGLU. Ring8 PD test SW2 is0.224175, compared withDP0.198738 andscalar0.208719. The registered PD intervention has not shown an advantage.

Muon does not shrink P here. Product16 mean per-layer largest singular value is2.029 with scalar AdamW3e-4,7.070 withMuon0.006, and19.173 withMuon0.02. Mean stable rank increases from9.94 to17.62/18.18. The observed fit improvement coexists with larger P norms. These are descriptive endpoint associations, not proof that the spectral change causes the improvement; full Jacobians, gates and update trajectories must be examined.

## Exact recorded endpoints

Field MSE and both SW2 values are lower-is-better. Field MSE uses the fixed validation oracle probe. Test SW2 uses a separate fixed test sampling bank. All rows are reported; no checkpoint was selected using these results. The sampling floor is reference-versus-reference SW2. P summaries average six matrices; SwiGLU projection ranks/shapes differ and should not be compared as if they were the same square operator.

### gaussian_aniso_2d

| Family | Projection optimizer/LR | Field MSE | Test SW2 | Sampling floor |
|---|---|---:|---:|---:|
| scalar_k2 | adamw 0.0003 | 0.002871 | 0.048884 | 0.051840 |
| scalar_k2 | muon 0.006 | 0.003037 | 0.041891 | 0.051840 |
| swiglu128 | adamw 0.0003 | 0.003224 | 0.056238 | 0.051840 |
| swiglu128 | muon 0.006 | 0.003925 | 0.059241 | 0.051840 |
| swiglu192 | adamw 0.0003 | 0.003470 | 0.055701 | 0.051840 |
| swiglu192 | muon 0.006 | 0.004149 | 0.060952 | 0.051840 |
| vector_dp_k2 | adamw 0.0003 | 0.002889 | 0.049541 | 0.051840 |
| vector_pd_k2 | adamw 0.0003 | 0.002900 | 0.049202 | 0.051840 |

### gmm_ring8_2d

| Family | Projection optimizer/LR | Field MSE | Test SW2 | Sampling floor |
|---|---|---:|---:|---:|
| scalar_k2 | adamw 0.0003 | 0.015426 | 0.208719 | 0.153257 |
| scalar_k2 | adamw 0.001 | 0.015654 | 0.197607 | 0.153257 |
| scalar_k2 | muon 0.006 | 0.013401 | 0.181819 | 0.153257 |
| scalar_k2 | muon 0.02 | 0.016714 | 0.173570 | 0.153257 |
| swiglu128 | adamw 0.0003 | 0.012599 | 0.178598 | 0.153257 |
| swiglu128 | adamw 0.001 | 0.015852 | 0.216320 | 0.153257 |
| swiglu128 | muon 0.006 | 0.012885 | 0.173339 | 0.153257 |
| swiglu128 | muon 0.02 | 0.011484 | 0.164520 | 0.153257 |
| swiglu192 | adamw 0.0003 | 0.012961 | 0.187002 | 0.153257 |
| swiglu192 | adamw 0.001 | 0.016151 | 0.213909 | 0.153257 |
| swiglu192 | muon 0.006 | 0.012773 | 0.165181 | 0.153257 |
| swiglu192 | muon 0.02 | 0.011231 | 0.165805 | 0.153257 |
| vector_dp_k2 | adamw 0.0003 | 0.013511 | 0.198738 | 0.153257 |
| vector_pd_k2 | adamw 0.0003 | 0.014423 | 0.224175 | 0.153257 |

### rotated_product_bimodal_16d

| Family | Projection optimizer/LR | Field MSE | Test SW2 | Sampling floor |
|---|---|---:|---:|---:|
| scalar_k2 | adamw 0.0003 | 0.102262 | 0.090331 | 0.058493 |
| scalar_k2 | adamw 0.001 | 0.033431 | 0.109359 | 0.058493 |
| scalar_k2 | muon 0.006 | 0.028314 | 0.096885 | 0.058493 |
| scalar_k2 | muon 0.02 | 0.023849 | 0.088098 | 0.058493 |
| swiglu128 | adamw 0.0003 | 0.016855 | 0.081848 | 0.058493 |
| swiglu128 | adamw 0.001 | 0.017352 | 0.089324 | 0.058493 |
| swiglu128 | muon 0.006 | 0.020706 | 0.098586 | 0.058493 |
| swiglu128 | muon 0.02 | 0.026850 | 0.114173 | 0.058493 |
| swiglu192 | adamw 0.0003 | 0.015668 | 0.079206 | 0.058493 |
| swiglu192 | adamw 0.001 | 0.016849 | 0.088557 | 0.058493 |
| swiglu192 | muon 0.006 | 0.019149 | 0.099398 | 0.058493 |
| swiglu192 | muon 0.02 | 0.016653 | 0.084756 | 0.058493 |
| vector_dp_k2 | adamw 0.0003 | 0.042745 | 0.087277 | 0.058493 |
| vector_pd_k2 | adamw 0.0003 | 0.051505 | 0.087924 | 0.058493 |

## Evidence boundary and next check

This is a descriptive endpoint audit, not a completed mechanism analysis or an image-generation result. No significance tests or confidence intervals are computed from checkpoint repetitions. Next: independently recompute saved-sample SW2, replay fixed checkpoints, inspect time-resolved P/update spectra and entrance-gate Jacobians, and draw the registered whole-curve comparisons. The completed grid is preserved; no extra training family is queued.
