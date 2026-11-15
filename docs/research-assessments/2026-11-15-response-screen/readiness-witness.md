# Fresh documented CPU readiness witness

Both documented shell blocks executed verbatim, in order, and exited 0. No mismatch or repair occurred.

## Inputs and scope
- Repository: /Users/mr.z/ode-workspace-prep/response-screen-20261115/repo
- Original and final HEAD: b5fe7aa279d5ce027f9658334376d906560d844b
- Original and final worktree status: clean.
- Provider: response-screen-cpu@04f11f75.
- Existing Python 3.12.4, Torch 2.10.0, NumPy 2.2.6; macOS arm64 CPU only, one Torch/OMP/OpenBLAS/MKL thread per worker.
- Author preflight had ended before benchmark; prebenchmark-processes.txt records no active Python check_screen, screen_trial, pytest, or benchmark process.
- No installs, source edits, remote operations, deletion, or formal 20k training. All generated artifacts and receipts are outside the repository.

## Check results
- Kernel: 16/16 polynomial/gradient and initialization-RNG checks passed. Only the two softsign zero cases use the documented nonsmooth-zero exclusion.
- Native continuation: 16/16 native 8-to-16 resumes and continuous-16 comparisons exactly match; evaluator restoration also exactly matches for every cell.

| Cell | Resume exact | Evaluator restoration exact |
|---|---|---|
| rotated_product_bimodal_16d__screen_isru_k1__adamw_0.0003 | true | true |
| gmm_ring8_2d__screen_isru_k8__adamw_0.0003 | true | true |
| rotated_product_bimodal_16d__screen_tanh_k1__adamw_0.0003 | true | true |
| gmm_ring8_2d__screen_tanh_k8__adamw_0.0003 | true | true |
| rotated_product_bimodal_16d__screen_arctan_k1__adamw_0.0003 | true | true |
| gmm_ring8_2d__screen_arctan_k8__adamw_0.0003 | true | true |
| rotated_product_bimodal_16d__screen_softsign_k1__adamw_0.0003 | true | true |
| gmm_ring8_2d__screen_softsign_k8__adamw_0.0003 | true | true |
| rotated_product_bimodal_16d__screen_erf_k1__adamw_0.0003 | true | true |
| gmm_ring8_2d__screen_erf_k8__adamw_0.0003 | true | true |
| rotated_product_bimodal_16d__screen_silu_k1__adamw_0.0003 | true | true |
| gmm_ring8_2d__screen_silu_k8__adamw_0.0003 | true | true |
| rotated_product_bimodal_16d__screen_gelu_k1__adamw_0.0003 | true | true |
| gmm_ring8_2d__screen_gelu_k8__adamw_0.0003 | true | true |
| rotated_product_bimodal_16d__screen_linear_k1__adamw_0.0003 | true | true |
| gmm_ring8_2d__screen_linear_k8__adamw_0.0003 | true | true |

## Actual evaluator witness
- erf K8, gmm_ring8_2d, seed 0, checkpoint step 16, selection split.
- COMPLETE.json: passed=true, witness=true, parameters_unchanged=true.
- Witness checkpoint/config/manifest hashes match the corresponding continuation record.
- Checkpoint SHA256: c244a535024e09b5a783a6ea5bd879521e155ef82d87994b07d495b08057f2b4
- Witness field_mse=2.3202102452278166; sw2_mean=1.1972298339861167; sector_l1=0.1875. These are readiness-witness values, not formal training or quality results.

## Native 200-step CPU benchmark

| Workers | Total wall seconds | Updates/second | Native training wall seconds |
|---|---:|---:|---|
| 1 | 12.618727875 | 15.849458201 | 6.913907041 |
| 2 | 13.357225583 | 29.946338595 | 7.375349750, 7.194940542 |
- Two-worker throughput ratio: 1.889423487923.
- Selected concurrency: 2.
- All three trial logs end at step 200; outputs are in separate benchmark folders.

## Output paths
- First PASS: /Users/mr.z/ode-workspace-prep/response-screen-20261115/preflight/documented-v1/PASS.json
- Evaluator COMPLETE: /Users/mr.z/ode-workspace-prep/response-screen-20261115/preflight/documented-v1/evaluation-witness/COMPLETE.json
- Benchmark PASS: /Users/mr.z/ode-workspace-prep/response-screen-20261115/preflight/benchmark-v1/PASS.json
- Execution receipts: /Users/mr.z/ode-workspace-prep/response-screen-20261115/preflight/agent-readiness-20261115T142317Z

## Changed and untouched files
- Changed repository files: none.
- New outputs: documented-v1/, benchmark-v1/, and this receipt directory under preflight/.
- All tracked source and documentation files remained untouched; final git status is clean.
