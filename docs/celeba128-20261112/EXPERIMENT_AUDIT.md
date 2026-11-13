# CelebA128 checkpoint100 integrity audit

13 November 2026. Fresh reviewer, requested gpt-6-astra / ultra. Native transport
did not expose an actual returned model identity; semantic acceptance remains
provisional. This document reproduces the review with repository-relative links.
The original response is retained locally under `.aris/traces/`.

**Overall judgment: PASS for the recorded checkpoint100 comparison; WARN for causal interpretation and generalization. No invalid measurement was found. The current launcher is not ready for a distinct follow-up run.**

The defensible conclusion is: **at the registered 100-epoch, seed0 exposure, native SwiGLU achieved lower FID and training loss than ISRU-D K8, with less training time.** This does not identify why K8 lost.

| Check | Verdict | Evidence and interpretation |
|---|---|---|
| **A. Ground-truth/reference provenance** | **PASS** | Training and FID use the official CelebA training split through [util/datasets.py:13](../../util/datasets.py#L13). I freshly hashed the archive and all five metadata files on Torch: all match torchvision’s official MD5 values and the saved SHA256 values. Independently counted train/valid/test = **162,770 / 19,867 / 19,962**. Reference preprocessing is deterministic author crop plus `PILToTensor`, without augmentation: [engine_jit.py:152](../../engine_jit.py#L152). Official checksums and split selection were cross-checked against [torchvision v0.22.1](https://raw.githubusercontent.com/pytorch/vision/v0.22.1/torchvision/datasets/celeba.py). |
| **B. Score and image normalization** | **PASS** | Training converts uint8 pixels to `[-1,1]`: [engine_jit.py:36](../../engine_jit.py#L36). Sampling reverses that transform, clips, rounds and saves uint8 PNGs; RGB-to-BGR conversion correctly precedes OpenCV output: [engine_jit.py:130](../../engine_jit.py#L130). The installed Inception extractor requires uint8 RGB, resizes to 299 and applies `(x−128)/128`. No custom FID rescaling was found. |
| **C. Results exist and numbers agree** | **PASS** | Remote raw `result.json`, original console logs and TensorBoard independently agree. SwiGLU FID = **35.91785430908203**, K8 = **43.90766906738281**; difference = **7.989814758300781**, relative increase = **22.244688%**. Each retained sample directory contains **50,000 PNGs**. Local records: [FID100-swiglu.json:18](../../docs/celeba128-20261112/FID100-swiglu.json#L18), [FID100-isru_d.json:18](../../docs/celeba128-20261112/FID100-isru_d.json#L18), [comparison:24](../../docs/celeba128-20261112/FID100-comparison-verification.json#L24). These are the exact **persisted TensorBoard scalars**, not a claim to retain the estimator’s original double precision. |
| **D. Metric actually executed** | **PASS** | Execution calls `torch_fidelity.calculate_metrics` with generated images, the real reference dataset, `fid=True`, and no precomputed CelebA statistics: [engine_jit.py:167](../../engine_jit.py#L167). Installed package provenance points to pinned fork **bfe72b995473e4fee9351705d7b005f17442a4ca**. I inspected its executed FID statistics and Inception code and checked the [pinned source](https://raw.githubusercontent.com/LTH14/torch-fidelity/bfe72b995473e4fee9351705d7b005f17442a4ca/torch_fidelity/metric_fid.py). Fresh Inception weight SHA256 equals **6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2**. |
| **E. Scope of the conclusion** | **WARN** | This is one training trajectory per arm, matched in exposure and recipe, with different parameter counts and compute. It establishes the outcome of this candidate comparison, not a universal architecture ranking or a causal diagnosis. The report already limits its interpretation: [results.md:19](../../docs/celeba128-20261112/results.md#L19), [results.md:95](../../docs/celeba128-20261112/results.md#L95). |
| **F. Evaluation type** | **WARN** | This is **fixed-checkpoint generative evaluation against the training distribution**. It is not held-out generalization, a synthetic oracle-field test, or a convergence/FID trajectory. Training loss is the sampled velocity MSE defined at [denoiser.py:51](../../denoiser.py#L51). The periodic-evaluation fixture explicitly computes no actual FID: [periodic-eval-check.json:4](../../docs/celeba128-20261112/periodic-eval-check.json#L4). |

Additional bindings passed:

- **Checkpoint identity and state.** Fresh remote SHA256 values match the curated records: SwiGLU `317153d1954e2723214c310dd25e93897a5030358488a01d06ff0967ba03fab1`; K8 `289a4d7b43daca8c6aef076927e482ab0df49b5aaa3162498362a987012a5284`. Both checkpoints contain epoch index **99**, **63,500 optimizer steps**, finite model weights and both finite EMAs. Checkpoint arguments match the intended recipe. See [training-results.json:17](../../docs/celeba128-20261112/training-results.json#L17) and [training-results.json:105](../../docs/celeba128-20261112/training-results.json#L105).
- **Code identity.** The remote frozen runtime is clean at **65864adb8280675be76b6fc6b871b494b746d7d9**. All seven examined runtime file hashes match the corresponding frozen/current files except the deliberately updated active `main_jit.py`. Fresh AST comparison against pinned author source confirmed the eight loss/time/sampler/EMA methods and native SwiGLU definitions unchanged. Crop and learning-rate scheduler are byte-identical to the upstream originals. Attribution: [UPSTREAM.json:2](../../UPSTREAM.json#L2), [UPSTREAM.patch:1](../../UPSTREAM.patch#L1).
- **Evaluation identity.** The wrapper hash matches the launch receipt. It verifies checkpoint SHA256 before execution, passes the fixed recipe, and reconciles console/TensorBoard output: [celeba_eval.slurm:36](../../slurm/celeba_eval.slurm#L36), [celeba_eval.slurm:83](../../slurm/celeba_eval.slurm#L83), [celeba_eval.slurm:99](../../slurm/celeba_eval.slurm#L99). First EMA selection is explicit: [engine_jit.py:102](../../engine_jit.py#L102).
- **P history.** There are exactly **101** sequential files `epoch-0000.pth` through `epoch-0100.pth`; the final snapshot contains **24 operators**, all exactly matching the raw trained checkpoint. This does **not** make raw-P measurements equivalent to measurements of EMA P, input-dependent `PD(h)`, or the FFN Jacobian. Storage selects raw model parameters: [util/misc.py:230](../../util/misc.py#L230).

Primary remote console evidence, independently inspected:

- `/scratch/zz5070/ODE_FFN/runs/celeba/evaluations/20261113-b16-swiglu-s0-ckpt100-ema9999/evaluation.log:150` loads the checkpoint; `:152` states epoch100; `:154` switches to EMA; `:354` prints FID **35.9179**.
- `/scratch/zz5070/ODE_FFN/runs/celeba/evaluations/20261113-b16-isru_d-s0-ckpt100-ema9999/evaluation.log:153` loads the checkpoint; `:155` states epoch100; `:157` switches to EMA; `:357` prints FID **43.9077**.

**Concrete issues affecting the next decision**

1. **FAIL — a fresh follow-up launch cannot use the active wrapper unchanged.** It hardcodes the completed historical output directory at [celeba.slurm:39](../../slurm/celeba.slurm#L39), requires the old readiness receipt’s source commit to equal the executing HEAD at [celeba.slurm:44](../../slurm/celeba.slurm#L44), and rejects a changed source marker at [celeba.slurm:76](../../slurm/celeba.slurm#L76). These protections preserve the old run; they are not a new experiment launch configuration. **Minimum fix before any later authorized run:** distinct output paths, a pinned new source, and readiness evidence or an explicit compatibility validation bound to that source. Do not bypass the historical source guard.

2. **WARN — missing intermediate weights limit convergence claims.** Both original run directories contain only `checkpoint-last.pth`; checkpoint arguments confirm `online_eval=false`. P snapshots cannot reconstruct epochs20/40/60/80. The completed final comparison remains valid. **Minimum action:** keep the historical limitation explicit; require numbered full checkpoints in future runs. Do not infer missing FID points or rerun solely to relabel the history. See [periodic-fid.md:45](../../docs/celeba128-20261112/periodic-fid.md#L45).

3. **WARN — mechanism is unmeasured.** Raw-P spectra and finite weights do not establish Taylor truncation error, the behavior of `PD(h)`, or why K8 loses. The implementation fixes its entrance diagonal within each eight-term series but remains input-dependent across examples: [ffn.py:38](../../ffn.py#L38). **Minimum action:** keep any mechanism explanation labeled as a hypothesis and bind subsequent diagnostics to the exact raw/EMA state being discussed.

Historical `FID_available=false` in the training-collection snapshot and the inner `epoch0000` sample folder are **not invalid-measurement findings**. The first predates evaluation; the latter follows the frozen evaluation call’s logging argument while actual checkpoint loading is independently verified.

**Audit boundaries:** read-only inspection of `/Users/mr.z/ODE-FFN-upstream` and the stated Torch runtime/data/run paths. No files changed by this auditor; no training, sampling, deletion, submission, or report rebuild. Finalized reports and frozen runs remained untouched. I did not recompute FID from all retained images or decode all 100,000 PNGs. Semantic reviewer identity remains provisional: requested `gpt-6-astra / ultra`; actual returned model identity was unavailable.

<oai-mem-citation>
<citation_entries>
MEMORY.md:100-106|note=[Historical run context used only to require fresh verification]
</citation_entries>
<rollout_ids>
01a0c2bb-3714-7242-a294-be0d543caa4b
</rollout_ids>
</oai-mem-citation>

