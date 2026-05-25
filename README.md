# FFN Block Designs for Flow Matching in JiT

A systematic exploration of feed-forward network (FFN) block redesigns within a JiT (Joint image Transformer) architecture for class-conditional image generation via flow matching. The central question: can FFN blocks be structurally aligned with flow matching dynamics, and does this alignment improve generation quality?

**Status:** Phase paused. Core experiments complete; full quantitative benchmarking pending.

---

## Overview

Standard transformer FFN blocks treat feature transformation as a static, time-agnostic mapping. This project investigates whether making FFNs aware of the diffusion timestep — through dynamical system formulations, functional decomposition, or semantic redefinition — improves a flow matching generative model.

The base architecture is a JiT-like transformer trained on **ImageNet 256×256** (class-conditional, 1000 classes) using **flow matching / rectified flow** as the generative objective. Each transformer block applies AdaLN conditioning and a residual gate:

```
x' = x + g_attn(c) · Attn(AdaLN₁(x, c))
y  = x' + g_ffn(c) · FFN(AdaLN₂(x', c), c)
```

where `c = t_emb + y_emb` is the shared condition vector (timestep + class). All FFN variants output a token-wise feature update; the outer residual and gate are handled uniformly by the JiT block.

---

## Repository Structure

```
jit-codebase/
├── src/
│   ├── main_jit.py            # Entry point: config loading, training orchestration
│   ├── model_jit.py           # JiT model (BottleneckPatchEmbed, JiTBlock, JiT)
│   ├── ffn_blocks.py          # All FFN implementations (~1700 lines, 21+ variants)
│   ├── ffn_factory.py         # FFN registry and factory
│   ├── engine_jit.py          # Training loop, EMA, FID evaluation
│   ├── denoiser.py            # Flow matching wrapper, Lipschitz regularization
│   ├── perceptual_loss.py     # VGG perceptual loss
│   ├── time_condition.py      # Time embeddings and scalar resolution
│   └── util/
│       ├── model_util.py      # VisionRoPE, sincos positional embeddings, RMSNorm
│       ├── compile_control.py # torch.compile toggle utilities
│       ├── fid.py             # FID score computation
│       ├── lr_sched.py        # Learning rate schedulers
│       ├── misc.py            # MetricLogger, SmoothedValue
│       └── prefetch.py        # CUDA data prefetcher
├── configs/
│   ├── ffn/                   # Per-variant YAML configs (JiT-B/16)
│   └── Large/                 # JiT-L/16 configs
├── tests/                     # Unit and integration tests
├── Designs/                   # Design rationale documents
├── results/                   # Saved checkpoints (last.pt per experiment)
├── fid_stats/                 # Pre-computed FID reference statistics
├── run.sh                     # Local single-GPU launcher
└── train.slurm                # SLURM job script (1× H100, bf16)
```

---

## FFN Variants

All 21+ variants are registered in `FFN_REGISTRY` (see [src/ffn_factory.py](src/ffn_factory.py)) and selected via the `ffn_type` field in the YAML config.

### Tier 1 — Static Baselines

| Key | Class | Description |
|-----|-------|-------------|
| `mlp` | `MLP` | Standard two-layer GELU FFN. No time conditioning. |
| `swiglu` | `SwiGLUFFN` | GLU-gated variant with SiLU gate. Stronger baseline; still time-agnostic. |

### Tier 2 — ODE-Dynamical FFNs

These variants treat the FFN as a finite-step approximation to a continuous dynamical system. The core primitive is `ODELayer`, which computes a truncated matrix-exponential propagation:

```
z₀ = x
z_{k+1} = (s / (k+1)) · P · z_k
output = Σ_{k=0}^{K} z_k   ≈ e^{sP} x
```

where `s` is a condition-derived scalar step size and `P` is a learned linear operator.

| Key | Class | What changes |
|-----|-------|-------------|
| `ode` | `ODEOnlyFFN` | Pure ODE residual; replaces the two-projection structure entirely. |
| `ode_swiglu` | `ODESwiGLUFFN` | SwiGLU base path + ODE correction term gated by a learned scalar. |
| `mh_ode_swiglu` | `MultiHeadODESwiGLUFFN` | Multi-head ODE: each head evolves under its own dynamics matrix. |
| `headwise_ode_value_glu` | `HeadwiseODEValueGLU` | ODE applied only to the value branch; gate branch stays static. |
| `lowrank_state_ode` | `LowRankStateODEFFN` | State-dependent ODE: `A(x) = diag(d(x)) + U·(Vᵀ·r(x))`. Dynamics depend on the input token. |
| `tied_flow` | `TiedFlowFFN` | Multi-step Euler integration inside the FFN with a shared vector field and per-step learned step sizes. |

### Tier 3 — Explicit Functional Decomposition

These variants hypothesize that a single FFN cannot optimally serve all noise levels, and explicitly split into specialized branches.

| Key | Class | Decomposition |
|-----|-------|--------------|
| `nav_refine` | `NavRefineFFN` | Navigation branch (head-wise ODE) + refinement branch (SwiGLU), mixed by condition-derived gates. |
| `time_split` | `TimeSplitFFN` | Coarse path (global-context-aware MLP) + fine path (SwiGLU), routed by a learned gate from pooled features and condition. |
| `freq_split` | `FrequencySplitFFN` | Low-frequency path + high-frequency path via local-pooling approximation; tracks `gate_mean`, `low_energy_mean`, `high_energy_mean`. |
| `freq_split_dual` | `FrequencySplitDualFFN` | Dual-branch variant of frequency split. |
| `time_moe` | `TimeMoEFFN` | Three-expert soft MoE: coarse expert (global pooling), balanced expert (SwiGLU), refined expert (SwiGLU on residual-mean). Softmax-routed by condition. |
| `deepseek_moe` | `DeepSeekMoEFFN` | Load-balanced sparse MoE following the DeepSeekMoE design. |

### Tier 4 — Semantic Redefinition

These variants change *what the FFN predicts*, not just *how it computes*.

| Key | Class | Redefinition |
|-----|-------|-------------|
| `clean_target` | `CleanTargetFFN` | FFN predicts a denoised latent `x̂₀`, then converts it to a velocity-like update `(x̂₀ − x) / (1 − t_frac)`. RMS clipping prevents blow-up near `t → 1`. |

### Additional Experimental Variants

| Key | Description |
|-----|-------------|
| `ta_gate` | Timestep-adaptive gating (SwiGLU with time-modulated gate strength). |
| `flow_evolved_gate` | Gate controlled by a small flow-dynamics sub-network. |
| `spatial_adaptive` | Injects spatial position features into the value branch. |
| `progressive_refine` | Progressive multi-stage refinement path. |
| `multistep_ffn` | Explicit Euler integration with multiple learned steps. |
| `adaptive_patch_perceptual` | Patch-level adaptation with perceptual loss coupling. |
| `soft_lipschitz` | SwiGLU baseline augmented with a directional Lipschitz regularizer (λ=1e-4). |

---

## Training Setup

**Model sizes**

| Variant | Hidden dim | Heads | Layers |
|---------|-----------|-------|--------|
| JiT-B/16 | 768 | 12 | 12 |
| JiT-L/16 | 1024 | 16 | 24 |

**Default training config (JiT-B/16)**

| Hyperparameter | Value |
|---------------|-------|
| Epochs | 80 |
| Batch size per device | 256 |
| Base learning rate | 1e-4 |
| LR schedule | constant (with warmup) |
| Optimizer | AdamW (β₂=0.99) |
| EMA decay | 0.9999 |
| Mixed precision | bf16 |
| Sampling method | Heun ODE solver |
| Sampling steps | 50 |
| CFG scale | 2.9 |
| FID evaluation images | 5000 |

**Flow matching parameters**

| Parameter | Value |
|-----------|-------|
| `P_mean` | −0.8 |
| `P_std` | 0.8 |
| `t_eps` | 0.05 |
| `label_drop_prob` | 0.1 |

**Infrastructure**

- Distributed training via HuggingFace `accelerate`
- SLURM: 1 node, 1× H100 80GB, 100 GB RAM, 24h walltime
- Dataset: ImageNet 256×256 from `/projects/work/public/ml-datasets/imagenet`
- Experiment tracking: Weights & Biases (project `jit-training`)

---

## Completed Experiments

Results (checkpoints) are stored under `results/`. The following runs have saved `last.pt`:

**JiT-B/16 experiments**
- `jit_b16_in256_imagenet_mlp`
- `jit_b16_in256_imagenet_ode`
- `jit_b16_in256_imagenet_ode_swiglu`
- `jit_b16_in256_imagenet_mh_ode_swiglu`
- `jit_b16_in256_imagenet_headwise_ode_value_glu`
- `jit_b16_in256_imagenet_lowrank_state_ode`
- `jit_b16_in256_imagenet_tied_flow`
- `jit_b16_in256_imagenet_nav_refine`
- `jit_b16_in256_imagenet_time_split`
- `jit_b16_in256_imagenet_clean_target`
- `jit_b16_in256_imagenet_time_moe`
- `freq_split`, `freq_split_dual`
- `spatial_adaptive`, `flow_evolved_gate`, `ta_gate`
- `progressive_refine`, `multistep_ffn`
- `deepseek_moe`
- `adaptive_patch_perceptual`
- `swiglu_soft_lipschitz`, `swiglu_soft_lipschitz_small`
- Architecture probes: `adaln_lora`, `adaln_lora_deep`, `adaln_single`
- Attention probes: `timestep_aware_attention`, `time_dependent_qkv`, `topology_rewire`, `unet_skip`

**JiT-L/16 experiments**
- `jit_l16_in256_imagenet` (SwiGLU baseline)
- `jit_l16_in256_imagenet_ode`

Quantitative FID results and head-to-head comparison tables are pending aggregation.

---

## Quickstart

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Run a single-GPU training job (local)**

```bash
CONFIG=swiglu bash run.sh
```

**Submit a SLURM job**

```bash
CONFIG=ode_swiglu sbatch train.slurm
```

**Launch with accelerate directly**

```bash
accelerate launch src/main_jit.py --config configs/ffn/ode_swiglu.yaml
```

**Add a new FFN variant**

1. Implement in [src/ffn_blocks.py](src/ffn_blocks.py), subclassing `BaseFFN`.
2. Register in [src/ffn_factory.py](src/ffn_factory.py): add an entry to `FFN_REGISTRY`.
3. Create a YAML config in `configs/ffn/` with `model.ffn_type: your_key`.

---

## Design Rationale

Detailed design documents are in [Designs/](Designs/):

- **[frequency_split_branch.md](Designs/frequency_split_branch.md)** — Motivation and implementation notes for frequency-aware FFN decomposition using local pooling as a spectral proxy.
- **[soft_lipschitz_experiment.md](Designs/soft_lipschitz_experiment.md)** — Hypothesis and setup for the Lipschitz smoothness regularization probe (directional finite-difference approximation, λ=1e-4).

---

## Key Design Dimensions

The FFN variants span four structural dimensions:

| Dimension | Static end | Dynamic end |
|-----------|-----------|------------|
| Time awareness | MLP, SwiGLU | ODE variants, clean_target |
| Granularity | Single path | Multi-head, low-rank state-dependent |
| Functional specialization | Unified FFN | nav_refine, time_split, time_moe |
| Prediction target | Feature update | clean_target (denoised latent → velocity) |

---

## Dependencies

```
torch >= 2.0.0
torchvision
accelerate
wandb
einops
scipy
Pillow
tqdm
PyYAML
pytorch-fid
```

---

## Citation / Reference

This codebase builds on the JiT architecture and is trained following the SiT / Lightning-DiT training setup. Flow matching objective follows the rectified flow formulation.
