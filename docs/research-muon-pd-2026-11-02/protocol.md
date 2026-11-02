# Muon updates and input-diagonal ODE generators

Registered before training on 2 November 2026. Seed 0. Hosts: 145 and Lenovo.

## Questions

1. Does changing the optimizer of P improve the original second-order ODEFFN? Does the spectrum of the **actual parameter update** change, and does P subsequently become better conditioned? Apply the same optimizer comparison to ordinary SwiGLU projections to distinguish a general optimization benefit from an ODE-specific benefit.
2. Does channel-wise input scaling help the original second-order layer, and does its placement matter? Historical first-order PD failed the 5k screen (Ring8 field MSE 0.02557; Product16 0.30168, both worse than SwiGLU). The present test changes the order, not that historical conclusion.

## Fixed operators

Use the archived width-64, three-block time-conditioned **vector-field MLP**, not an image DiT. Each original FFN contains two ODE maps with SiLU between them. At each map entrance compute the gate once; retain it throughout that map's polynomial.

```
scalar: A(h) = s(h) P
DP:     A(h) = D(h) P
PD:     A(h) = P D(h)
map:    h + A(h) h + A(h) A(h) h / 2
gate:   sigmoid(logit / 10) * 0.8 + 0.2
```

The second term in PD is P D P D h / 2. This is the truncated exponential of a generator frozen at the entrance state; it is not the second-order Taylor solution of a continuously recomputed state-dependent vector field. For the latter, derivatives of D would be required. Nevertheless, training and Jacobian diagnostics differentiate through the entrance gate.

Scalar gate hidden width 64; vector gate hidden width 32, as in the historical parameter-matched construction. DP and PD share every initial parameter and buffer. Vector and scalar share P and backbone initialization, with different gate output dimensions. All SwiGLU arms share the same non-FFN initialization as scalar ODE. SwiGLU nominal hidden widths 128 and 192 give actual widths 85 and 128 under the copied JiT 2/3 convention: approximately parameter-matched and projection-MAC-matched controls, respectively. Publish exact counts; do not call the controls identical in compute or parameter count.

## Two separate interventions

**Optimizer comparison:** scalar K2 and both SwiGLUs. Only ODE P matrices or SwiGLU FFN projection matrices change optimizer. All gates, biases, normalization, input/time embeddings and output heads use torch AdamW at 3e-4. Compare selected-matrix AdamW LR {3e-4, 1e-3} against official SingleDeviceMuon LR {0.006, 0.02}. This is a diagnostic of P/projection optimization, not a claim about the best all-hidden-matrix Muon recipe. Muon momentum 0.95 and five official BF16 Newton–Schulz iterations; AdamW betas (0.9,0.999), eps 1e-8; weight decay 0 for both. Global gradient clipping 1.0. The same two-rate search budget is given to both optimizers; report every arm.

**Structure comparison:** scalar K2, DP K2 and PD K2, all AdamW 3e-4. Do not combine vector gates with Muon in this first wave. No new activation, spectral penalty, normalization, or loss.

## Data, budget and pairing

- Ring8 (2D) on Lenovo; rotated bimodal Product16 on 145. Four optimizer/rate settings for three families, plus two vector structures: 14 arms per task.
- Anisotropic Gaussian (2D) on 145 is the simpler control: nominal AdamW 3e-4 and Muon 0.006 for three families, plus two vector structures: 8 arms.
- Total 36 trajectories. Each receives 20k updates, batch 512, model seed 0, constant LR. Final 20k endpoint is primary; 5k is a historical-budget diagnostic. No outcome-dependent early stopping. Stop a failing arm for nonfinite values or infrastructure errors and retain its failure record.
- Reuse fixed archived NPZ banks and their exact manifest/file hashes (`data_identity.json`). Gaussian/Ring8 data-generation seed 20260923 and Product16 data seed 0 describe the existing banks; neither is an additional model seed. Labels are never supplied.
- Independent standard-normal source and uniform time; copied linear conditional-FM objective. Each paired run on a task/device uses the same training RNG and batches. All arms of a task stay on the same device and host. CPU/GPU choice is made from a seeded throughput witness before formal training, never from quality results.

## Readouts and interpretation

Use the archived analytic marginal-field MSE and fixed-sample sliced W2 (4096 samples, 256 directions, 32 Heun steps / 64 NFE). Curves use validation targets. Final test sampling uses an independent fixed test source/target bank and is reported for **all** arms, never used for selecting a checkpoint. The reference-versus-reference sampling floor is always shown. Retain raw squared-error conventions. These vector data have no meaningful image FID.

At initialization and registered checkpoints, retain full P singular values/eigenvalues, stable rank, singular-value entropy, nonnormality and gate/activation statistics. During optimization additionally retain pre-step clipped gradients, Muon momentum, and the actual post-step Delta P spectra. Report full layer Jacobian norms at fixed hidden-state probes: P spectra alone do not bound an input-dependent layer's Jacobian. Log losses, gradient norms, elapsed time and resource usage. Spectral flattening by itself is not a quality gain.

Compare each ODE change with its exact original-ODE control and both ordinary controls, including their tested rates. A gain over original ODE alone is not sufficient for an FFN-advantage claim. Disagreement between field MSE and generation distance is reported rather than averaged away. Do not choose a new wave until this distinction is resolved.

## Source and execution

The active image trainer remains unchanged. `studies/muon_pd_20261102/reference/` copies four hashed modules and their five unchanged model dependencies from the audited archived source. Its single trainer patch injects an optimizer factory. The adapter uses the existing PD recurrence at K2 after verified construction; dense-polynomial and derivative tests guard this deliberately broadened path. Muon is copied unchanged from KellerJordan/Muon commit f98f1cacc0263b04290753e32be8d498c1efc806 with its MIT license. Upstream loss, sampling and checkpoint RNG/buffer handling are reused.

Before launch require native CPU/CUDA operator, optimizer and checkpoint-resume witnesses, file identity checks, and an independent agent following the documented invocation. Run snapshots are immutable commits. A persistent queue records PIDs, per-cell exits, steps and GPU resource samples. Never overwrite an existing run or automatically restart a failed cell.

Sources: [official Muon implementation](https://github.com/KellerJordan/Muon/tree/f98f1cacc0263b04290753e32be8d498c1efc806), [author explanation](https://kellerjordan.github.io/posts/muon/), [historical rate-order evidence](../research-architecture-search/analysis-rate-order/analysis-report.md).
