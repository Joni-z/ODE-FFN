# Shared-controller screen

Computing one duration tensor and using it for both K1 ODE solves saves15.3% of the model's linear MACs and parameters relative to independent vector clocks. It does not retain the required Ring8 quality. The registered continuation criteria fail, so neither reused nor tied-controller variants are extended.

| Task | Independent vector | Tied weights, recomputed rates | Reused rate | Budget SwiGLU | Full SwiGLU |
|---|---:|---:|---:|---:|---:|
| gaussian_aniso_2d | 0.00377666 | 0.00377619 | 0.00374642 | 0.00430438 | 0.00392798 |
| gmm_ring8_2d | 0.01943755 | 0.02390319 | 0.02302461 | 0.01518420 | 0.01622070 |
| gaussian_aniso_16d | 0.00659516 | 0.00669685 | 0.00670319 | 0.00744039 | 0.00697032 |

The table is final5000-step random-time validation oracle MSE. Reuse has69282 parameters/67904 MACs at dim2; budget SwiGLU has69570/67904, so the narrower baseline controls the apparent saving. On Ring8, reuse derivative error is1.02533 versus.39370 for budget SwiGLU; sample SW2 is.24002 versus.23039. No practical benefit is established by reducing compute relative to a larger model while losing to its same-budget reference.

Tied weights retain the full evaluation count and reach.02390 Ring8 oracle MSE, also worse than independent vector.01944. This prevents attributing any reuse behavior solely to avoiding a second controller evaluation. The candidates retain the middle SiLU, original sigmoid temperature and P initialization. Their shared controller remains differentiable from both solves; tests verified the complete analytic Jacobian and a single actual controller call for reuse.

On this protocol the budget SwiGLU also beats the larger SwiGLU on Ring8, while its Gaussian controls worsen. This is a baseline observation, not a width-tuning result or a universal recommendation to narrow SwiGLU. All five arms and all tasks are retained.

The independent frozen-source audit passed90 checkpoints and75 common-field arrays, recomputed the oracle and random-time validation error, and verified non-gate/retained-controller initialization pairing. New checkpoints contain the actual Fourier-frequency buffer and restore it on resume/replay. Source snapshots and raw results are preserved under the stage identifier; no historical Report1 edits were made.

The next experiment is an unchanged-architecture test on product16. Its purpose is to assess directional clocks with higher-rank posterior variation, not to rescue a shared controller by adding a new trick. See protocol-product16.md and product16-mathematics.md.
