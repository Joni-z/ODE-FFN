# Existing evidence: what each comparison answers

Reviewed at source commit `652087e`. This is a cross-stage reassessment, not a pooled leaderboard. Training and evaluation protocols differ across stages; numerical rows remain attached to their original comparisons. No new model was trained for this review.

## Evidence matrix

| Question | Actual comparison and observation | Supported conclusion | Missing link |
|---|---|---|---|
| Does a spectral cap improve the model? | K1/K2, three targets: cap lowers all six sampled mean layer-Jacobian norms; SW2 improves in three. Ring8 K2: norm 3.014→2.099, SW2 0.18482→0.18893. | The tested cap controls part of hidden sensitivity; it is not a reliable quality proxy. | No proof the remaining full-field error comes from excessive amplification. No singular-value equalization was tested. |
| Is increasing K an improvement? | Separately trained Ring8 K2/K4/K8 SW2: 0.18482/0.17518/0.17483. On Gaussian K1 is better than higher K at the same endpoint. | More terms can change performance, with no uniform benefit. | K changes compute and the learned optimum; this does not isolate numerical accuracy. |
| Does removing truncation error help a trained model? | Frozen K1/K2→exp worsens all six target/model cases. Ring8 K2 SW2 0.18482→0.38133; oracle MSE 0.02871→0.16944. | Low-order checkpoints are adapted to their polynomial computation. | A model trained with exp was not compared; no conclusion that exp is intrinsically worse. |
| Is the sampler mainly responsible? | Ring8 K2→exp at Heun256: 0.18613→0.38371. K4/K8 swap effects also persist. | These large intervention effects survive tested solver refinements. | Heun256 is a finite numerical reference, not an exact solution or a proof for every model. |
| Must s depend on hidden state? | Input vs time gates have mixed signs across targets and K. Latest K2 input vs fixed/scalar helps Ring8 but not consistently Gaussian/Moons. | Input dependence is a plausible source of useful nonlinearity in Ring8. | Time gates change input distribution and optimization; fixed gates remove ~25k parameters. Neither comparison cleanly isolates the derivative term's causal benefit. |
| Does learning a constant duration add capability? | Fixed/scalar nearly coincide; Ring8 oracle MSE 0.07462/0.07373 and 8192-sample SW2 0.14059/0.13965. | Observation agrees with exact scale absorption into free P. | Optimization differences are still possible; equal function class does not imply identical training. |
| Has ODE beaten the practical baselines? | Old GPU study: K2 beats same-update SwiGLU on all three; approximate MAC-matched SwiGLU is better on Gaussian/Ring8. Latest CPU study: SwiGLU has better Ring8 oracle MSE and cleaner modes. | There is no established general ODE advantage. | Architecture, sampling and endpoint sensitivity need a mechanism-specific comparison, not another best-score table. |
| Can one SW2 number identify the better mechanism? | Latest Ring8 input/SwiGLU: 2048 test 0.17288/0.15129; 8192 test 0.09167/0.09592; same generated sets against 8192 validation 0.12064/0.10563. | Fine rankings are sensitive to the empirical evaluation distribution. | These sample/target swaps do not change the learned models and are not training replications. |

Primary sources within this repository:

- [Original cap and K1/K2 analysis](../analysis-2026-10-13/analysis-report.md).
- [SwiGLU and full-field/time-bin diagnostics](../research-2026-10-19/results.md).
- [Gate source, order and MAC-budget study](../research-2026-10-20/stage-report.md).
- [Frozen-order interventions and solver refinement](../research-2026-10-20-followup/analysis-report.md), [exact intervention data](../research-2026-10-20-followup/exact-effects.json).
- [Latest gate study](../research-2026-10-21-gates/findings.md), [endpoint CSV](../research-2026-10-21-gates/results/results.csv), [larger-sample evidence](../research-2026-10-21-gates/results/sample-count-check.json).

## Two observations worth explaining, rather than turning into claims prematurely

**Ring8 gate benefit.** At the latest fixed endpoint, input/fixed/scalar oracle MSE is 0.03250/0.07462/0.07373. The fraction farther than 0.6 from every target center is 8.35%/19.87%/19.68%, while SwiGLU has 3.42% and the reference target sample 1.03%. This is more informative than a tiny SW2 lead: the simplified models leave mass between modes. It does not establish whether the original gate corrects an anisotropic field error, merely adds capacity, or changes optimization.

**Hidden sensitivity is not full-field sensitivity.** In the older Ring8 diagnostics, the cap lowers internal K2 layer norms, yet the full velocity field's mean maximum symmetric-Jacobian eigenvalue in t=0.4–0.5 is approximately 2.105 (original) versus 2.127 (cap). There is no direct propagation of the cap into proportional full-field contraction. In the latest gate study, input gates yield smaller P norms than fixed gates (1.837 vs 2.139) but larger layer-Jacobian norms (3.176 vs 2.703). These are local sampled means, not global bounds.

Existing diagnostics already contain oracle errors by time bin, full-field largest singular values, maximum symmetric eigenvalues and skew-part norms. They do **not** save a pointwise correspondence between hidden gate corrections and oracle posterior-covariance directions. Aggregate tables cannot reconstruct that correspondence. This is the missing measurement relevant to the proposed mechanism, not another global norm average.

## What the image archive can and cannot contribute now

The checked-out source has a significant naming distinction:

- [synthetic/models.py](../../synthetic/models.py) implements the professor's two input-gated ODELayers separated by SiLU.
- [src/ffn_blocks.py](../../src/ffn_blocks.py) contains an `ODEFFN` class, but the current registry maps the string `ode` to **ODEOnlyFFN**, whose default is K1, a residual delta, RMS normalization and an extra scalar output gate.
- [src/main_jit.py](../../src/main_jit.py) adds an embedding gate for `ode`; [src/model_jit.py](../../src/model_jit.py) passes the joint time/class condition. That path is not the professor's original input gate.
- The default K1 ODEOnly delta is `s*P*h`. Ignoring epsilon, its RMS normalization cancels positive s exactly. With the actual epsilon the cancellation is approximate and fails near tiny delta norms. This makes the historical variant even less interchangeable with the professor block.
- JiT predicts clean data and converts it into velocity; the toy model predicts velocity directly.

The [checkpoint inventory](../checkpoints.csv) documents historical image checkpoints, but filenames alone do not establish which historical source, effective configuration and evaluation protocol produced them. The [legacy overview](../legacy-overview.md) itself says quantitative aggregation is pending and incorrectly expands JiT as “Joint image Transformer”; the official name is **Just image Transformer**. It should be treated as an archival orientation document, not a scientific record of every run.

Torch authentication failed during this review. The 145 copy contains the synthetic evidence but not the historical image `results/` directory. Accordingly, this review audits the current image implementation and inventory, but **does not claim to have re-read every historical image log or reconstructed historical FID comparisons**. This does not block the synthetic/theoretical reassessment. Before using any image result as support, recover its execution-time source/configuration and evaluation settings.

## Claims to retire or narrow

- Retire “the exponential series may not converge unless P is constrained.” It converges for every finite P.
- Retire “higher K should generate better because FM is continuous.” The hidden polynomial and the external flow solve are different objects.
- Narrow “the gate helps” to the observed Ring8 comparisons, keeping the capacity confound and stronger SwiGLU evidence visible.
- Retire “flattening P's singular values should fit FM.” Neither the objective nor our evidence establishes this; our Gaussian target explicitly needs anisotropic behavior.
- Retire “small hidden Jacobian implies good generation.” The measured cap counterexamples and the full-network chain rule contradict that shortcut.
- Retire any direct merging of the historical image `ode` label with the professor-style synthetic ODEFFN without source verification.

The current evidence supports a research question about **structured input-dependent operators and optimization**, rather than a demonstrated architecture improvement or an established numerical-instability failure mode.
