# The cached-bank teacher differs sharply from the population teacher late in time

This read-only check answers a prerequisite for the proposed fitting/noise comparison. It sums over all100000 product16 training target atoms at each of128 fixed exposed validation anchors, at each of five times. No neural model or training objective was changed. The choice of anchors and measurements was recorded in [next-identification-question.md](next-identification-question.md) before computation.

| FM time | Bank-versus-population velocity MSE | Bank-versus-population Jacobian error | Median posterior ESS | Median largest atom weight |
|---|---:|---:|---:|---:|
| .1 | .0000337 | .000394 | 66199.8 | .000072 |
| .3 | .001035 | .092804 | 2179.0 | .003876 |
| .5 | .058299 | 13.118 | 22.09 | .132402 |
| .7 | .602738 | 324.232 | 1.56 | .785110 |
| .9 | 8.262924 | 31008.725 | 1.00000001 | .999999995 |

ESS=1/sum_i(weight_i²) measures the effective number of target atoms in the exact posterior at an anchor. At late times the Gaussian conditional width becomes small relative to the spacing of this high-dimensional finite bank. The empirical posterior can then select almost one training target, while the analytic population still describes a distribution of possible targets inside a mode.

The derivative makes the difference concrete. A single selected empirical atom gives v_emp=(Y_i-x)/(1-t), hence J_emp≈-I/(1-t). At t=.9 this is about−10I. In a resolved mode of the smooth population with variance.04, the coordinate derivative is instead approximately(.9*.04-.1)/(.1²+.9²*.04)=−1.5094. Both are symmetric conditional-mean fields. Symmetry or the covariance decomposition alone does not distinguish the desired population behavior from empirical memorization.

## What changes in the next experiment

Training samples Y from the finite bank, but scientific evaluation measures the known underlying population field. This is a generalization problem, and all architectures in the completed screens used the same bank and references. The teacher difference does not invalidate their comparisons or explain why ODE loses to SwiGLU. A neural model can learn reusable population structure rather than reproducing the exact empirical posterior, particularly on a factorized analytic population.

It does invalidate treating an analytic-population mean-label replacement at the same bank inputs as a pure conditional-noise ablation of the original training risk. The replacement changes the conditional mean being fitted. A future variance-isolation experiment must either use the bank's own exact teacher, or use the same fresh population input law for both pair-target and mean-target arms. A separate population-teacher fit can still test achievable function fitting, provided its changed target is stated explicitly.

The conditional label-noise covariance is Sigma=(I+(1-t)J_v*)/t for each law's own teacher. At t=.9 the population trace averages15.095 on these anchors, versus4.445 for the empirical bank; the latter mean includes a few non-collapsed posteriors. Neither a label-noise trace nor a teacher mismatch alone is a causal explanation of optimization. The next frozen-weight gradient diagnostic must also include the model parameter response.

## Verification and limits

The NumPy moment formula passed the single-atom limit and finite-difference derivative checks. An independent PyTorch direct-distance softmax/autograd calculation checks nine real-bank anchors across t=.5,.7,.9; maximum velocity/Jacobian discrepancies are4.19e-13/1.11e-11. The smallest computed covariance eigenvalue is−9.0e-13, recorded as floating-point cancellation and not silently clamped. The large teacher difference is far larger than that numerical error.

Full summary and audit are analysis-operator/empirical-target-summary.json and empirical-target-autograd-check.json. Raw moments, anchor arrays and hashes are outside Git beside the product16 evidence. The figure analysis-operator/figures/empirical-target-difference.pdf shows velocity/Jacobian differences and median ESS. These128 repeated validation anchors are descriptive probes, not an untouched holdout, independent training replications, or a theorem about all higher-dimensional datasets. This finding selects a diagnostic control requirement, not a new architecture or a successful completion of the goal.
