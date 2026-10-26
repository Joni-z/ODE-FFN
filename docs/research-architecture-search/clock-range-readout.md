# Clock range did not resolve the observed gap

The registered Stage8 screen is complete: five arms on Gaussian2D, Ring8 and product16, each trained for5000 updates. All initial ODE tensors, RNG draws and linear MACs match across the original, wider-positive and affine clocks. Their clock values/slopes agree locally at zero logit, not at every input. The frozen protocol is [protocol-clock-range.md](protocol-clock-range.md).

| Clock or reference | Gaussian oracle MSE | Ring8 oracle MSE | Product16 oracle MSE | Ring8 Jacobian error | Product16 Jacobian error |
|---|---:|---:|---:|---:|---:|
| Original vectorK1 | .00377666 | .01943755 | .29524142 | .91532 | 53.73663 |
| Wider positive | .00377382 | .01963767 | .29284939 | .97037 | 53.16720 |
| Affine | .00376918 | .01976813 | .30183454 | .98664 | 54.71981 |
| Small SwiGLU | .00430438 | .01518420 | .09083060 | .39370 | 22.59627 |
| Full SwiGLU | .00392798 | .01622070 | .10823417 | .57103 | 25.27259 |

Relative to the original clock, widening changes Ring8 oracle error by+1.03% and product16 by−.81%; the affine clock changes them by+1.70% and+2.24%. These do not approach the registered baseline-sensitive continuation rule. Product16 sampling SW2 is.11427/.11360/.11474 for the ODE arms versus.09822/.09462 for small/full SwiGLU. Neither relaxed arm advances. No K2, cap or controller enlargement is added to rescue it.

This is a5000-update screen. The saturation observation that motivated it came from20000-update models; the screen does not prove that saturation is harmless at every training stage. It does show that removing outer saturation is insufficient under this tested budget and initialization.

Remote frozen-source audits passed90 checkpoints and75 common-point diagnostic arrays, including recorded initial-tensor pairing, source/data hashes, oracles and validation MSE. The complete metadata/diagnostics and initial/final checkpoint archive is collected locally and SHA256-verified: `4115a940b7b777c37c47fa9e77dac51ccab431aa3947ecb2082b7b2e5b1b0faf`. Intermediate checkpoints remain on Lenovo. See collection.json for raw paths. Cross-platform endpoint replay, when recorded separately, must not be confused with this remote audit.

Figures in analysis-range/ show the full reference comparison and learning curves. The separate frozen-scalar response-space result uses the older20000-update checkpoint, not these endpoints: see [clock-response-bound.md](clock-response-bound.md). Its conditional floor of51–85% of current Jacobian error motivates examining response geometry, but cannot identify P alone as the cause. The vector model need not have the same response-space obstruction.

Cross-platform replay is now complete: the15 final checkpoints reproduce all75 saved common-validation velocity/Jacobian arrays on the Mac using frozen source and recorded runtime buffers. Maximum forward discrepancy is7.61e-15 and maximum derivative discrepancy3.20e-14. Both replay summaries are saved in analysis-range/. This does not regenerate sampling trajectories or create an untouched holdout.

The duration manipulation was actually used. On the product16 diagnostic probes, the affine arm spans−2.48 to5.07, with up to75.9% of entries in one layer outside the original(.2,1) interval; the wider-positive arm reaches up to69.1% outside. Ring8 affine durations span−.60 to2.39. These are extrema or maximum per-layer fractions across the recorded probes, not population-wide probabilities. Thus the near-identical quality is not explained by all new clocks remaining inside the original interval. Exact per-layer values are in analysis-range/duration-use.json.
