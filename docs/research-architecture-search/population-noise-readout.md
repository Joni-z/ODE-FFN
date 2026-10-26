# Conditional noise does not close the fixed-budget ODE gap

Exact conditional-mean supervision improves all three unchanged models, but the ODE remains worse than both SwiGLU sizes. Under mean supervision its error is2.43 times the better reference on Ring8 and14.42 times on product16. This meets the registered interpretation that conditional noise alone has not closed the fitting gap. It is not a proof of insufficient expressivity: the nonconvex optimization budget is still finite.

| Task | Model | Pair-label oracle MSE | Mean-label oracle MSE | Pair/mean error ratio |
|---|---|---:|---:|---:|
| Ring8 | VectorK1 ODE | .02326299 | .00224867 | 10.35 |
| Ring8 | Small SwiGLU | .01653752 | .00092464 | 17.89 |
| Ring8 | Full SwiGLU | .01720626 | .00120753 | 14.25 |
| Product16 | VectorK1 ODE | .27082130 | .18311354 | 1.48 |
| Product16 | Small SwiGLU | .08684151 | .01808341 | 4.80 |
| Product16 | Full SwiGLU | .08358915 | .01269912 | 6.58 |

The two label choices use exactly the same fresh target/source/time sequence within each task, including all models and every minibatch. Both use the analytic population specified by the existing manifest. In real arithmetic their squared risks differ by a parameter-independent conditional variance, so their expected gradients before clipping coincide. Mean labels remove conditional target noise, while retaining input sampling, the architecture, initialization, AdamW and clipping. The model receives no population parameters or extra inputs. The analytic teacher is privileged diagnostic supervision and cannot be accepted as the requested architecture.

The gap also persists in derivatives. Mean-supervised Ring8 Jacobian error is.11588 for ODE versus.05739/.07462 for small/full SwiGLU. Product16 is37.4685 versus7.8596/6.2196. Full endpoint rows retain sampling SW2 and costs; no selected checkpoint or Gaussian-only improvement substitutes for these failures.

## Clipping is observed, not changed

| Task | Model | Pair-label clipping fraction | Mean-label clipping fraction |
|---|---|---:|---:|
| Ring8 | VectorK1 ODE | .998 | 0 |
| Ring8 | Small SwiGLU | .982 | 0 |
| Ring8 | Full SwiGLU | .978 | 0 |
| Product16 | VectorK1 ODE | .384 | .156 |
| Product16 | Small SwiGLU | .014 | 0 |
| Product16 | Full SwiGLU | 0 | 0 |

These fractions cover the final500 updates. Adam's moments and clipping are nonlinear responses to the gradient distribution, so exact risk equivalence does not imply identical mean optimizer updates. The observations do not isolate clipping as a cause, nor justify changing it as an untested fix. In particular, the Ring8 mean-label gap remains while none of the three models clips during the final interval.

The earlier cached-bank runs used a different training input law and random-draw sequence. Their values are context, not the paired supervision comparison here. The large late-time empirical-bank/population teacher difference therefore was controlled by giving **both** new arms fresh population inputs, not by silently substituting an analytic teacher into the old bank objective.

## Verification and resulting direction

Frozen native audits regenerate all10000 minibatches across the two tasks and match every arm's checkpoint input fingerprints and generator states. Initial pair/mean tensors match. Source/data/configuration hashes,72 checkpoints and60 common-validation arrays pass. Raw initial/1000-step/final evidence is collected and SHA256-verified. Mac replay independently reloads all12 final checkpoints and reproduces all60 saved velocity/Jacobian arrays, with maximum forward/derivative discrepancies1.12e-14/3.06e-14. These checks do not regenerate sampling trajectories or create an independent holdout. Exact paths/hashes are in collection.json.

The next test stays with the original P and ordinary pair labels: an invertible rescaling of the last controller layer and temperature that preserves the initial function and full feasible function class. If a fixed-budget result changes under that transformation, expressivity has not changed; parameter coordinates and the optimizer's response have. An unpaired initial-temperature control separates that effect from changing the initial durations. The proof, numerical equivalence control and new frozen protocol are in clock-coordinate-mathematics.md and protocol-clock-coordinates.md. No new architecture has yet met the full completion requirements.
