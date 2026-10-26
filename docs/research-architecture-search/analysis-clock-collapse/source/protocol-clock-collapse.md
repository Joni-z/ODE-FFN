# Finite contribution of the original clocks

Registered 2026-10-25 before this new checkpoint evaluation. This is one
read-only diagnostic, with no architecture, training, loss or optimizer change.
The previous goal turn completed the bias comparison and its independent
audit; that was progress, not achievement of the architecture objective.

## Fixed question and inputs

Does the trained state-dependent clock modulation improve the complete frozen
function over its own learned constant-clock core? Use only Stage23's final
homogeneous `vector_parent` checkpoints on Gaussian2D, Ring8 and Product16,
their existing configs and immutable 595-file runtime. Preserve all six
independent original K1 solves, their middle SiLU, backbone and runtime buffers.
The earlier anchored derivative diagnostic preserved every forward prediction;
the present intervention changes predictions and is a different question.

For each checkpoint, read only `xt[:256]` and `t[:256]` from the original
`probe.npz`, as the native trainer does. Restore the native float32 model and
record all six duration tensors during one unmodified forward pass. Fix the
per-channel mean `sbar[layer]` across these 256 probes. No targets, oracle
outputs, per-time fitting, interpolation, refitting or selected coefficient
enter these six vectors. Record their exact values, hashes and precision;
check their overall means against the native final layer logs.

On a separately restored float64 model, replace every clock by its fixed
`sbar`, converted from those float32 values. Recompute downstream states and
the complete input derivative. The resulting maps are

```text
F_i(h) = (I + diag(sbar_i) P_i) h
FFN(h) = B SiLU(A h).
```

External time conditioning remains present. This removes input/time dependence
from the clocks, not all time dependence from the network. An independent
implementation must fold the six fixed maps into these matrices and reproduce
the altered fields and Jacobians, rather than merely call the same clock patch.

## Complete fixed readout

Use every existing common validation panel: five times (.1,.3,.5,.7,.9),
512 inputs each, on all three tasks. Native float64 output/Jacobian replay must
first reproduce the saved arrays at the established 1e-10/1e-9 tolerances.
Bind checkpoint, config, data manifest/probe/validation, native arrays,
runtime source, script and protocol hashes. No fresh holdout, primary
uniform-time validation or sampling result is claimed.

For every panel save native/collapsed predictions and full Jacobians, their
oracle errors, finite changes and the exact risk polarizations:

```text
delta = prediction_collapsed - prediction_native
r = prediction_native - oracle
MSE_collapsed - MSE_native = 2 mean(r*delta) + mean(delta^2).
```

For Jacobians use the corresponding mean squared Frobenius norms, with sums
over both matrix axes before averaging inputs. Retain all fifteen panels and
three equal-time means, including each sign; do not select times or gate groups
after seeing outcomes. All means are read-only probe summaries, not population
expectations or estimates from unseen data. Show native/collapsed field and
Jacobian curves and error-alignment terms graphically.

## Interpretation and stop

Worse collapsed error demonstrates a useful finite contribution at those
checkpoints, not that a retrained constant-clock model cannot compensate or
that the controller computation is optimal. Similar or improved error weakens
the claim that the learned variation helps this frozen function; it does not
prove a function-class limit or a successful cheaper architecture. Large
output changes alone do not establish useful adaptation; the polarization
separates their magnitude from alignment with prediction error.

Raw duration variance, P spectra and their product bounds do not determine this
finite complete-model effect. The trained clocks are not certified to be near
constant. No result authorizes reopening stopped temperature, scale, order,
feature-sharing, bias, head, width or objective settings. The full useful ODE
architecture goal remains active and unachieved. Any later design needs its
own mechanism, budget controls, transfer and independent holdout evidence.
