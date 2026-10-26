# What the trained clocks contribute

Replacing the learned clocks by fixed channel means worsens velocity error on
all fifteen saved panels. The clocks therefore make a useful finite contribution
at these checkpoints. This rules out discarding their variation as negligible
while keeping the remaining weights fixed. It does not establish an efficient
architecture: the native ODE parent still loses to the established SwiGLU controls.

For each Stage23 homogeneous parent, we recorded six clock tensors during one
native float32 forward pass on the original 256 probe inputs. Their per-channel
means use no target or oracle values. We then replaced all six clocks by these
constants, preserving P, the middle SiLU, external time conditioning and every
other parameter. Downstream states and full input derivatives were recomputed
in float64. An independent implementation folded each FFN into `B SiLU(A h)`,
where `A = I + diag(sbar1) P1` and similarly for B, and reproduced the intervention.

The table averages the five fixed times equally. These are exposed validation
panels, not the primary uniform-time validation metric or a new holdout.

| Task | Native field MSE | Constant-clock MSE | Ratio | Native Jacobian error | Constant-clock error |
|---|---:|---:|---:|---:|---:|
| Gaussian2D | 0.003279 | 0.008788 | 2.68 | 0.016638 | 0.024716 |
| Ring8 | 0.016598 | 0.335110 | 20.19 | 0.828581 | 5.603272 |
| Product16 | 0.256488 | 0.723120 | 2.82 | 50.001937 | 67.407573 |

Jacobian error is the mean squared Frobenius error, without division by the
number of matrix entries. It increases on thirteen panels. The exceptions
are Ring8 at t=0.1 and 0.3; all times remain in the CSV and figures.

Ring8 at t=0.7 is the clearest case: field MSE rises from 0.036427 to 1.062077.
Writing `delta = collapsed - native`, the increase decomposes into
`2 mean((native-oracle)*delta) + mean(delta^2)`.
Here the alignment term is -0.020957, but the finite-change energy is 1.046608,
giving a net increase of 1.025650. The change partly opposes the original
residual, yet is much too large to improve the result. Output-change magnitude
alone would miss that distinction.

For Product16, equal-time field-error growth is 0.466633: alignment contributes
0.005640 and change energy contributes 0.460993. Its Jacobian alignment is
negative (-2.902251), but the change energy is 20.307887, so total derivative
error still increases by 17.405636. The complete finite intervention therefore
adds information beyond isolated P spectra or local gate derivatives.

This strengthens one constraint on the next design: it must preserve or replace
useful conditional computation, rather than assume the learned block is already
a constant two-projection MLP. It does not show that each clock is individually
necessary, distinguish input dependence from explicit time dependence, or prove
that a retrained constant-clock network cannot compensate. All clocks changed
together, and the existing backbone retains time conditioning. No timing,
sampling, fresh-label or retraining claim follows. No new architecture or
training run is selected; the useful-ODE objective remains open.

All fifteen native replays match the historical arrays exactly in the main
diagnostic. The separate folded replay agrees within 9.77e-15 for outputs and
3.07e-14 for input Jacobians; all eighteen risk rows agree exactly. See
[audit details](stats-appendix.md), [full data](clock-collapse.csv), and
[both figures](figure-catalog.md).
