# Frozen central-symmetry diagnostic

Registered before computation, following unchanged Heun32/64/128 ordering and
the exact quadratic-feature interpretation. This is a read-only diagnostic,
not an inference projection, new loss or architecture selection.

Both current population targets and the independent Gaussian source are centrally
symmetric. Under simultaneous negation of the FM pair, x_t and x1-x0 both change
sign; consequently the exact marginal velocity satisfies u(-x,t)=-u(x,t).
Verify the archived analytic oracle numerically before interpreting predictions.
The finite cached training bank need not be exactly symmetric.

For any predictor define v_o(x)=(v(x)-v(-x))/2 and v_e(x)=(v(x)+v(-x))/2.
For each frozen validation point with oracle u(x), the following is exact:

```
.5*(||v(x)-u(x)||^2 + ||v(-x)+u(x)||^2)
    = ||v_o(x)-u(x)||^2 + ||v_e(x)||^2.
```

On the symmetric population, the even part contributes only approximation error.
The empirical paired diagnostic measures the same identity on x and -x; it does
not replace the original unpaired validation score. An odd-projected model is
not trained, selected or sampled here. Hidden-state parity need not equal full-
predictor parity; the two-solve SiLU architecture can synthesize odd responses.

Freeze final20k checkpoints from six families per task, chosen by existing
validation within family: scalarK2, scalarK1, rawK2, rawK1, Swi64, Swi128. Choices:
Ring8 Muon.006/.006/.02/.02/.02/.02; Product16 Muon.02/.02/.006/.006,
AdamW.001/.0003 respectively. No new sampling or checkpoint choice.

Use the original fixed validation x/t bank and its negatives with unchanged
native float32 inference. Accumulate paired total, odd error and even energy in
float64; report the exact decomposition residual, oracle symmetry residual,
per-time-bin profiles and checkpoint hashes. If even energy is small relative
to total error or similarly sized in ordinary controls, stop treating parity as
an explanation for the raw-diagonal deficit. A large difference is a diagnostic
lead, not proof that enforcing symmetry improves training or sampled quality.

Run after the registered cycle4 training releases the GPUs. No new data cache,
package, benchmark metric or changes to existing checkpoints.
