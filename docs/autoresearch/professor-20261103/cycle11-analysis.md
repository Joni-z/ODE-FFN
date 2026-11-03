# Cycle11 analysis and interpretation

The numerical runtime is frozene493b05. Later analysis never replaces the native
loss, validation oracle, Heun sampler or stored test arrays. Finish all ten cells,
replay every20k endpoint and verify four exact native reference states/samples
before interpretation. The external ledger must append each new ID only once.

The factorial table has24corners: eighteen source-bound archived rows plus six
new cells. Verify every archived row field against its source CSV, all source
SHA256 values and both parent checkpoint bindings for each new candidate.
Native repeats validate compatibility; they are not additional independent
replicates. Keep all three optimizer/rate settings on both tasks.

The log-MSE interaction is

```
log(L_linear_affine/L_MLP_affine) - log(L_linear_sigmoid/L_MLP_sigmoid).
```

Negative means removing the controller nonlinearity is relatively more favorable
with affine output, under these fixed recipes. It is not a significance test or
a decomposition of the complete predictor's causal loss. Sigmoid versus affine
also changes boundedness, signs and optimization/gauge freedom, so this test
cannot attribute a gain solely to avoiding saturation.

The rank32 control concerns the logit/input Jacobian, not rank of the diagonal
matrix D or of P D, which may remain64. It fixes tensor budget/factorization,
not the full function class. In a local proof, E=P diag(delta_d)=O(||h||^2), and

```
Q2(A+E)-Q2(A) = E + (AE+EA+E^2)/2.
```

Multiplication by h gives an O(||h||^3) layer difference, without assuming
A and E commute. This proves zero-anchor agreement through quadratic order
in real arithmetic. Native float32 folding introduces the explicitly measured
rounding discrepancies; hidden states need not remain near that anchor.

Use factorial.py only after generic endpoint/source/anchor curation. It retains
all four corners, log interactions, separate test scores, the full fresh20k
SwiGLU frontier and every new learning curve. Its continuation flag implements
the registered two-parent5% and strongest-ordinary validation rule. A generic
provisional keep alone is insufficient. Any sampling loss remains a tradeoff.
No stronger claim follows from P shape, gate-range use or CPU/CUDA readiness.
