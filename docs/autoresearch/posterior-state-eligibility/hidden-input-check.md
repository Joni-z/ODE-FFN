# The actual FFN input is not an additive-Gaussian observation

Zhizhe Zhang · October 28, 2026. Application check after the data-free identities.

The mixture derivation assumes a particular observation law. It cannot be
transferred to JiT's hidden token merely because the original image input
contains Gaussian noise. The current FFN input is normalized and modulated.

## Source identity

The sealed native image builder replaces the FFNs inside `model.blocks`;
it preserves the surrounding normalization/modulation. Its actual frozen
`source/jit/model_jit.py` and `source/jit/util/model_util.py` match the current
[block](../../../src/model_jit.py) and [RMSNorm](../../../src/util/model_util.py)
byte-for-byte and match their native binding entries. This is a targeted
source check, not a new model evaluation or full parent-data rehash.

`JiTBlock.forward` passes `modulate(norm2(x),shift_mlp,scale_mlp)` to the FFN.
The actual norm is **RMSNorm**, with learned elementwise weight and eps1e-6;
it is not LayerNorm. The time/class modulation is fixed when its external
conditioning is fixed. The native image study has one fixed class input.

## A bound from the actual operator

Let x be a d-vector, a the finite norm weight, s/b the fixed modulation
scale/shift, and eps>0. In ideal real arithmetic the FFN input is

```text
h = b + [a*(1+s)] * x / sqrt(||x||²/d+eps)
||h-b|| <= sqrt(d) * max_i |a_i*(1+s_i)|.
```

The inequality uses the diagonal operator norm and
`||x||/sqrt(||x||²/d+eps) <= sqrt(d)`. It holds for every x, independently
of the attention distribution or prior. Thus the fixed-conditioning FFN
input has bounded support. The same argument concerns the whole normalized
token, not a measured local Gaussian approximation.

An observation `H=H_clean+sigma*e` with independent nondegenerate Gaussian e
and sigma>0 has unbounded support. It therefore cannot exactly equal this
bounded hidden-input law. Adding a time-only scale or whitening matrix does
not supply the omitted observation model. The original clean image Z_t
still has its declared Gaussian corruption; this argument does not dispute it.

## Consequence

Calling component states exact hidden posterior statistics would require an
additional model or approximation. This check does not prove that Gaussian
kernels, attention, posterior-inspired parameter tying, or normalization are
unhelpful empirically. It only blocks using the input-image Bayes identity as
an exact justification for this normalized FFN's internal dynamics.

No norm bypass, new input skip, teacher/oracle label, loss change or training
is selected. A next design must state its hidden-state role and its proposed
inductive bias without silently assuming this incompatible observation law.
