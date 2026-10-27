# Nonlinear responses along a preactivation flow

## Decision and evidence

One bounded screen is justified after the mathematical fixtures pass. Cycle04 spent extra parameters recombining already activated features; the matched wider ordinary SwiGLU won on all three16D laws. This motivates testing a different computation: changing the affine directions **before** activation and integrating their nonlinear responses. It does not predict a gain. The screen retains the stronger generic controls and an ordinary midpoint reparameterization.

This is not a partition, coefficient, rank or feedback rescue of cycle04. Its learned operator and training budget stay fixed; activation moves inside a different vector field. It also differs from the stopped original nonlinear-drift field `D0 P2 SiLU(P1 z)`, the state-dependent original clock, and the forced `SiLU(a)*phi1(lambda*sigmoid(a))` scalar response. None of those controllers, rates, solver settings or output feedback are reused. Their failures lower the empirical prior for another ODE-themed change. No further node-count, activation, duration, initialization, partition or width rescue follows a failure here.

## Exact path, approximate accumulated response

Let `a=Wg*h+bg`, `v=V*h+bv`. Split the85 gate channels into42 receivers and43 sources. The only added parameter per block is a zero-initialized42-by43 matrix C. Define `P=[[0,C],[0,0]]`, so `P²=0`, and hold v fixed during auxiliary time s:

```text
a'=P*a;       a(0)=a0
u'=v*SiLU(a); u(0)=0
a(s)=a0+s*P*a0
F(h)=U*[v*integral_0^1 SiLU(a0+s*P*a0) ds]+bo.
```

There is no activation approximation in the a path. The u integral is generally non-elementary. Use only the fixed two-node Gauss-Legendre rule: `s±=.5±1/(2sqrt(3))`, weights1/2. Source coordinates do not move and their activations need only one evaluation. The ordinary midpoint control uses the same C and `SiLU((I+P/2)*a0)`.

## What changes functionally

Midpoint absorbs exactly into `Wg_eff=(I+P/2)Wg`, `bg_eff=(I+P/2)bg`. The matrix is invertible, with inverse `I-P/2`, so this is the ordinary SwiGLU function class with redundant optimization coordinates.

For a receiver write its centre c and displacement b. The exact mean is `SiLU(c)+b²*SiLU''(c)/24+O(b⁴)`. Two-node quadrature has the same quadratic term; midpoint omits it. A single-receiver witness with centre/direction inputs(x,y) has two-node Hessian at0 equal to `[[1/2,1/4],[1/4,1/6]]`, determinant1/48. A single affine SiLU ridge has Hessian rank at most one. This proves a local feature distinction. It is not a capacity lower bound against an entire wider SwiGLU or a multilayer network; two ordinary tied-feature channels can implement the same finite sum.

## Derivatives and the nilpotency caveat

With `A_k=I+s_k P` and `g=sum w_k SiLU(A_k a)`, the complete FFN input derivative is

```text
Jg = sum_k w_k diag(SiLU'(A_k a))*A_k*Wg
JF = U*[diag(v)*Jg + diag(g)*V].
```

For a batch loss with receiving derivative E, `grad_C=sum_k w_k*s_k*(E_left*SiLU'(a_left+s_k*C*a_right))^T*a_right`, where `E=(grad_output*U)*v`. Both gate and value paths remain live. At C=0, two-node and midpoint values, all common parameter/input derivatives and the C derivative agree, including a nonzero output-head probe. Their distinction starts at second order in C.

At fixed v the joint vector-field Jacobian is `J=[[P,0],[diag(v)*diag(SiLU'(a)),0]]`. Pointwise `J³=0`, but this is **not** a linear generator: `u'''=v*SiLU''(a)*(P*a)^2` is generally nonzero. Powers of a frozen Jacobian do not give the nonlinear flow Taylor series. No hidden nilpotency, invertibility or volume property is imported into the complete input-conditioned FM model.

## Numerical meaning

The componentwise two-node integration error is bounded by `|b|^4/4320 * sup_segment |SiLU''''|`. A conservative global bound is `|SiLU''''|<=1+2/e`: write it as `4 sigmoid'''+x sigmoid''''`, use `|sigmoid'''|<=1/4`, `|sigmoid''''|<=2 sigmoid(1-sigmoid)` and `|x| sigmoid(1-sigmoid)<=1/e`. Large learned displacements can invalidate a small-error assumption; this bound does not certify trained quality. The protocol requires a fixed32-node reference diagnostic and forbids increasing the training node count after outcomes.

## Primary literature and claim limits

- [Shazeer, GLU Variants Improve Transformer, Section2](https://arxiv.org/html/2002.05202v1): ordinary gated FFNs and fair three-matrix budgets. Our native biases and synthetic objective remain unchanged.
- [Biloš et al., Neural Flows, NeurIPS2021, Section2 and limitations](https://proceedings.nips.cc/paper_files/paper/2021/file/b21f9f98829dea9a48fd8aaddc1f159d-Paper.pdf): directly modeling flow maps is established; existence of an ODE representation is not itself a new useful inductive bias. Their flow conditions concern invertible maps, not our encoder/readout composition.
- [Norcliffe and Deisenroth, TMLR2023, Section3](https://arxiv.org/abs/2308.10644): Gauss-Legendre quadrature for adjoint integrals. They do not test this gated FFN, and supply no prediction of an FM improvement.
- [NIST DLMF3.5(v)](https://dlmf.nist.gov/3.5#v): classical Gauss quadrature and its remainder. The two nodes and error constant are standard numerical analysis.

Targeted searches for integrated SwiGLU and trajectory activations did not establish an exact prior match. They are not an exhaustive novelty audit. This is a controlled local architecture test, not a frontier-method claim, a new integration theorem or an FM-specific approximation guarantee.
