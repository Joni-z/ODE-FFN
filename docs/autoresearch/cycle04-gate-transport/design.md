# Proposed next test: exact gate transport with a square-zero matrix

Design review, not a completed experiment or an effectiveness claim. This is intended to test an actual computational change after cycle03's exact ordinary baseline, rather than declare an equivalent representation successful. Runnable source and a fixed acceptance protocol are still required before training.

## One added operator

Keep the ordinary activated gates `g=SiLU(Wg*h+bg)`, values `v=V*h+bv` and output `U`. Split the r gate channels once at m=floor(r/2). Learn only a rectangular C of shape m by (r-m), initialized to zero. Define

```text
P_C = [[0,C],[0,0]],  so P_C²=0.
g'=P_C*g;   u'=diag(v)*g;   v is held fixed.
g(0)=g0, u(0)=0; duration=1.
```

For the joint state (g,u), `K(v)=[[P_C,0],[diag(v),0]]` satisfies K³=0. Thus the professor-style degree-two exponential polynomial is exact:

```text
exp(K)=I+K+K²/2
u(1)=diag(v)*(I+P_C/2)*g0
F_C(h)=U[((I+P_C/2)*g0) elementwise v]+bo.
```

The efficient implementation only adds `C*g_B/2` to g_A before the existing product. There is one small dense multiplication, no repeated FFN, numerical integration, learnable duration, new activation, spectral penalty or approximation error. C=0 is exactly SwiGLU. The half coefficient comes from integrating a linearly evolving gate; it is not selected using data.

For M=I+P_C/2, the complete FFN Jacobian is

```text
JF_C = U[diag(v)*M*diag(SiLU'(a))*Wg + diag(M*g0)*V].
```

All gate/value derivatives and the unchanged surrounding normalization/conditioning must remain live. K's zero eigenvalues do not prove stability or the FM oracle's full-map properties.

## What computation changes

At a fixed gate dictionary g_j, ordinary SwiGLU can be expanded as `sum_j g_j(h)*A_j*h` (ignoring biases just for this rank statement), where A_j=u_j*v_j^T has rank at most one. The new expansion has

```text
A_j = sum_i M_ij*u_i*v_i^T.
```

A gate in the source half can therefore modulate several output/value directions without learning additional nonlinear gate features. This is the proposed computational purpose of the new matrix. It is not a lower bound for an entire learned multilayer network.

An exact two-dimensional witness uses U=V=I, g_j=SiLU(h_j) and C=[2]. Then `F_C=[h1*(g1+g2), h2*g2]`, so the coefficient associated with g2 is I, of rank two. With those same two gate features, ordinary SwiGLU has only a rank-one coefficient per gate. The four scalar products h1*g1, h2*g1, h1*g2 and h2*g2 are linearly independent on an open neighbourhood of the origin: setting one coordinate to zero removes the diagonal coefficients; dividing the remaining identity by h1*h2 leaves independent sigmoid(h1),sigmoid(h2). This proves the stated fixed-dictionary difference. Allowing new gates or more layers changes the comparison and must be faced experimentally.

## Distinction from stopped constructions

Stage20 changed scalar gate response functions through a forced coordinatewise ODE. The old coupled-generator proposal fed an already completed SwiGLU output back through U*diag(g)*V, producing gate-gate-value terms and extra FFN reevaluation. Here the new term is linear in the existing activated gate dictionary and linear in the value dictionary, and its augmented exponential terminates exactly. There is no old stopped scalar response, feedback recurrence, changed order or duration rescue.

The architecture is also exactly an ordinary partial linear gate mixer. That algebraically identical implementation cannot be an independent control or evidence of an ODE-specific causal advantage. The proposed scientific claim would concern a useful constrained architecture motivated by dynamics, not the benefit of attaching an ODE name to the same formula.

## Controls and prospective budget

At d=64 and nominal hidden128 (actual r=85), C has 42*43=1806 parameters/MACs per block. The resulting 16D network should have 89016 parameters and 87210 dense inference MACs. A nominal SwiGLU143 (actual hidden95) should have 89418 parameters and 87552 MACs, a close ordinary budget control. These predicted counts must be measured before launch. Reused SwiGLU128 and the stronger SwiGLU160 remain visible. An ordinary diagonal gate-rescaling control tests whether redundant per-channel reparameterization alone explains a gain.

The prospective screen is three fresh arms (transport, diagonal scaling, SwiGLU143) on Gaussian2D, Ring8, Product16, dependent signs and the half-mixture:15 cold seed0 trajectories,5000 updates each, with all10 relevant historical SwiGLU128/160 endpoints explicitly reused. Preserve the existing data and pair-FM training, exact time/batch streams, checkpoint/resume semantics, derivative panels and generation metrics. No partition, coefficient, norm cap, width or optimizer sweep is proposed.

A registered protocol should require one fixed transport setting to improve field MSE by at least10% against all ordinary controls on at least one16D law, stay within5% of the generic frontier on the remaining tasks, keep full Jacobian and SW2 within10% on nonlinear tasks, and verify matched cost. Any survivor still needs longer training, a frozen fresh holdout and runtime evidence. Stop failures without partition/rank/coefficient/feedback rescues. The user retains decisions about which direction to continue.

## Literature boundary

[Shazeer2020](https://arxiv.org/html/2002.05202v1) supplies the unchanged gated baseline. [Pearce et al.2024](https://arxiv.org/html/2410.08417v1), Sections2–3, analyzes bilinear MLP tensors without pointwise activation; it supports the importance of distinguishing interaction structure, not a performance prediction for our SiLU field. The [2026 rank-one note](https://arxiv.org/html/2606.22172v1) puts its nonlinearity in a different location in parts of its argument and provides no experiment establishing this candidate. It is not an empirical justification. The [2026 OQC preprint](https://arxiv.org/html/2604.09709v1) adds projection/complement machinery; none is adopted here. Broad bilinear or coupling ideas are established. No novelty claim follows from this review.
