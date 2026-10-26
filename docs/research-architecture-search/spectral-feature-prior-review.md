# Fourier features after the state-clock screen

**Decision: NO architecture or training selection from the present evidence.** Oscillatory features are a genuinely different representation from real exponential gains, but the proposed rotation construction reproduces known learned Fourier features. Neither the completed experiments' middle-time error concentration nor the papers below identifies missing periodic features as the cause. Renaming FAN as an ODE block would not meet this project's objective. Stage21's final decision still requires its independent audit; this review neither accepts its raw results nor proposes a clock/solver rescue.

## Three primary sources, with different evidential scope

**1. FAN: Fourier Analysis Networks — NeurIPS2025 main conference.** The [proceedings record](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5fbdcd4d2190682b913c6b39b58e95f0-Abstract-Conference.html) verifies the venue. [Full text, Eq9 and Table1](https://arxiv.org/html/2410.02675v3#S2) defines

```text
FAN(h) = concat(cos(Wp h), sin(Wp h), activation(Wg h+bg)).
```

The same phase projection supplies sine and cosine, reducing projection work at fixed output width. Experiments include periodic function extrapolation, formula representation, forecasting, language and image recognition; they do not establish a DiT/FM FFN advantage over our SwiGLU reference. Its runtime table also shows that fewer nominal FLOPs need not improve small-layer latency. This is directly relevant prior art and generic representational evidence, not an FM-specific explanation.

The [official FANLayer code](https://github.com/YihongDong/FAN/blob/main/FANLayer.py) computes exactly those branches, with an optional phase bias enabled by default. Let input width be d, feature width m, phase count p and ordinary-activation count m−2p. Independently counting this code plus an m→d readout gives `d(2m−p)` dense MACs and `(m−p)(d+1)+md+d` parameters when both input biases and the readout bias are included. At p=m/4, the complete two-projection FFN saves 12.5% dense MACs against a same-feature-width ordinary MLP, not 25%; the latter concerns the input feature layer. SwiGLU with actual hidden width r costs `3dr`. Equal hidden widths therefore do not mean equal budgets. Trigonometric cost and latency remain additional.

**2. FANformer, published as “Reasoning is Periodicity? Improving Large Language Models Through Effective Periodicity Modeling” — NeurIPS2025 main conference.** [Venue/title](https://proceedings.nips.cc/paper_files/paper/2025/hash/e816e9f8901b57331e800fb79cd99dce-Abstract-Conference.html); [full text, Section3 and AppendixE](https://arxiv.org/html/2502.21309v3#S3). Its main change is `Attention(concat(cos(Wp X),sin(Wp X),Wg X+bg))`. **The FFN remains SwiGLU**, explicitly Eq8. It is not a successful Fourier-FFN replacement study. The paper's parameter-matched configuration narrows the FFN to pay for added attention-side projections. At p=.25 its leading extra cost is `1.5 L S D²` FLOPs, canceled by reduced FFN work in that configuration; elementary functions and runtime are not established equal by this accounting. The [released implementation](https://github.com/YihongDong/FANformer/blob/main/olmo/model.py) contains the attention-side FAN path and a separate SwiGLU activation, plus additional block variants. Its language-model results do not supply a diffusion-transfer result or an ODE mechanism.

**3. FourierFlow: Frequency-aware Flow Matching for Generative Turbulence Modeling — June2025 arXiv preprint; no accepted venue verified here.** [Full text, Eqs8–9 and AppendixG.3](https://arxiv.org/html/2506.00862v1#S3.SS2), [official code](https://github.com/AI4Science-WestlakeU/FourierFlow). Its Fourier branch applies a spatial transform, frequency-dependent learned mixing and an inverse transform: `F^{-1}(W(xi) F[u](xi))`, based on AFNO. The overall method also changes attention, branch fusion and MAE feature-alignment training. Its ablations concern turbulent spatial spectra and those components. This is direct FM literature, but spatial Fourier mixing is not pointwise sine/cosine activation of hidden channels. It gives no controlled evidence that the proposed FFN beats SwiGLU. Importing the complete recipe would change multiple axes and the objective.

## Exact ODE interpretation and its limits

For each input-dependent phase omega(h), let `A=[[0,-omega],[omega,0]]`. Solving `z'=Az` from `(1,0)` for unit auxiliary time gives `(cos(omega),sin(omega))`. With affine phases and a linear readout, this is precisely a learned Fourier feature map; adding a conventional activation branch yields FAN. The imaginary eigenvalues explain the conditional rotation, not architectural novelty.

For p phase pairs, the feature vector gamma has `||gamma(h)||²=p`, but direct differentiation gives

```text
J_gamma(h)^T J_gamma(h) = J_omega(h)^T J_omega(h).
```

Its input sensitivity is therefore controlled by phase derivatives, not conditional norm preservation. With affine phases it equals `Wp^T Wp`, generally not identity. Neither the readout nor the complete FM network inherits isometry, PSD Jacobians or stability. A zero-frequency initialization can also collapse cosine/sine feature directions; it is not automatically an initialization paired with ordinary SwiGLU.

## Why Gaussian smoothing does not select this feature family

For independent standard-Gaussian Z, target Y and `X_t=aZ+tY`, a=1−t,

```text
characteristic_Xt(w) = exp(-a²||w||²/2) characteristic_Y(tw).
v_t(x) = (x + a*grad(log p_t(x)))/t,   0<t<1.
```

The first identity concerns the **density's characteristic function**. The score divides its derivative by p_t; it is nonlinear in the density. No analogous Gaussian attenuation bound on the velocity's spatial spectrum follows. Nor does it identify the spectral content of learned hidden coordinates.

An instructive analytic counterexample is a one-dimensional target with equally weighted Gaussian components at ±mu and variance sigma². Set `b=a²+t² sigma²`. Its exact velocity is

```text
v_t(x) = ((t*sigma²-a)/b)*x + (a*mu/b)*tanh(t*mu*x/b).
```

This is affine plus a nonperiodic mode-assignment transition. Multiple modes and large middle-time errors alone do not imply a periodic representation deficit. Fourier features may approximate such transitions, but that is a generic approximation question, not evidence of superiority or a consequence of continuous-flow training.

## Consequence for the next decision

The missing link is evidence that a reproducible part of the actual residual is efficiently represented by oscillatory features and is not explained by generic added feature capacity. A future positive Fourier result would first belong to the known FAN/learned-feature family. An ODE-specific claim would require a substantive distinction from that algebraically equivalent baseline. This review selects no implementation, diagnostic sweep or training grid.

Search scope on 2026-10-25 included FAN/FANformer with diffusion, DiT, flow matching and SwiGLU; periodic/sinusoidal FFN activations; and FourierFlow. Full methods and relevant cost sections were inspected, and FAN/FANformer source formulas were checked. No direct matched DiT/FM Fourier-FFN-versus-SwiGLU result was verified in this bounded search; this is not a claim that none exists.
