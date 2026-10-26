# Is a sigmoid-GLU control justified?

**Conditional GO for one ordinary sigmoid-GLU baseline comparison; NO new ODE architecture is selected.** Its purpose is to test a concrete generic alternative before attributing future probability-shaped features to an ODE construction. This is useful only if both outcomes constrain the next decision: a gain strengthens the generic baseline; failure stops this particular gate substitution. It is not permission for an activation sweep or for reopening the stopped phi1, rotation, state-clock or spectral-constraint families.

## What the primary literature actually establishes

**Shazeer, “GLU Variants Improve Transformer” (2020 technical paper).** [Full text, Eqs4–6 and Table1](https://arxiv.org/html/2002.05202v1#S2) directly defines sigmoid-GLU and SwiGLU and compares them in T5 with matched parameter/operation counts. Its long-run held-out log-perplexity is 1.663 for GLU versus 1.636 for SwiGLU; lower is better. This is evidence against assuming the original sigmoid gate is generally superior. It establishes the exact generic control and prior art, not a result about continuous Gaussian flow matching. The paper's text-denoising objective is not the FM velocity objective. Its experiments omit biases, whereas our existing SwiGLU has them; preserving those biases is part of the proposed controlled substitution.

**Kaleido, “Scaling Sequence-to-Sequence Generative Neural Rendering” (ICLR2026).** [Official proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6b055b95d689b1f704d8f92191cdb788-Abstract-Conference.html); [full text, Section3.2.1 and AppendixA](https://arxiv.org/html/2510.04236v3#S3.SS2.SSS1). This is a direct rectified-flow Transformer study. It describes replacing GLU with SwiGLU, but the reported architectural comparison simultaneously replaces MHA with GQA; the table labels the combined intervention. Equal reported throughput does not isolate the activation effect. The reviewed text also does not supply the original gate formula, so its “GLU” terminology alone cannot establish an exact sigmoid-GLU comparator. It cannot answer our one-factor question.

**Krea2 Technical Report (2026, official research report).** [Architecture section](https://www.krea.ai/blog/krea-2-technical-report) describes GELU-MLP→SwiGLU in its diffusion Transformer and reports consistent gains. Its sigmoid gates belong to a separate attention modification. This is current primary diffusion-architecture evidence favoring retaining a strong SwiGLU reference; it is not a sigmoid-GLU-versus-SwiGLU FFN comparison. The reported expansion-factor description does not by itself establish our exact matched-parameter/MAC comparison.

Searches combining DiT/diffusion/flow matching with sigmoid, tanh, GLU, SwiGLU and ablations did not verify an isolated, matched-cost diffusion/FM comparison of the proposed gates. That is a bounded search result, not proof of absence. The old exact comparison is cited for prior art, not offered as a frontier reading recommendation.

## The question is narrower than “posterior-aware architecture”

Using the same independent gate, value and output projections,

```text
a = Wg h+bg; value = Wv h+bv
GLU(h)    = U[sigmoid(a)*value]+bo
SwiGLU(h) = U[a*sigmoid(a)*value]+bo.
```

The intervention removes the extra factor a. With actual hidden width r and model width d, both have `3dr` dense MACs and `3dr+2r+d` parameters, including all three biases. GLU saves one pointwise multiplication per hidden channel; runtime must still be measured. No new clock, generator, mixture atoms or dense matrix is needed.

The mathematical motivation is specific. For a balanced equal-variance two-Gaussian target, at each fixed interior FM time, the ideal velocity is affine plus a scaled tanh of a linear projection. Since `tanh(u)=2*sigmoid(2u)-1`, a biased GLU can supply these known scalar features using constant value channels alongside linear terms. This is a raw-input, fixed-time representational observation. It does not prove the same economical realization through our normalized/time-modulated hidden backbone, a single finite parameterization for every time, learnability under pair supervision, or a separation from a sufficiently capable SwiGLU.

General multimodal responsibilities are softmax probabilities, not independent sigmoid gates; unequal component covariances also introduce quadratic log-odds. Unconstrained GLU channels do not sum to one, so they are not automatically posterior probabilities or a convex mixture. A learned GLU remains a generic gated FFN.

Existing [posterior-region evidence](posterior-regions-readout.md) identifies errors near ambiguous signs **and** within resolved, contracting modes. It does not locate the whole gap in responsibility transitions. A gate substitution also changes initial feature amplitude, nonlinear shape and optimization. Identical tensors/RNG therefore do not give identical hidden functions or NTKs. A positive result would not distinguish all those explanations or prove that learned gates estimate the posterior.

## Conditions and decision rules for a useful control

1. Fix **one existing width** before outcomes, with an exact same-width SwiGLU counterpart. Retain independent gate/value projections, biases, backbone/time conditioning, optimizer, pair-supervised loss and seed0. Keep the already stronger/larger SwiGLU results visible. Do not provide target component labels, oracle weights/rotation, analytic initialization or exact-mean training targets.
2. State the corrected prediction before training: removing the gate's extra a factor may improve the affine-plus-signed-offset response, including resolved coordinates. The [registered regional readout](posterior-velocity-regions-readout.md) places78.26% of the saved Product16 ODE–SwiGLU128 velocity gap in the lowest-entropy bin, and93.71% at t=.7; it refutes an ambiguous-boundary-only rationale without identifying which response component caused the error. Retain all fixed time/entropy bins and overall fitting/sampling measures; validation oracle MSE remains primary. Gate resemblance to a posterior is not itself success. *Updated after the registered regional readout, before GLU training.*
3. Predeclare a practical improvement threshold and other-task/Jacobian/SW2 guards. A small or isolated panel gain does not warrant further design. A useful fixed-width gain establishes a **generic GLU baseline**, still requiring fresh evaluation before generalization claims. It does not complete the ODE architecture objective, and it cannot be advertised as beating a wider SwiGLU unless that comparison actually passes.
4. Failure closes this exact gate question without centering, temperature, slope, width or extra-response rescues. Success requires any later probability/simplex-based ODE design to beat both ordinary GLU and SwiGLU at the claimed budget. An augmented-matrix or differential-equation rewriting of GLU supplies no additional architectural contribution.

The rationale for this one control is eliminating a foreseeable baseline confound. The literature does not predict that it will win, and this review supplies no implementation or selected training grid.
