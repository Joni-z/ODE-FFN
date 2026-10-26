# Next question: is the original bounded positive clock a harmful restriction?

Status: tested as Stage8. Both relaxed clocks failed the registered5000-update screen and are stopped; see clock-range-readout.md. The following records the mathematical question before that test, not an established architecture. The failed shared-clock and nonlinear-drift screens remain stopped. Return to the originalK1 ODELayer for this question and change only its duration transfer function if the experiment is registered.

## Evidence

The product16 native-clock diagnostic uses1024 common validation anchors at five times. In the scalar model,83.8% and80.0% of the second-solve clocks in the middle/final block lie within the outer1% tails of their sigmoid probability. The final vector second solve has57.2% of channel/sample entries there. Detailed values are in evidence/product16_20261024/composition.json.

Saturation can be useful near-binary selection. These observations do not prove it causes the quality gap. They provide a measurable alternative to expanding rank or adding a spectral constraint: test whether the positive bounded interval (.2,1) limits the needed operator modulation or controller adaptation.

## One center/slope-matched family

Let g be the unchanged existing controller MLP output. Write its duration as

```text
s_a(g) = c + a*tanh(k*g/a),  c=.6, k=.02.
```

The original professor sigmoid is exactly this family with a=.4. A positive wider-range control uses a=.6, equivalent to shift0, scale1.2, tau15 in the existing sigmoid implementation. The affine limit is s_infinity(g)=.6+.02*g. All three share their value, first derivative and zero second derivative at g=0:

```text
s_a(g)=c+k*g-(k^3/(3*a^2))*g^3+O(g^5).
```

This avoids changing initial mean duration or its first-order logit sensitivity when comparing ranges. Initial functions are only locally matched, not exactly matched for every g; record that difference. The wider positive range changes both lower and upper limits, so it cannot isolate just removal of the lower floor. The affine control removes outer saturation and allows signed durations. The MLP, matrices, middle SiLU, K1, conditioning, optimizer and initialization remain unchanged.

At a frozen input, the original D(g)P scales each row of P by at most a5:1 ratio across duration values; its row direction remains fixed. This is a restriction of that hidden operator family, not a bound on the complete velocity Jacobian, which includes gate derivatives and other nonlinearities. It does not prove an impossible full-network fit. Affine clocks retain the controller MLP's SiLU nonlinearity while relaxing the final saturation.

A signed scalar duration can be read as an internal forward/reverse integration horizon; a signed vector D defines a different frozen generator. It is no longer a positive elapsed-time interpretation shared by all channels. K1 also need not be invertible. If a relaxed clock helps, describe the result as state-dependent operator modulation and test any later series separately. For unbounded durations, P norms alone no longer bound all effective generator norms across inputs; measure D(x)P rather than adding a speculative cap.

## Required comparison before acceptance

First compare original vectorK1, the wider positive control, the affine limit, and both SwiGLU budgets, with the same Gaussian/Ring8/product16 validation protocol. Register endpoint and stopping criteria before training; do not add a new operator or nonlinear-drift return choice to this test. If a relaxed vector clock is useful, freeze it and then include the corresponding scalar control before attributing benefits to granularity. Complete longer-budget, independent fresh holdout and actual latency/throughput checks before claiming a useful architecture. A candidate still losing to the same-budget baseline is not accepted because it beats an earlier ODE variant.
