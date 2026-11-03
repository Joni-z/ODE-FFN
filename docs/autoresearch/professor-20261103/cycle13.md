# Cycle13: an odd bounded feature inside P D(h)

Registered after cycle12 closure, before implementation or training. This is a
D-function question; rejected radial, spectrum and range settings remain closed.

## Evidence and exact operator

Historical [free ISRU](../../research-cubic-split-2026-10-26/analysis-report.md)
had a Product16 field lead at5k AdamW updates, with reversals on other tasks.
It was an ordinary independent-matrix FFN, not an ODE success. Its exact response
and initializer are in cubic_core.py at58849ec8bed2a09514a7dffc624b7f31eef04ed9.
Those old scores are not controls for the present Muon/20k study. Searches of
earlier input-D/activation notes found no current ISRU-D or free-ISRU comparison.
Cycles4/5/10 tested constant, sigmoid, shifted, absolute and radial functions.

Set d_i(h)=(1+1.2h_i^2)^(-1/2), q_i=h_i d_i. Each K1 solve becomes
F(h)=h+P q(h). Keep both independent solves and their middle SiLU. The1.2
coefficient comes from the old response, not tuning. No learned controller,
new parameter, P constraint, loss or optimizer is added.

q is odd and bounded by1/sqrt(1.2); q'=(1+1.2h^2)^(-3/2).
J_F=I+P diag(q'), so ||J_F||2<=1+||P||2. Constant-D1 has the same upper
bound; this does not establish a stability advantage over that control.
q(h)=h-.6h^3+O(h^5): the two maps share their zero-input value, Jacobian and
Hessian. Conditional P D and full J use different diagonals. The increment
is bounded at fixed P; F retains an unbounded identity. No contraction,
invertibility, external-flow accuracy or unique ODE benefit is claimed.

In a latent Product16 coordinate the native FM oracle is
u_t(x)=a_t x+b_t tanh(c_t x), with v_t=(1-t)^2+t^2 sigma^2,
a_t=(t sigma^2-(1-t))/v_t, b_t=m(1-t)/v_t, c_t=t m/v_t.
Conditioning the two symmetric Gaussian components gives this formula. It
motivates an affine-plus-bounded-odd response, not a theorem about hidden
normalized coordinates. The ordinary free ISRU shares that motivation.
Only q is odd: the retained middle SiLU and surrounding normalized network do
not become odd, so this is not a parity-enforcement method.
Verify the formula against the native oracle. [ISRU is established prior art](https://arxiv.org/abs/1710.09967v2);
the abstract was rechecked and the exact implementation is copied from our
preserved historical source, not presented as a new activation.

## Eighteen fixed20k trajectories

Each task uses constant-D1, ISRU-D and free ISRU at AdamW3e-4 and Muon.02,
plus the existing selected native Swi64/128/192 references. Muon.02 is the
validation-best archived constant-D1 rate on both tasks, selected before these
outcomes. No new rate sweep. Nine cells per host: Ring8/Lenovo, Product16/145.
Seed0, batch512, native manifests, pair-velocity loss, clip1, Heun32 and all
evaluation banks stay fixed. Two workers per free GPU0; constant and Swi64
launch first. Repeat all ten native references with exact archived numerical
state and saved-array checks. Preserve all earlier ordinary frontier scores.

Copy FreeISRU and cubic_flow unchanged from the historical file, not its
trainer. Both independently trainable matrices initially equal
I+.3P1+.045P1^2. Add matrix1/matrix2 to the existing FFN projection selector,
so official Muon/AdamW treats both as FFN matrices; native families retain
exactly their old selection. Pair surrounding tensors/buffers and construction
RNG. ISRU-D and constant share all tensors; free ISRU changes the FFN initial
function and parameterization. A potential gain cannot isolate all causes.
Record parameters, dense matrix work, spectra and actual cost.

## Decision and checks

Final validation oracle MSE selects. Provisional keep requires >=5% against
matching constant-D1. Continuation also requires >=5% against same-optimizer
free ISRU and lower field error than every ordinary control in this study and
the existing20k frontier. Test SW2 is separate, never selects a model/rate/
checkpoint; worse sampling is a tradeoff. Any qualifying screen needs its own
matched80k registration. A free-ISRU win stays an ordinary-response result.
No K2/coefficient/temperature/clipping/optimizer rescue follows a failure.

- [x] Isolated adapter and provenance-pinned copied control.
- [x] CPU checks: common tensors/RNG, complete derivatives, local matching,
      exact copied control, native oracle and optimizer groups.
- [x] Freeze/publish/deploy; fresh documented CUDA/native8->16 checks.
- [x] Baseline-first monitored runs;18 endpoints and ten exact anchors.
- [x] Independent replay, full controls/curves/spectra/cost and decision.

Closed: [complete analysis](cycle13-results/analysis-report.md). Two AdamW parent
gains are retained; none qualifies against the registered ordinary-control
gates. No80k or image continuation is selected from this screen.
