# Re-evaluating the original clock inside one solve

**Conditional GO for one vector-clock comparison.** This tests a distinct, previously untrained interpretation of the original layer: the same clock/controller is evaluated at an internal midpoint. It does not repair a mathematical bug in the professor's conditional linear flow. Its empirical prior is weak; no result predicts a gain over SwiGLU. Only the dynamic-midpoint arm could be nominated, after implementation checks and a registered comparison. There is no order, clock-range, width or temperature rescue attached to failure.

## Two different differential equations

Let D(z)=diag(s(z)), with the unchanged sigmoid clock in (.2,1), and let f(z)=D(z)Pz. At input h define

```text
D0=D(h), f0=D0*P*h, h_mid=h+f0/2.
frozen K2: F_fr=h+D0*P*h_mid
dynamic midpoint: F_dyn=h+D(h_mid)*P*h_mid.
```

F_fr is exactly the professor's second-order polynomial `h+D0Ph+.5D0PD0Ph`, including the vector generalization's multiplication order. It approximates the conditional linear ODE z'=D(h)Pz, whose clock is selected by the initial input and held fixed. F_dyn is explicit midpoint applied to the different autonomous ODE z'=D(z)Pz. Both share the same learned P and controller; the latter evaluates that controller twice.

The exact finite difference is

```text
F_dyn-F_fr = diag[s(h_mid)-s(h)] * P*h_mid.
```

No extra learned matrix is involved. In exact arithmetic, a bit-paired implementation can use the original frozen output plus `(s_mid-s)*(Ph+P*f0/2)`; this equals the explicit-midpoint expression and requires two P products. Canonical integration terminology does not make the resulting tied two-stage neural block novel or uniquely ODE-based.

## The missing term belongs to the nonlinear ODE, not the original one

With J_s the vector clock Jacobian,

```text
J_f(z)=D(z)P+diag(Pz)J_s(z)
exact nonlinear second-order expansion:
h+f0+.5D0*P*f0+.5diag(Ph)*J_s(h)*f0.
```

For a precise asymptotic claim, set P=epsilon*P0 while keeping the controller fixed, and write g(z)=D(z)P0z. On compact input sets and sufficiently small epsilon, smoothness gives

```text
Phi_dynamic(h)=h+epsilon*g+.5epsilon^2*J_g*g+O(epsilon^3)
F_dyn-Phi_dynamic=O(epsilon^3)
F_fr-Phi_dynamic=-.5epsilon^2*diag(P0h)*J_s(h)*g+O(epsilon^3).
```

The last error is generically order epsilon^2 when its coefficient is nonzero. In contrast, F_fr approximates its **own conditional linear** exact flow to order epsilon^3. Re-evaluating the clock is therefore a change of vector field, not a solver-order correction to the professor's intended flow. These local statements do not establish small error at a learned unit-size step.

For reference, the leading explicit-midpoint error for the nonlinear field is `-epsilon^3*(D²g[g,g]/24+J_g*J_g*g/6)`. This permits an independent small-epsilon numerical check rather than an assumed error order.

## Scalar clocks do not create new exact orbits

If s is one positive scalar, z'=s(z)Pz has solution `z(u)=exp(r(u)P)h`, where `r'(u)=s(z(u))>0`. It follows the same P-linear orbit with a different clock, and r(1) remains in (.2,1). A sufficiently general conditional scalar clock could select that effective time directly. A changed finite-controller parameterization or midpoint polynomial may behave differently, but it does not establish new scalar exact-flow orbit geometry. This statement is per internal solve, not an expressivity theorem for the complete two-solve network.

Vector clocks can change direction because D(z) need not be scalar. Along a trajectory, the matrices D(z(u))P need not commute. The earlier free-clock tangent result still applies: a nondegenerate vector gate already has a full pointwise receiving span. The new hypothesis concerns composition and state feedback of the **shared controller**, not the discovery of another missing pointwise direction.

Stage5 shared or reused controllers across ode1/ode2; it did not re-evaluate one controller within an integration stage. Stage7 used the different field `D0*P2*SiLU(P1z)` with D0 frozen; its trained arms were K1. These distinctions are explicit in [the clock-reuse protocol](protocol-clock-reuse.md), [the nonlinear-drift protocol](protocol-nonlinear-drift.md) and `synthetic/models.py`. This comparison is not a continuation of Stage20's stopped forced-SwiGLU-gate family.

## Derivatives and stability limits

The full midpoint input derivative is

```text
J_Fdyn = I+J_f(h_mid)*(I+.5J_f(h)).
```

All clock responses at both stages remain live. For comparison, with A=D0P and B=diag(Ph)J_s(h), the complete frozen-map derivative is

```text
J_Ffr=I+A+B+.5[diag(P*f0)J_s(h)+A*(A+B)].
```

Freezing the clock during the auxiliary solve does not mean detaching it from network backpropagation.

Finite parameters and bounded clocks imply `||f(z)||<=||P||*||z||`. Together with local smoothness this gives a global exact autonomous flow. It does not make f globally Lipschitz with constant ||P||: a local bound includes `||Pz||_infinity*||J_s(z)||`. Large hidden inputs, controller derivatives or P can make the feedback term and numerical step large despite the clock's bounded range.

Explicit midpoint is not unconditionally stable or invertible. For constant A=-3I its amplification is 2.5 while exp(A) contracts. For the real two-dimensional matrix A=[[-1,-1],[1,-1]], `I+A+A²/2=0`, whereas exp(A) is invertible. Setting D=.6I and P=A/.6 realizes this inside the proposed layer. Neither a hidden numerical flow nor its exact counterpart automatically satisfies the complete FM Jacobian constraints derived earlier.

## Pairing initialization, including its limitation

Zero the final controller affine weights and biases in **all three new solver arms**, after constructing the same native tensors without consuming extra RNG. Then D(z)=cI with c=.6 everywhere. Frozen K2 and midpoint have identical forward functions and P/input derivatives. This is a declared change from the old random controller initialization; the Euler reference must be rerun with it.

Their controller tangents are different. If a controller perturbation produces q(z)=delta s(z), then at this constant initialization

```text
delta F_dyn-delta F_fr = diag(P*h_mid)*[q(h_mid)-q(h)].
```

The final controller bias has constant q and hence no difference; final controller weights see different hidden features at h and h_mid. First-layer controller gradients are initially zero because the final affine weights are zero. This is the intended state-feedback response, not a matched local controller kernel.

The complete VelocityNet has a zero output head. It therefore initially masks both backbone/controller tangents. Frozen K2 and midpoint have identical complete initial parameter Jacobians because their hidden features also agree; a nonzero output-head probe exposes the local controller difference. Euler generally has different hidden features and output-head gradients, so its complete initial kernel is not claimed to match the K2 pair.

## Cost, controls and decision boundary

At width64 with the retained vector-controller hidden32, each controller costs4096 dense MACs and each P product costs4096. For the retained two-solve FFN:

| Solver | Controller evaluations | P products | Dense MACs |
|---|---:|---:|---:|
| Euler K1 | 2 | 2 | 16384 |
| Frozen K2 | 2 | 4 | 24576 |
| Dynamic midpoint | 4 | 4 | 32768 |

All three have the same learned tensors. Nominal128 SwiGLU costs16320 MACs per FFN and is nearly parameter-matched; nominal256 costs32640 and is nearly midpoint-compute-matched, with more parameters. Actual pointwise work, runtime and memory still need measurement.

The bounded grid is six arms: paired Euler/frozen-K2/midpoint plus SwiGLU96/128/256. Swi96 retains the stronger cheap Ring reference. Swi192 is unnecessary when frozen K2 is only a mechanism control, not a nominee. The candidate must materially outperform both frozen K2 and the relevant SwiGLU frontier under the registered task/transfer guards; beating Euler alone could simply reflect extra evaluation cost. The original loss, coupling, time sampler, output head, P parameterization and clock bounds remain fixed.

Before training, verify the conditional polynomial identity, explicit-midpoint values and full gradients, the nonlinear Taylor clock term and local error orders, exact constant-clock forward/P/input pairing, the nonzero-head controller-tangent difference, and actual executed dense MACs. Log both stage clocks and their change, the extra correction, frozen quadratic term and nonlinear clock-Taylor term separately. None is a whole-network stability certificate.

A positive result would support one state-feedback implementation at its measured budget; it would not prove that approximate integration is uniquely responsible. A negative result stops this fixed state-clock setting. No scalar variant, extra stage, modified gate slope or new loss follows automatically.
