# Cycle 01: does a complete hidden-field correction help frozen checkpoints?

Registered before new diagnostic outcomes. This is a read-only screen, not a new training run or a rescue of stopped Stage20/21 settings. The prior frozen-gate review rejected a training proposal without task-specific evidence. This diagnostic supplies a direct test of the narrower premise; it cannot establish a useful trained architecture.

## Complete field
For block i at fixed external time t, define

```text
c = 1/sqrt(3)
z(h,t) = LayerNorm_i(h)*(1+scale_i(t))+shift_i(t)
A_i(h,t) = U_i[SiLU(Wg_i z+bg_i)*(V_i z+bv_i)]+bo_i
```

The complete hidden ODE is y'=A_i(y,t) over duration c, with t fixed and the block's weights held constant within that interval. Different blocks remain different fields; there is no shared-depth/autonomous-whole-network claim. The unchanged block is Euler: h+c*A(h). Its second-order Taylor update is h+c*A(h)+c²/2*J_A(h)A(h). This is a nonlinear-flow correction; it is not a power of a frozen P.

Let n=h-mean(h), r=sqrt(mean(n²)+eps). For direction v, LayerNorm's input tangent is gamma/r * [v-mean(v)-n*mean(n*v)/r²]. Multiply this tangent by (1+scale(t)) before differentiating the SwiGLU. With a=Wg*z+bg, b=V*z+bv, dz=D_h z[v]:

```text
J_A*v = U[SiLU'(a)*(Wg*dz)*b + SiLU(a)*(V*dz)]
```

Output biases do not recur in the tangent; input biases affect the operating point. The full tangent includes both normalization and gate derivatives.

## Frozen cohort and controls
Replay native nominal SwiGLU128 (actual hidden85), width64, depth3, model seed0 at steps1000,2500,5000 on Gaussian2D, Ring8, Product16 and the dependent-sign32 mixture. All twelve checkpoints are existing training outcomes; do not retrain or select a checkpoint. Restore all nonpersistent buffers with native restore_model. Baseline MSE must replay saved metrics before interventions.

At each checkpoint, intervene at position1,2,3, or all three blocks. The unchanged Euler prediction is common to all positions. Compare four fixed updates:

- complete Taylor2: h+c*A+c²/2*J_A*A;
- frozen-gate tangent: same formula but remove only the gate-derivative term, retaining LayerNorm/value tangents;
- explicit midpoint: h+c*A(h+c*A(h)/2), reevaluating normalization and gates;
- two half Euler steps: h1=h+c*A(h)/2, result=h1+c*A(h1)/2.

The last two execute two field evaluations. Full Taylor executes the primal SwiGLU and its analytic tangent, with approximately twice its dense MACs. Frozen gate uses fewer tangent MACs. Record executed matrix MACs rather than assuming equal runtime. These are mathematical and compute controls, not new independent learned models.

## Data, metrics and fixed budget
Use every row of the existing uniform-time validation field banks (20,000 per law). Keep pair-velocity and exact conditional oracle targets unchanged. Save all prediction arrays outside Git. Do not inspect test sets. At final checkpoints also evaluate all-block interventions on all five existing time panels (512 each), their complete network Jacobians, and generated samples using the historical 2048 sources, Heun32 and128 slice directions. One-block cases receive full uniform-time field evaluation but no extra sampling. Native generation/metric implementations are unchanged.

Run one one-thread CPU process on the Mac, sampled memory/CPU telemetry every2seconds, wall-time ceiling600seconds excluding preflight. Set an exclusive output directory. Refuse output overwrite. A timeout is a failed diagnostic, not grounds to rerun with a longer budget. CPU is used because these tiny models previously underutilized large GPUs; Lenovo reachability is checked separately.

## Decision before outcomes
Only the all-block complete Taylor update at step5000 can pass this screen. It must improve uniform-time oracle MSE by at least5% on both Product16 and the dependent-sign law, regress by no more than5% on either 2D task, keep nonlinear Jacobian/SW2 within10% of unchanged Euler, and beat the matched-computation midpoint and half-Euler controls on both16D laws. This is a practical premise test, not a statistical significance claim. Earlier checkpoints, positions and frozen-gate outcomes are explanatory only; they cannot nominate a replacement if the registered case fails.

If it fails, reject post-training complete-Taylor upgrading at this fixed residual duration. Do not tune duration, correction sign, coefficients, widths, checkpoint choice, number of stages, or selected positions. Do not start fresh Taylor training solely on that failure. Failure does not prove that independently trained higher-order methods cannot work. A pass would require a separate registered training comparison, longer-trained/matched-compute ordinary baselines, broader transfer and audit before claiming a useful ODE design.

## Verification
Hash the twelve checkpoints, all consulted banks, recorded baseline metrics, frozen native Python sources, and the new protocol/runner before and after the run. Preflight analytic tangents against independent automatic differentiation in float64 on actual hidden states in all twelve checkpoints, including the gate-derivative decomposition and its nonzero magnitude. Check c=0, tangent-of-constant-bias behavior, complete wrapper/Euler equality, and that output/parameter/buffer state is unchanged. Save terminal process outcomes and all failed preflight evidence. Publish all outcomes, including failures, in a curated results ledger.
