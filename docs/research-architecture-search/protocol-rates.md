# Stage2: rate granularity crossed with Taylor order

Written before this stage's outcomes. Static-stage evidence shows the fixed K2 matrix parameterization is not a default advantage: a function-matched collapsed FFN lowers Ring8 field error from.07652 to.02890 at identical parameter count, with fewer MACs. Both remain worse than the practical SwiGLU reference. This motivates testing the input-dependent operator family rather than further static powers.

## One design axis

The candidate changes scalar s(h) into coordinate rates s(h), with frozen generator `A(h)=diag(s(h))P`. The same P is reused across terms. Mathematical specification, Jacobian and noncommutativity are in [mathematics-rates.md](mathematics-rates.md). This is not claimed as a novel general idea or as a proved FM-specific bias.

Five arms:

- scalarK1, scalarK2: professor layer, gate hidden64.
- vectorK1, vectorK2: vector gate hidden32, scalar-reference RNG hidden64 for paired surrounding weights/P.
- SwiGLU: unchanged practical reference.

The vector gate has33 fewer parameters per layer than the scalar gate at width64. No new projection matrix, spectral constraint, special output-gate initialization, loss or time conditioning is added. Scalar/vector initial gate functions differ, while all non-gate parameters are paired exactly. Gate width changes to maintain parameter/MAC budget and is explicitly a design tradeoff; it is not a proof of width-independent granularity causality.

## Training and screening

Existing Gaussian2D, Ring8 and Gaussian16D data; exact copied v1 files. Training seed0,5000updates,batch512,AdamW3e-4,wd0,clip1,width64,depth3,uniformt,independent Gaussian source, directv. CPU2 workers based on the measured resource benchmark. All arms are fresh runs in one immutable runtime.

Validation oracle MSE is the primary screen. Common512-state/5-time velocity and Jacobian errors, validation2048-sample SW2, parameters, forward linear MACs and measured time are secondary checks. No heldout test set is used to select an architecture.

Predeclared next-step rule: a vector arm must reduce Ring8 oracle error by at least10% relative to its scalar arm at the same K, with no>5% regression on either Gaussian control, to support further investigation of directional adaptation. A candidate must also approach or improve the SwiGLU quality/compute frontier to qualify for the full goal. Directional adaptation success alone is not completion. If K1 matches K2 with fewer MACs, prefer K1 and do not attribute benefits to high-order structure.

If no candidate clears the first rule, stop this mechanism branch rather than enlarge the gate or add more generators. If one clears it, freeze that candidate before a nonlinear higher-dimensional task and heldout evaluation. Report mismatches between field/Jacobian/generation rankings rather than hiding them.
