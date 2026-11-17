# Exact linear-operator factorization of the analytic observer

Registered after the eager profile, before evaluating this factorization.
The original compiled-step comparison remains separate and unchanged.
The reference16,B4096profile records7,300aten::mmcalls; matrix products use
59.15%of recorded CUDAoperator time and scalar powers24.58%. This motivates
algebraic reuse in the SAMEcoupled equations, not another learned layer.

## Derivation

Let q0=q(y),q1=q'(y),...,q4=q''''(y), with componentwise products below.
Use the same matrix P throughout and define

    f   = P q0
    a   = P(q1*f)               # Jf
    b   = P(q1*a)               # J^2 f
    c   = P(q2*f^2)             # H[f,f]
    w   = P(q1*(-b/120+c/480) - q2*f*a/240 - q3*f^3/720)
    r   = q1*w + q2*(f*b/120-f*c/480+a^2/160)
          + q3*f^2*a/480 + q4*f^4/2880
    m   = -q2*f^2/24 - q1*a/6

Then the original RK4and midpoint defects satisfy exactly

    g_RK4 = P r,       g_midpoint = P m.

Expand P r and use linearity to recover every term of
rk4-defect-derivation.md. In particular P(q1*w)groups the four terms whose
outer operation is J; the remaining five curvature terms share the same
outer P. No commutativity between P and diagonal q derivatives is assumed.

For the actual solver prefactors, define

    cR = (1/2)^4 * (1[cross=RK4] - 1[native=RK4])
    cM = (1/4)^2 * (1[cross=midpoint] - 1[native=midpoint]).

The coupled drift is exactly

    y' = f
    z' = P(q1*z + cR*r + cM*m).

This takes six Papplications instead of the original nineteen per nonzero-
forcing drift evaluation. The integration scheme,256reference steps, FP64,
full network and every experimental input remain unchanged. For zero forcing
use the original two-product unforced variation directly.

## Separate power reuse

Check a second algebraic variant separately. For u=1+1.2*y^2, set
r0=rsqrt(u),v=1/u. Then u^(-3/2)=r0*v,u^(-5/2)=r0*v^2,and similarly for
the remaining half-integer powers. These are the same real-valued derivatives;
their floating-point order differs and therefore requires the same numerical
identity checks. Do not infer correctness from the symbolic identities alone.

## Checks and selection before target transport

1. On ALLsix learned layers and first16continuous-reference entrances from
   each of the eight field cases, compare both factored drift variants with
   the original drift at a fixed nonzero tangent. Retain all768states, both
   solver directions and zero forcing. Component tolerance1e-12+1e-10*abs(ref).
2. Count actual linear calls in the isolated drift to verify19versus6without
   counting unrelated model operations. Preserve the original helper source.
3. Both compiled variants require the same independent JVP/FD/zero/sign/RNG
   witness and ALL4,096-input primal/tangent comparisons as the original
   compiled-step protocol. Same FP64component gates, no omitted points.
4. Use the same three paired resident timings on ProductRK4confirmation,
   report compilation cost separately and select only a numerically qualified
   measured speed gain. No fitted coefficient or new data/training.

This is an execution optimization of the already fixed analytic predictor.
Its benefit must not be reported as an FFNtraining or image-quality gain.
Only newly empty physical145GPUs4-7may be used;0-3remain prohibited.
