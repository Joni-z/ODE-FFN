# Map versus increment, without attributing initialization to expressivity

The professor's finite Taylor map includes an identity term. Its documented
increment alternative subtracts that term. At K1,

```text
F_map(h) = h + D(h) P h
F_inc(h) =     D(h) Q h.
```

Both are inserted twice around the same SiLU, inside the same outer residual.
The direct identity contributes I to the solve derivative, but also changes
the input of the next solve. Removing it from each solve differs from
subtracting h or SiLU(h) from the whole block.

## Fixed-rate equivalence and its boundary

For constant invertible D, Q=P+D^(-1) makes both maps identical for all h.
Both cover every linear map when P/Q are unconstrained. The translation has
unit parameter derivative, so under wd0 their P gradients, clipping and Adam
updates correspond exactly when initialized this way. Eight actively clipped
double-precision updates test the correspondence. A spectral constraint or
weight decay could destroy it and is excluded here.

If D depends on h, the required shift D(h)^(-1) is generally not fixed.
This does not prove a universal expressivity ordering for finite-width
controllers, but it blocks declaring the trained families a pure coordinate
change. The actual layer derivative is

```text
J_gamma(h) = gamma*I + D P + diag(P h) J_s.
```

At both P matrices zero, the increment composition is zero. A first
perturbation in either matrix alone still gives zero, since SiLU(0)=0.
Thus the joint-zero point is bilinearly degenerate, while the map composition
is SiLU(h) and generally has nonzero first P variations. This is a precise
failure boundary, not a reason to initialize P at zero.

## A same-initial-function dynamic control

Zero the last controller affine so initially D=cI,c=.6. Start map P at
the legacy draw and increment Q at P+I/c, using no extra random draws.
Each solve, intermediate SiLU input and complete backbone then coincide in
exact arithmetic. Float32 addition to P loses some low bits; native preflight
records the actual numerical mismatch. Only the final output head is zero
in ordinary training, so tests also use a nonzero head to expose differences.

The two models cease to correspond when the rates adapt. Even at their
paired initial function the controller has different parameter response.
For a common downstream derivative q, gradients with respect to durations are

```text
grad_s,map = q * (P h)
grad_s,inc = q * (Q h)
difference = q * h / c                       (vector durations)
difference = sum_j q_j h_j / c               (scalar duration).
```

The sigmoid's zero-logit sensitivity is r=scale/(4*tau)=.02. If a(h) is
the hidden SiLU controller feature, the last weight-gradient difference is
r times the sum of these contributions outer a(h); the bias difference is
r times their sum. Meanwhile P gradients agree at the paired function.
Unit tests verify both identities through autograd. First-controller-layer
gradients are zero while its last weight is zero in either control; that
shared initial delay is not evidence of a failed first controller.

These exact local identities predict a difference in learning response,
not its sign or a convergence rate. The zero output head initially masks
all backbone responses, and ordinary Adam/clipping remain part of the later
dynamics. No state-independent matrix-spectrum theorem selects the winner.

## What an outcome could establish

Legacy-only gains can arise through substantially different initial features.
Agreement with the paired-control comparison would more directly support a
role for the dynamic zero-order structure and its controller response, but
would still not prove global approximation superiority. Both comparisons
must retain practical SwiGLU references, full derivatives, sampling and cost.
Nothing here requires a hidden FFN to be an invertible data transport or
identifies its P with the FM target's velocity Jacobian.
