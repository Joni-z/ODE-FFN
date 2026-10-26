# A parity question for a later mechanism test

Post-result mathematical note. This is not a measured explanation of the training gap or a selected architecture. The internal probes found no strong affine residual reservoir, so a useful next question concerns which nonlinear responses the operations can create, rather than another affine readout.

## Exact identities

For rho(z)=SiLU(z),

```text
rho(z)-rho(-z)=z.
```

Consequently, for a bias-free two-matrix map G(h)=B*rho(A*h), the part odd under h->-h is B*A*h/2. If both original input-selected durations are replaced by constants, the professor's two K2 solves become linear matrices A,B and this statement applies in their FFN input coordinates. The actual architecture has learned clocks, shifts, biases and normalization; the identity is not a theorem that its raw-input odd part is linear.

The distinction is useful because the analytic Product16 population and its velocity are centrally symmetric: v*(-x,t)=-v*(x,t). Its target contains a nonlinear odd tanh response. The cached training bank does not have exact population symmetry, and generic image distributions need not have it.

For bias-free linear projections u(h),v(h), the corresponding gated maps have

```text
odd[ sigmoid(u)*v ] = v/2
odd[ SiLU(u)*v ]    = (u*v/2)*tanh(u/2).
```

Thus the second map can create an odd nonlinear term even without a bias, whereas an ordinary SiLU ridge map or sigmoid-GLU needs other symmetry-breaking/nonlinear structure. These formulas follow directly from the activation identities. The [original GLU-variant paper](https://arxiv.org/html/2002.05202v1), Sections1–2, supplies the bias-free definitions; it is background for the definitions, not evidence for an ODE advantage or a new performance recommendation.

## A pointwise bound that tolerates learned offsets

For any paired preactivations p+=m+d and p-=m-d, define the odd contribution newly produced by the activation beyond half of its input difference:

```text
R(m,d) = (rho(m+d)-rho(m-d))/2 - d/2.
```

R(0,d)=0. Since rho'=sigmoid(z)+z*sigmoid'(z), and |z|*sigmoid'(z)<=1/4, rho' lies in[-1/4,5/4]. Also |rho''|<=3/4. The latter follows from

```text
rho''=2*sigmoid' + z*sigmoid'*(1-2*sigmoid).
```

The bound |z|*sigmoid'(z)<=1/4 follows from 2|z|<=1+cosh(z), using cosh(z)>=1+z²/2. Integrating dR/dm=(rho'(m+d)-rho'(m-d))/2 gives

```text
|R(m,d)| <= (3/4)*|m|*min(|d|,1).
```

This is a loose but explicit global bound for standard SiLU, not an empirical small-offset claim. Actual m,d may depend nonlinearly on raw x. In particular, d can already carry odd nonlinear information from preceding normalization or layers; R measures what the activation adds beyond d/2, not all odd nonlinearity in the network.

## What must be measured before using the idea

For a symmetric target, a finite x/-x pair has the exact risk identity

```text
(||f(x)-v*(x)||²+||f(-x)+v*(x)||²)/2
    = ||f_odd(x)-v*(x)||² + ||f_even(x)||².
```

A later fixed-checkpoint test could measure that decomposition and the actual paired preactivation centers/activation contributions, retaining ordinary GLU/SwiGLU and full-model controls. It must specify its comparison and inputs before outcomes. Small-bias or almost-odd hidden states cannot be assumed from initialization or an activation name.

The previous sigmoid-GLU and vector-clock failures remain evidence against automatic substitutions. This note selects no new activation, clock, parity projection, width, optimizer or training run. Any useful local ODE design still needs a cold generic-controlled experiment and transfer/cost evidence.
