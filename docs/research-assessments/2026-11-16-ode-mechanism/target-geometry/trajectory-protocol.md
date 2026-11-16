# Next test: is a transported-error explanation quantitatively valid?

The independent derivative comparison confirms the focal ordering. It does not
explain why ordinary Product16SwiGLU samples better despite greater average
Jacobian error. Before another architecture, test whether first-order transport
of actual field error predicts the existing models' endpoint displacement.

## Exact object and prediction

Let e=v_theta-v*, and x_lambda'=v*(x_lambda,t)+lambda*e(x_lambda,t), all starting
at the same Gaussian source. Write x*=x_0 for the oracle path and z=d x_lambda/d
lambda at zero. Differentiation gives the exact sensitivity equation

```text
z' = J_v*(x*,t) z + e(x*,t),   z(0)=0
x_lambda(1)-x*(1) = lambda*z(1) + O(lambda^2).
```

The last line is a local approximation, not an equality at lambda=1. The question
is whether it remains useful at the learned error scale. The existing
[derivation](../mathematical-basis.md#4-the-necessary-bridge-to-the-actual-generative-task)
supplies the theory; this protocol fixes its next empirical check.

## Bounded calibration before population interpretation

Use all twelve existing20kcheckpoints, with source points0:32from each of the
eight existing **selection** banks:256sources/task, no newly drawn data. Retain
every source/model. Integrate lambda=.125,.25,.5,1, plus the oracle/sensitivity
system. First bind lambda=1to the unchanged native sampler and its stored
endpoints. The observation cannot change trained weights or hidden resolution.

Use FP64 Heun128, repeated at256for numerical qualification. Verify both the
oracle and augmented equations with independent Gaussian linear-plus-bias and
linear-coefficient examples; check error against exact closed forms. Preserve
all intermediate endpoints and actual runtime. An unqualified numerical result
cannot be interpreted as a model failure. Register exact numerical tolerances
and the executable commands after the implementation checks, before the pilot.

Primary diagnostic at lambda=1:

```text
explained_fraction = 1 - mean||delta-z(1)||^2 / mean||delta||^2
delta = x_1(1)-x*(1).
```

Positive means the linear prediction beats predicting no displacement; zero or
negative means it does not. Report all twelve values and every bank, not only
the focal pair. The small-lambda remainder and its observed scaling calibrate
the expansion; they cannot rescue a failed full-scale prediction. No arbitrary
percentage improvement is an architecture gate.

## Decision and limits

If full-scale predictions are useful and numerically qualified, register a
separate all-source analysis of the transported error's timing/direction and
distribution-relevant effect. If they fail, reject this first-order account at
the trained scale rather than using it to design a penalty. Mixed results stay
model specific. No additional training or image promotion follows the pilot.

Paired endpoint error is a coupling-dependent upper bound for Wasserstein
distance, not our SW2 or an exact distribution error. Gaussian rotational
controls explicitly show that large paired displacement can preserve the law.
Any later generation explanation must retain the native large-bank SW2 and
ordinary controls; this calibration alone cannot settle ODE novelty.
