# Next causal question: is hidden numerical bias exposed or compensated?

Registered after the completed matched-solver result, before observing the
crossed-weight or complete continuous-field substitutions. Better hidden
fidelity did not improve common-bank generation. The next step is a frozen
weight diagnostic, not a new response search or another training architecture.

## Fixed objects

Keep the four 20k checkpoints (midpoint/RK4 on Ring8/Product16), the same native
model tensors and all 4,096 selection-field inputs/oracle velocities per task.
No checkpoint selection or parameter changes. Preserve both ordinary controls
and the existing target SW2 comparison. Use FP64 for the numerical diagnosis;
first reproduce saved FP32 predictions under the existing replay/conversion gates.

For each trained weight set, evaluate the complete network with native
midpoint4 and native RK4 two-step local solves, then with continuous local solves.
Every one of the six hidden solves is changed together. Continuous evaluations
must integrate from their own actual sequential entrances, rather than inserting
references computed at the original native entrances. Keep the hidden interval
[0,1], generator P, response ISRU and every surrounding projection/SiLU fixed.

## Exact field-risk accounting

For one weight set and numerical method define:

    e = F_num(x,t) - v_oracle(x,t)
    delta = F_cont(x,t) - F_num(x,t)
    R = mean ||e||²
    S = mean ||delta||²
    C = mean <e,delta>
    R_cont - R = 2 C + S

This is an exact finite-sample identity. All means use the same states and
vector-norm convention. Its Cauchy-Schwarz bound is
abs(R_cont-R) <= 2 sqrt(R S) + S. When R>0, define exposure r=S/R and alignment
c=C/sqrt(R S) when S>0; relative risk change is 2c sqrt(r)+r. Preserve zero cases.
This distinguishes numerical-bias magnitude from its error-correcting direction.
A large hidden-state correction need not cause a large field correction.

## Falsifiable compensation comparison

Write L_MM/L_MR for midpoint-trained weights evaluated with midpoint/RK4;
L_RM/L_RR for RK4-trained weights evaluated with midpoint/RK4. A strong
solver-specific compensation hypothesis predicts BOTH diagonal advantages:

    L_MM < L_MR and L_RR < L_RM

on both tasks, with paired-state evidence and qualified numerical resolution.
If either advantage fails, reject that strong compensation explanation. A small
continuous-field effect relative to the actual field error instead motivates
studying output sensitivity; quantify r and the exact risk decomposition rather
than declaring an arbitrary 1% threshold. Opposite task effects stay separate.

## Numerical qualification and decision

Use two independent-tolerance complete continuous evaluations, and independently
check each saved local solve. Keep all 4,096 inputs and all six layers. Record
failed gates; do not select a favorable layer or shrink to a passing subset.
Bind input/checkpoint/source arrays and reconstruct the risk identity separately.
Keep numerical uncertainty separate from conditional paired evaluation intervals.

If risk changes materially, check whether its direction survives actual target
sampling before attributing generative value. If faithful solving is weakly
exposed at the field level, measure that before another order/activation change.
The investigation remains active until these alternatives are tested; the RK4
result alone does not explain every possible ODE architecture.
