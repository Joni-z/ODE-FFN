# Stage4: is the extra SiLU necessary once the rates depend on the input?

Registered before training this stage. This is one structural deletion, crossed with scalar/vector gates. It is a falsifiable hypothesis, not a claimed improvement.

## Evidence motivating the question

The vectorK1 lead improves Ring8 field fitting but does not improve its heldout generation over SwiGLU. It also fails to transfer its field advantage to the rotated16D task. Increasing Heun32 to128 changes sampling little. K2 and a static polynomial parameterization have already failed to justify their extra restrictions or computation.

Validation-only forward hooks on the final20000-step references measured the intermediate SiLU derivative. Across the five registered times, Ring8 vector-gated blocks have weak-coordinate fractions (absolute derivative<.05) of20.87%,19.52%,16.87%; scalar blocks18.86%,18.57%,21.22%. On rotated16D these fractions are only2.26%,3.53%,3.40% for vector gates. These are activation measurements, not evidence that deleting SiLU will help after retraining; the second task is a useful rival explanation to a universal saturation story.

## Operator and exact mathematical consequences

Write one K1 ODELayer as F_i(h)=h+r_i(h), r_i(h)=D_i(h)P_i h. A scalar rate is the special case D_i=s_i I. The professor composition is F2(SiLU(F1(h))). The proposed ablation is F2(F1(h)). The gates retain their existing internal SiLU and sigmoid, so the proposed block remains nonlinear whenever the input-dependent rates make it so.

The old composition tends to SiLU(h) as both generators tend to zero. The new composition tends to h. Its differential is J_F2 J_F1 instead of J_F2 diag(SiLU'(F1(h))) J_F1. This removes a diagonal factor that can have tiny, negative or zero entries. For one K1 layer,

```text
J_F = I + D P + diag(P h) J_s.
```

If the norm of the residual Jacobian D P+diag(P h)J_s is less than rho<1 at a point, the smallest singular value of J_F is at least1-rho there. Composition gives the product of these lower bounds at the corresponding intermediate points. A global residual Lipschitz bound<1 would additionally imply global invertibility; it is NOT established or enforced here. Existing P norms alone do not bound the complete residual Jacobian.

SiLU's exact derivative is sigmoid(u)*(1+u*(1-sigmoid(u))). Its noninjectivity prevents the professor's internal composition from being a globally invertible ODE flow map on all hidden states. However a velocity-field approximator need not itself be invertible, and the outer residual and normalization change the complete backbone. Removing this obstruction is mathematical structure, not a proof of a better FM objective or a useful architecture.

Constant rates make the proposed composition linear, whereas the original retains the intermediate SiLU. Thus the two function classes and their initial hidden functions differ deliberately. Parameters, RNG draws and all initial trainable tensors are exactly paired across the activation factor. The zero-initialized output head gives the same initial zero velocity but does not make hidden optimization identical. Do not describe this as a pure optimization-coordinate change or as a globally exact flow solver.

## Frozen screen

Five arms: scalarK1+SiLU, scalarK1+identity, vectorK1+SiLU, vectorK1+identity, SwiGLU. Existing gate widths64 versus32 keep budgets nearly matched; the activation factor changes no parameter or linear MAC. Retain the same width64, depth3, seed0, batch512, AdamW settings, independent source-target coupling, time sampling, data manifest and5000-step endpoint. Datasets: anisotropic Gaussian2D and Ring8. Train two CPU workers, preserving all six earlier20000-step trajectories.

Primary readout is final random-time validation oracle MSE. The common-panel field/Jacobian errors, generation SW2, learning curves and measured compute remain required checks. Advance an identity composition only if it reduces Ring8 oracle MSE by at least10% versus its same-gate parent, has no>5% Gaussian oracle regression, and has no>5% Ring8 Jacobian regression. If it passes, continue the unchanged candidate and controls to20000 and then check nonlinear16D. A benefit versus the scalar parent alone is insufficient for the full goal; compare the practical frontier to SwiGLU.

No test split is consulted in this screen. Historical Stage3 test outcomes were already read and cannot be called untouched in a future architecture selection. A later frozen confirmation must use a newly generated independent evaluation bank from the same declared populations, with generation details fixed before its output is read. No additional seeds, spectral caps, losses, activation families or learning-rate choices are included.
