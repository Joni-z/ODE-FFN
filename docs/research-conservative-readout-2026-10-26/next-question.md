# Distinguish input-derivative features from complete-field conservativity

This question follows the completed cold-training outcomes; it is explicitly post-result reasoning. The registered potential setting stops and is not extended or tuned. This note does not select a new training run.

The gradient readout strongly improves original ODE, yet direct ODE's Product16 skew-Jacobian error is only0.83% of its total. A simple removal of skew at the old weights is therefore not an adequate explanation. The new output also computes an additional state-dependent response through the complete backbone and changes parameter gradients.

Two operators give a precise possible future contrast:

```text
A[G] = .5*(G+J_G^T*x)   # observed conservative readout
B[G] = .5*(G+J_G*x)     # radial directional-derivative readout
A[G]-B[G] = .5*(J_G^T-J_G)*x.
```

Both preserve a symmetric homogeneous linear field and add input-derivative features. A is always conservative under the smoothness conditions. B generally is not. For conservative G they coincide; for a general nonlinear G, their difference is the radial action of the antisymmetric raw Jacobian. This is a potential way to separate the transpose/integrability structure from derivative computation, not a theorem of equal function classes, optimizer behavior or hardware cost.

Any such comparison needs a separately registered cold-training protocol, the same generic readouts and actual differentiation costs. It must not become a posthoc multiplier, different scalar potential or extension of a failed checkpoint. Frozen A-versus-B output differences alone cannot predict which class trains better. No result here licenses claiming that conservativity caused the observed gain or that any more-ODE architecture is superior.
