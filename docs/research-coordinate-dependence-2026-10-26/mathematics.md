# Conditional-redraw error identity

At a fixed t, Product16 latent coordinates are independent. For a fixed learned field define `f_i(q)=R^T v_theta(Rq,t)` and scalar error `e_i(q)=f_i(q)-v_i*(q_i,t)`. Conditional on q_i, A and H_i are independent completions, with the same conditional law. Consequently

```text
E[e_i(A)e_i(H_i)|q_i] = E[e_i|q_i]²
E[(e_i(A)-e_i(H_i))²/2|q_i] = Var(e_i|q_i).
```

Integrating gives the conditional-mean/conditional-variance risk decomposition. The target cancels from the difference exactly; the conditional component need not be affine or have any fitted basis. Both population contributions are nonnegative; the finite Monte Carlo cross-product estimator need not be.

This is a pick-freeze/global-sensitivity construction, not a new model. Primary methodological records: Jansen (1999), Analysis of variance designs for model output, DOI10.1016/S0010-4655(98)00154-4; Sobol (2001), Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates, DOI10.1016/S0378-4754(00)00270-6. Publisher metadata/abstracts were verified; the displayed estimator is derived directly above rather than attributed as an independently checked theorem from inaccessible full text.

The relevant population law, cached training-bank distinction and exact latent oracle are already derived in ../research-architecture-search/posterior-basis-design-review.md. Finite local cross-derivatives do not alone determine this integrated conditional variance. Conversely, large D would not prove that constraining hidden P to be diagonal or symmetric solves the problem: P acts in learned normalized coordinates and the measured decomposition acts in known physical latent coordinates.

Intervals resample independent row pairs. Common draws across t and model are deliberate pairing, so neither16 output coordinates nor five times increase the bootstrap's independent sample count. The new T risk is a componentwise paired population estimate, not the old finite-panel metric or a fresh trained-model result.
