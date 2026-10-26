# What the P-versus-W comparison can identify

For constant duration 1, the professor's K2 solve has effective matrix Q(P)=I+P+P²/2. The matched cores share a function at initialization, but they do not share an unrestricted matrix parameterization.

The exact differential and its Frobenius adjoint are

```text
DQ(P)[H] = H + (PH+HP)/2
DQ(P)^*[G] = G + (P^T G+G P^T)/2
```

Even in a known representable target class, the differential changes the optimizer's function-space geometry. Equal nominal learning rates and parameter counts do not make parameter updates equivalent. The experiment deliberately measures that finite-budget difference without adding a corrective optimizer.

There is also a matrix-image restriction:

```text
Q(P) = [I + (I+P)²]/2
```

For a real eigenvalue lambda of real P, q(lambda)=1+lambda+lambda²/2 is strictly positive. Nonreal eigenvalues occur in conjugate pairs, whose q values contribute nonnegative products |q(lambda)|² to the determinant. Thus det Q(P) is nonnegative, including possible singular cases. Negative real eigenvalues of Q can still occur in pairs; positivity of every eigenvalue is not claimed.

The two-map P core vanishes at zero and has derivative J(0)=Q(P2)Q(P1)/2, with nonnegative determinant. The direct W core also vanishes at zero, but J(0)=W2W1/2 can have negative determinant. For example W1=diag(-1,1,...,1), W2=I is a direct-core function whose origin derivative has negative determinant, so it cannot equal any P-core function globally as a differentiable map.

This is an exact function-class noninclusion example, not proof that a particular SwiGLU target or its values on the normalized feature distribution are unrepresentable. Zero lies away from much of that distribution. A learned W function crossing the sign boundary demonstrates that those effective matrices have left Q's image; it does not establish that another P factorization could not approximate the measured data.

The fixed-flow target is an essential control because it is explicitly representable by both parameterizations. A large P-versus-W fitting gap there cannot be explained by target absence. Conversely, a gap only on the unknown-representability SwiGLU target cannot isolate optimization from class restrictions. These distinctions prevent relabeling an ordinary SiLU control as an ODE improvement.
