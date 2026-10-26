# Derivative identities and interpretation

For the column-vector convention y=x+aPx+a²P²x/2, this layer's scalar duration a depends on x but not on its own P. With x and a fixed, an arbitrary matrix perturbation H gives

```text
D_P y[H] = aHx + a²(PHx+HPx)/2.
```

If u is the complete downstream adjoint of y, the two parameter-gradient routes are

```text
g1 = a*u*x^T
g2 = a²/2*[(P^T*u)*x^T + u*(Px)^T].
```

All earlier and later dependencies are retained by the native complete-network adjoint. This derivation would need extra terms if the same P were directly shared into its own clock or another solve; the six current matrices are independent and not used by their own controller. Calling g1 a route does not turn it into the loss gradient of a separate K1 model.

Let U=||u||², X=||x||², A=<u,P^T u>, B=<x,Px>. The main calculation avoids materializing a matrix per input by using

```text
||g1||² = a²*U*X
||g2||² = a^4/4 * [||P^T*u||²*X + U*||Px||² + 2*A*B]
<g1,g2> = a³/2 * [A*X + U*B].
```

The independent replay forms those matrices explicitly instead. It reproduces both norms, the cross term, matrix sums and native parameter gradients. The severe-cancellation threshold applies to energy ratios with denominator ||g1||²+||g2||², so the ratio is between zero and two when defined. Global coherent sums can have different cross terms from summed per-input energies; both are necessary to describe what was measured.

For the primary nonlinear functional, O=(f(x)-f(-x))/2 and e=O-v-XB, where B is the best no-intercept raw-linear coefficient for the frozen panel. The derivative of ||e||² has no coefficient-derivative contribution because X^T e=0. Its upstream weights are e/(5*n*d) for f(x), and -e/(5*n*d) for f(-x). These give the exact functional gradient at the frozen model. The main verifies the saved residual risk and orthogonality before taking the pullbacks. Total paired-error weights are (f(x)-v)/(5*n*d) and (f(-x)+v)/(5*n*d).

The normalized P-to-Q derivative is T(H)=H+a(PH+HP)/2. Its Frobenius quadratic form depends only on the symmetric part of P:

```text
<H,PH> >= mu*||H||²
<H,HP> >= mu*||H||²
<H,T(H)> >= (1+a*mu)*||H||²,
```

where mu=lambda_min((P+P^T)/2). If 1+a*mu>0, Cauchy–Schwarz gives ||T(H)||>= (1+a*mu)||H||, while submultiplicativity gives ||T(H)||<= (1+a*||P||2)||H||. Thus the upper condition bound follows from two norm inequalities, without assuming that eigenvalues determine singular values. The matrix P is held fixed; no bound on the loss Hessian or full network follows.

The complete derivative includes the prefactor a. At P=-I/a it vanishes identically although Q=I/2, supplying the preflight stationarity witness. A positive scalar P supplies constructive routes, preventing a universal cancellation claim. The earlier project scalar-gradient analysis and Sylvester observation are prior derivations, not new theory in this study.

The eigenvalues of T are 1+a*(lambda_i(P)+lambda_j(P))/2, even when P is not diagonalizable. Distance of these factors from zero tests singularity algebraically, not robust singular-value conditioning for nonnormal matrices. The spectral-context table retains this distinction. All six observed symmetric minima exceed -1, permitting a uniform positive certificate across the original allowed scalar durations. That uniform corollary is labeled as a post-result calculation from recorded matrices; it does not alter any preregistered decision.

This diagnosis concerns one trained checkpoint, exposed population panels, and complete-model gradients of two specified functionals. It does not prove an optimizer limit, an architecture-wide approximation lower bound, or practical benefit from a new parameterization. No earlier failed model family is reopened by these equations.
