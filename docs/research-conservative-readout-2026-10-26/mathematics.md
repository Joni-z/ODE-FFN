# A complete velocity gradient, rather than a symmetric hidden P

For independent Z~N(0,I), target Y, a=1-t and X=aZ+tY, let p_t be the density of X. For 0<t<1 and the smooth finite mixture targets in this project,

```text
score_t(x) = (t E[Y|X=x]-x)/a²
v*(x,t) = (E[Y|X=x]-x)/a
         = x/t + (a/t) score_t(x)
         = grad_x [||x||²/(2t) + (a/t) log p_t(x)].
```

Thus the population velocity has a symmetric input Jacobian. The same argument applies to a finite empirical target bank smoothed by Gaussian source noise at open times. Pair labels Y-Z need not themselves have this property as functions of X; their squared-loss conditional mean does. No analytic oracle is needed for ordinary pair-target FM training.

A symmetric hidden P does not impose this complete-field property because controllers, nonlinearities, embedding and output maps all contribute. The new comparison keeps those original components and changes only the final velocity parameterization. Let G_theta(x,t) be the complete existing vector network and define

```text
Phi_theta(x,t) = 1/2 * dot(x, G_theta(x,t))
v_theta(x,t) = grad_x Phi_theta(x,t)
             = 1/2 * [G_theta(x,t) + J_G(x,t)^T x].
```

Original SiLU, sigmoid and finite-epsilon LayerNorm are smooth, so the resulting input Jacobian is the symmetric Hessian of Phi. Gradients of the FM loss with respect to theta include mixed state/parameter derivatives; input-Jacobian evaluation needs second state derivatives. No derivative or gate may be detached. Under no_grad sampling/evaluation, the implementation locally enables differentiation only to compute the velocity, then returns detached outputs.

For G=Ax+b, v=sym(A)x+b/2; in particular, a symmetric linear map with b=0 is preserved. For nonlinear G this operation is not an L2 orthogonal projection of vector fields, and need not improve any given frozen prediction. The potential is not a log-density estimate, a convexity guarantee, a normalizable energy model, or a certificate of stable time integration. Since the potential depends on t, no single-energy descent claim follows.

On a star-shaped domain any smooth conservative target has the radial-path potential Phi(x)=integral_0^1 dot(x,v*(lambda*x)) d lambda, up to a constant. The inner-product parameterization contains it if G can represent twice that integrated field. This is a statement about unrestricted functions, not proof that the fixed normalized network represents it efficiently or trains well.

Earlier small antisymmetric errors make a simple curl-removal explanation weak. The hypothesis is an inductive-bias/capacity/optimization question to be tested with a cold-trained generic counterpart; exact symmetry alone cannot nominate it. The known inner-product energy precedent and negative training evidence are recorded in notes.md.
