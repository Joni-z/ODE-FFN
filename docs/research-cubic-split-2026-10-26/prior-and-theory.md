# Why this field, and what is already known

The evidence motivates testing a nonlinear response rather than another freedom in the same linear-flow/SiLU core. The paired P/W experiment did not show the registered major P-learning problem. Moving a SiLU threshold rather than its output offset did not close the local gap. These are negative controls, not proof that the cubic field will work.

[Carlile et al., Improving Deep Learning by Inverse Square Root Linear Units](https://arxiv.org/pdf/1710.09967), Section5, already defines ISRU as `z/sqrt(1+alpha z^2)`. Our cubic subflow has alpha=2a. This old paper is cited as direct activation prior art, not a frontier performance recommendation.

[Tai et al., A Mathematical Explanation of Transformers](https://arxiv.org/html/2510.03989v2), revised2026, uses operator splitting and constrained projections to interpret a specified Transformer. It does not establish a generation benefit for our chosen field. [Zhu et al., On Numerical Integration in Neural Ordinary Differential Equations](https://proceedings.mlr.press/v162/zhu22f.html), ICML2022, studies solver-dependent learned dynamics. It is relevant caution against identifying a finite-step trained block with an exact underlying physical ODE. The cubic field and the finite-map derivations below are our algebra, not a result attributed to either paper.

Let `B=I+(a/2)P+(a^2/8)P^2`, `B_a=.5P+.25aP^2`, `u=Bh`, `v=C_a(u)`, `g=grad a(h)`, `D=diag((1+2a u^2)^(-3/2))` and `c_a=-u^3*(1+2a u^2)^(-3/2)`. The exact implemented input Jacobian is

```text
J = B D B + [B(D B_a h + c_a) + B_a v] g^T.
```

The second term is the complete input-selected-clock feedback. Dropping it is not justified by the field interpretation. The same P participates in all four linear actions, so its adjoint must sum all four routes.

At fixed small a, direct expansion gives

```text
F(h) = h + a*(P h - h^3)
       + (a^2/2)*(P^2 h - P(h^3) - 3h^2*(P h) + 3h^5) + O(a^3).
```

The quadratic coefficient equals `.5 J_f(h) f(h)` for `f(h)=P h-h^3`. Thus the implemented symmetric composition agrees through second order with this nonlinear field. The linear polynomial is not an exact exponential, so exact-flow reversibility or global numerical stability is not asserted.

The continuous energy bound follows from `y^T P y <= mu ||y||^2` and `sum y_i^4 >= ||y||^4/d`. It provides a particular large-state damping mechanism. No assumption about the target distribution or FM optimum appears in that bound; useful regression remains an empirical question.

Standing workflow guidance: planning-with-files, hypothesis-generation, experimental-design, daily-coding and results-analysis. [Kassis et al. (2026), Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065) is acknowledged as procedural provenance only, with metadata checked in the preceding literature review.
