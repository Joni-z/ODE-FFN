Verdict: **PASS within the assigned mathematical scope; no open issues found.** R1 and R2 are resolved, and the U6 dependency row now includes U4. All nine source units U0–U8 were reread, and the original 35 microclaims were rechecked against the revised source.

Assurance: **native / same-family / provisional**. This is a same-session revision verification, not a fresh independent reviewer, cross-family semantic acceptance, formal proof certification, or numerical verification. No numerical outputs were used as proof evidence. No experiments or delegation occurred.

Inputs:

- `/Users/mr.z/ODE-FFN-autoresearch/docs/autoresearch/cycle22-denoiser-dynamics/derivation.md`
- `/Users/mr.z/ODE-FFN-autoresearch/docs/autoresearch/cycle22-denoiser-dynamics/PROOF_SKELETON.md`

Source SHA256 at this review:

```json
{
  "derivation.md": "c8f4b1bbc2bad39f40d9ec70bda711df62db4ef9de2bf85aa7362d2fbdcc4f38",
  "PROOF_SKELETON.md": "8d861480f7f01f251c820bb29b432632072967184d7c1fb65eebb69e75dd1644"
}
```

Complete scope receipt:

```json
{
  "shard_id": "S0",
  "round": 2,
  "assigned_unit_ids": ["U0","U1","U2","U3","U4","U5","U6","U7","U8"],
  "covered_unit_ids": ["U0","U1","U2","U3","U4","U5","U6","U7","U8"],
  "status": "COMPLETE",
  "entries": [
    {
      "unit_id": "U0",
      "source_lines": "11-43",
      "obligation_ids": ["U0-01","U0-02","U0-03","U0-04"],
      "status": "RECHECKED",
      "finding": "Finite-GMM moments, compact-set domination and posterior positivity remain sufficient; smooth Bayes-ratio versions are now explicit."
    },
    {
      "unit_id": "U1",
      "source_lines": "45-52",
      "obligation_ids": ["U1-01","U1-02","U1-03"],
      "status": "RECHECKED",
      "finding": "Conditional-expectation orthogonality and fixed-time positive weighting remain correct; uniqueness now explicitly uses the law of Z_t."
    },
    {
      "unit_id": "U2",
      "source_lines": "54-69",
      "obligation_ids": ["U2-01","U2-02","U2-03","U2-04"],
      "status": "RECHECKED",
      "finding": "Heat identity, posterior mean identity and covariance Jacobian, including PSD and degenerate cases, remain correct."
    },
    {
      "unit_id": "U3",
      "source_lines": "71-92",
      "obligation_ids": ["U3-01","U3-02","U3-03"],
      "status": "RECHECKED",
      "finding": "Score differentiation and denoiser PDE remain correct; the domain now explicitly distinguishes vector y from scalar r."
    },
    {
      "unit_id": "U4",
      "source_lines": "94-115",
      "obligation_ids": ["U4-01","U4-02","U4-03","U4-04","U4-05","U4-06"],
      "status": "RECHECKED",
      "finding": "Likelihood equivalence, spatial rescaling, time derivative, uncapped conditional velocity and material derivative remain correct."
    },
    {
      "unit_id": "U5",
      "source_lines": "117-138",
      "obligation_ids": ["U5-01","U5-02","U5-03","U5-04","U5-05"],
      "status": "RECHECKED",
      "finding": "Gaussian drift, invariant endpoint map and excess risk remain correct; the bounded non-invariance conclusion is preserved."
    },
    {
      "unit_id": "U6",
      "source_lines": "140-147",
      "obligation_ids": ["U6-01","U6-02","U6-03"],
      "status": "RECHECKED",
      "finding": "Wrong-prior PDE consistency and 2/9 true-law excess risk remain correct; the skeleton now lists U4."
    },
    {
      "unit_id": "U7",
      "source_lines": "149-165",
      "obligation_ids": ["U7-01","U7-02","U7-03"],
      "status": "RECHECKED",
      "finding": "The fixed-y normalized posterior derivative and contracted third-moment expansion remain correct and appropriately scoped."
    },
    {
      "unit_id": "U8",
      "source_lines": "167-178",
      "obligation_ids": ["U8-01","U8-02","U8-03","U8-04"],
      "status": "RECHECKED",
      "finding": "Both posterior laws, their fixed-prior realizations and unequal drifts remain correct; the closure impossibility is limited to a common function of (y,lambda,m,C)."
    }
  ],
  "errors": []
}
```

The receipt establishes source-unit execution coverage, not completeness of every possible obligation or semantic acceptance by itself. The deterministic microclaim IDs and their detailed extraction remain in `raw-round1.md`.

Exact revision checks:

1. **R1 RESOLVED.** `derivation.md:29-33` now specifies the smooth Bayes-ratio representative of m, using the Gaussian likelihood phi_r(y-x), and the analogous representative of D with likelihood phi_((1-t)^2)(z-tX). The latter is the correct likelihood of Z_t given X. Arbitrary almost-everywhere versions are explicitly excluded from pointwise differentiation. The positivity and domination argument in lines 35-43 provides the smoothness needed by these definitions. No hypothesis was strengthened and no theorem scope changed.
2. **R2 RESOLVED.** `derivation.md:47-48` now states uniqueness up to almost-everywhere equality under the law of Z_t. `derivation.md:73` explicitly quantifies every vector y in R^d and every scalar r>0. These changes remove the undefined p_t notation and the vector/scalar domain ambiguity without changing the result.
3. **DEPENDENCY CLARIFICATION VERIFIED.** `PROOF_SKELETON.md:17` now lists U1,U3,U4 for U6. U4 supplies the FM-coordinate application of the necessary PDE. The dependency graph remains acyclic.

Full-scope regression check:

- **Definitions and hypotheses.** Finite d, a fixed finite Gaussian-mixture prior, PSD component covariances, independent standard Gaussian observation noise, and interior t or r are still stated. Covariance and the contracted third moment remain posterior objects; they are not substituted for hidden-layer matrices. Singular components and point masses are allowed. No inverse of a prior covariance is used.
- **Derivative and expectation exchanges.** On any compact observation set and compact positive variance interval, each required finite-order likelihood derivative, including the extra X in the posterior numerator, is bounded by C(1+||X||^N) for a finite N. Finite-GMM moments make this integrable. The density is positive and continuous; it is bounded below on each compact set. Consequently quotient/logarithm derivatives and mixed derivative commutations used by U2-U4 are justified. U7's positive-precision local differentiation has the same polynomial domination. There is no endpoint-uniform conclusion or interchange of an endpoint limit with expectation.
- **Regression optimum.** Finite loss and X in L2 give the integrability required by the orthogonal decomposition. The positive, time-only denominator rescales fixed-time excess risk. No claim about a time-integrated finite risk under arbitrary delta, finite-width optimization, or a capped external velocity is added.
- **Heat, score and Jacobian.** The kernel satisfies partial_r p=.5 Delta p; differentiation yields m=y+r s and J_m=C/r. The latter gives symmetry and PSD, including rank-deficient posteriors. Differentiation of log p yields partial_r s=.5 Delta s+J_s s, giving partial_r m=.5 Delta m+J_m(m-y)/r.
- **FM coordinates.** The identity D(z,t)=m(z/t,((1-t)/t)^2) gives J_m=t J_D, Delta_y m=t^2 Delta_z D and r'=-2(1-t)/t^3. The resulting coefficients remain -(1-t)/t for the Laplacian, -2/(1-t) for J_D D and (1+t)/(t(1-t)) for J_D z. Adding J_D(D-z)/(1-t) gives the stated material derivative. The marginal-velocity formula follows directly by conditioning X-e and using e=(z-tX)/(1-t).
- **Gaussian material derivative and endpoint risk.** With V=t^2+(1-t)^2, the material derivative of tz/V is (1-t)z/V^2. At t=.5 it is 2z, while the instantaneous velocity is zero. Since u=(V'/2V)z, F=z/sqrt(V) is invariant and extends to F(z,1)=z. At t=.5, F-D=(sqrt(2)-1)z, Var(Z)=.5 and delta^2=.25, producing excess risk 2(sqrt(2)-1)^2. This endpoint evaluation is directly justified by the explicit Gaussian formula.
- **Wrong-prior PDE example.** Every v>0 Gaussian prior belongs to U0. For v=2 and t=.5, D_2=4z/3 whereas the true conditional mean is z. The true-law risk calculation is (.5)(1/3)^2/.25=2/9. Satisfying the necessary PDE therefore does not establish the correct prior or optimum.
- **Precision derivative.** Normalized differentiation at fixed y gives -.5 Cov(X,||X-y||^2). Writing X=m+xi yields Cov(X,||X-y||^2)=kappa+2C(m-y), hence C(y-m)-.5 kappa. Gaussian posterior odd central moments vanish, but the atomic examples demonstrate that this does not extend to all mixture posteriors.
- **U8 realizations and closure scope.** q_A has moments (mean,variance,third central moment)=(0,1,0); q_B has (0,1,1.5). Prior weights proportional to q_i exp(x_i^2/2) are positive and normalizable on either finite support. At the fixed anchor y=0,lambda=1, the likelihood cancels this exponential factor. The priors remain fixed when lambda is differentiated. The resulting drifts are 0 and -.75. Embedding these scalar examples along one coordinate preserves the result in every finite d>=1. Only a single prior-independent closure based on (y,lambda,m,C) is refuted.

Counterexample regression:

- Point masses still give constant m and D, zero covariance, and zero derivatives on both sides of all identities.
- Singular Gaussian priors still give m=mu+Sigma(Sigma+rI)^(-1)(y-mu), C=r Sigma(Sigma+rI)^(-1), zero spatial Laplacian, and matching variance derivative -Sigma(Sigma+rI)^(-2)(y-mu).
- For a symmetric two-point prior on {-a,a}, m=a tanh(lambda a y), C=a^2 sech^2(lambda a y), kappa=-2mC, and partial_lambda m=Cy. The third-moment term supplies exactly the correction missing from C(y-m).
- Identical mixture components and arbitrarily small positive weights do not break positivity or fixed-prior local domination. No uniform claim over mixture parameters is asserted.
- Potential growth near t=0, t=1, or r=0 remains outside the expressly interior-only differential scope. Heavy-tailed priors remain outside the specified prior class.
- The counterexample obtained by changing a conditional-mean version at one null point is now excluded explicitly by the smooth-ratio definitions.

No new counterexample to the assigned mathematical claims was found. No FATAL, CRITICAL, MAJOR or MINOR issue remains open within this review scope.

The non-claims remain intact: consistency is insufficient; these are not novelty claims; no architecture win, general hidden-state impossibility, or causal explanation of older failed covariance models is inferred. No benchmark/data certification, numerical certification, citation audit or model-quality claim is included in this verdict.

Files written by this reviewer are limited to:

- `/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle22/proof-review/raw-round1.md` — verbatim prior final report, including its original scope/status statements.
- `/Users/mr.z/ode-workspace-prep/autoresearch/oct28-cycle22/proof-review/raw-round2.md` — this revision verification.

Proof/source files and all other files were untouched by this reviewer. The only remaining assurance limitation is that this positive mathematical assessment is same-family and provisional; it is not an independent cross-family acceptance.
