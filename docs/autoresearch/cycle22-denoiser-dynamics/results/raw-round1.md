Verdict: **WARN — two minor notation clarifications; no substantive mathematical error found.** All nine units were read, and the 35 obligations below were checked analytically. The PDE signs and coordinate factors, Gaussian material derivative, endpoint-map excess risk, wrong-prior counterexample, precision derivative, and U8 prior realizations are correct.

Review assurance remains **native / same-family / provisional**. This is not cross-family semantic acceptance or numerical verification. No files were changed and no experiments were run.

```json
{
  "shard_id": "S0",
  "assigned_unit_ids": ["U0","U1","U2","U3","U4","U5","U6","U7","U8"],
  "covered_unit_ids": ["U0","U1","U2","U3","U4","U5","U6","U7","U8"],
  "status": "COMPLETE",
  "entries": [
    {
      "unit_id": "U0",
      "source_lines": "11-37",
      "obligation_ids": ["U0-01","U0-02","U0-03","U0-04"]
    },
    {
      "unit_id": "U1",
      "source_lines": "39-46",
      "obligation_ids": ["U1-01","U1-02","U1-03"]
    },
    {
      "unit_id": "U2",
      "source_lines": "48-63",
      "obligation_ids": ["U2-01","U2-02","U2-03","U2-04"]
    },
    {
      "unit_id": "U3",
      "source_lines": "65-86",
      "obligation_ids": ["U3-01","U3-02","U3-03"]
    },
    {
      "unit_id": "U4",
      "source_lines": "88-109",
      "obligation_ids": ["U4-01","U4-02","U4-03","U4-04","U4-05","U4-06"]
    },
    {
      "unit_id": "U5",
      "source_lines": "111-132",
      "obligation_ids": ["U5-01","U5-02","U5-03","U5-04","U5-05"]
    },
    {
      "unit_id": "U6",
      "source_lines": "134-141",
      "obligation_ids": ["U6-01","U6-02","U6-03"]
    },
    {
      "unit_id": "U7",
      "source_lines": "143-159",
      "obligation_ids": ["U7-01","U7-02","U7-03"]
    },
    {
      "unit_id": "U8",
      "source_lines": "161-172",
      "obligation_ids": ["U8-01","U8-02","U8-03","U8-04"]
    }
  ],
  "errors": []
}
```

This receipt establishes source-unit execution coverage only; it does not establish completeness of all possible mathematical obligations.

**Extracted microclaims and hypothesis discharge**

Each entry has the form “context ⇒ claim; justification and condition location.”

- **U0-01:** Fixed finite \(d\), finite Gaussian mixture with PSD component covariances and finite component means ⇒ every polynomial moment of \(X\) is finite. Degenerate Gaussian components are included. Assumptions: lines 13–18.
- **U0-02:** Compact observation set and variance interval \(0<a\le r\le b<\infty\) ⇒ every fixed finite-order likelihood derivative, including derivatives of its \(X\)-weighted numerator, has an integrable polynomial envelope. Finite moments from U0-01 discharge domination. Assumptions and stated justification: lines 29–37.
- **U0-03:** Gaussian observation noise with \(r>0\) ⇒ \(p_r(y)>0\) everywhere, and the density and posterior numerator are smooth. Their quotient is smooth; mixed derivatives commute. Uses U0-02 and positivity of the Gaussian kernel.
- **U0-04:** Canonical smooth posterior-ratio definitions ⇒ all displayed differential identities have pointwise meaning for \(y,z\in\mathbb R^d\), \(r>0\), \(t\in(0,1)\). This version is implicit in lines 33–35; make it explicit as noted in issue R1.

- **U1-01:** \(X\in L^2\), measurable \(f\), fixed finite positive \(\delta(t)\), and finite loss ⇒ \(f(Z_t,t)\) and \(D(Z_t,t)\) are in \(L^2\). For \(f\), use \(\|f\|^2\le2\|f-X\|^2+2\|X\|^2\); for \(D\), conditional Jensen.
- **U1-02:** U1-01 and \(D=\mathbb E[X\mid Z_t]\) ⇒ the cross term in the squared-error decomposition has zero expectation. Conditional expectation and Cauchy–Schwarz discharge conditioning and integrability.
- **U1-03:** U1-02 and \(\delta(t)>0\) ⇒ excess risk is \(\mathbb E\|f-D\|^2/\delta(t)^2\), with uniqueness up to the law of \(Z_t\)-null sets. Applies only to fixed-time unrestricted population regression, as lines 41–46 state.

- **U2-01:** U0 regularity and Gaussian covariance \(rI\) ⇒ \(\partial_r p_r=\frac12\Delta p_r\). Differentiate the Gaussian kernel and integrate using U0-02.
- **U2-02:** \(\nabla_y\phi_r(y-x)=(x-y)\phi_r(y-x)/r\) and \(p_r>0\) ⇒ \(m=y+r\nabla\log p_r\).
- **U2-03:** Differentiate the normalized posterior numerator and denominator ⇒ \(\partial_{y_j}m_i=\operatorname{Cov}(X_i,X_j\mid y)/r\). Third-party regularity assumptions are unnecessary; U0 supplies all required integrability.
- **U2-04:** U2-03 and finite posterior second moments ⇒ \(J_m\) is symmetric PSD. No positive-definiteness claim is made; rank-deficient cases are valid.

- **U3-01:** Positive smooth \(p_r\) satisfying the heat equation ⇒ \(\partial_r\log p_r=\frac12(\Delta\log p_r+\|\nabla\log p_r\|^2)\).
- **U3-02:** U3-01, smoothness, and symmetry of the Hessian ⇒ \(\partial_r s=\frac12\Delta s+J_s s\).
- **U3-03:** U2-02 and U3-02 ⇒ \(\partial_r m=\frac12\Delta m+J_m(m-y)/r\). The statement is necessary only; no converse is used.

- **U4-01:** \(t\in(0,1)\), \(y=z/t\), and \(r=((1-t)/t)^2\) ⇒ the conditional laws agree and \(D(z,t)=m(y,r)\). The likelihood factor is \(t^{-d}\), independent of \(X\).
- **U4-02:** U4-01 and smoothness ⇒ \(J_m=tJ_D\), \(\Delta_y m=t^2\Delta_zD\), \(y'=-z/t^2\), and \(r'=-2(1-t)/t^3\). All time derivatives here hold \(z\) fixed.
- **U4-03:** U3-03 and U4-02 ⇒ the three-term FM-coordinate PDE at lines 95–97, with the stated signs and coefficients.
- **U4-04:** Conditional straight-line velocity \(X-e\), with \(e=(z-tX)/(1-t)\) given \(Z_t=z\) ⇒ \(\mathbb E[X-e\mid Z_t=z]=(D-z)/(1-t)\). This explicitly verifies the “uncapped marginal velocity” label.
- **U4-05:** U4-03 and U4-04 ⇒ the material-derivative identity at lines 103–104.
- **U4-06:** Replacing \(1-t\) by \(\delta(t)\) in the external velocity ⇒ a different material derivative, while U1’s fixed-time optimum remains the same. No capped-sampler identity is inferred.

- **U5-01:** Independent scalar standard Gaussians ⇒ \(Z_t\sim N(0,V)\), \(D=tz/V\), and \(u=(2t-1)z/V\), where \(V=t^2+(1-t)^2>0\).
- **U5-02:** Differentiate those explicit functions ⇒ material drift \((1-t)z/V^2\), equal to \(2z\) at \(t=1/2\).
- **U5-03:** The explicit Gaussian velocity field ⇒ \(F=z/\sqrt V\) satisfies the transport equation and extends to \(F(z,1)=z\). Endpoint evaluation here is legitimate from the explicit formula and requires no general endpoint regularity theorem.
- **U5-04:** At \(t=1/2\), \(\operatorname{Var}Z_t=1/2\), \(\delta=1/2\), and U1 ⇒ excess risk \(2(\sqrt2-1)^2\) per coordinate.
- **U5-05:** U5-02–04 ⇒ invariance is not necessary for this supervised conditional mean. This does not reject predictors trained with a distinct endpoint target.

- **U6-01:** Hypothetical \(N(0,v)\) prior, \(v>0\) ⇒ \(D_v=tvz/[t^2v+(1-t)^2]\).
- **U6-02:** Every such prior belongs to U0 ⇒ each \(D_v\) satisfies the same necessary denoiser PDE.
- **U6-03:** Evaluation under the fixed true \(N(0,1)\) law, \(v=2\), \(t=1/2\), and U1 ⇒ excess risk \(2/9>0\). Hence zero PDE residual alone does not identify the correct regression function or guarantee quality.

- **U7-01:** Fixed \(y\), \(\lambda>0\), and posterior proportional to \(e^{-\lambda\|X-y\|^2/2}\) times the prior ⇒ normalized differentiation yields \(-\frac12\operatorname{Cov}_\lambda(X,\|X-y\|^2)\). U0 moments justify the derivative.
- **U7-02:** Write \(X=m+\xi\), use \(\mathbb E\xi=0\) ⇒ the covariance equals \(\kappa+2C(m-y)\), giving the stated precision derivative.
- **U7-03:** A Gaussian posterior has vanishing third centered moments, whereas general mixture posteriors need not ⇒ setting \(\kappa=0\) requires an additional closure assumption. No statement about arbitrary hidden state or historical architecture causation follows.

- **U8-01:** The specified atomic posterior laws ⇒ both have mean zero and variance one, but third central moments zero and \(3/2\).
- **U8-02:** For the fixed anchor \(y=0,\lambda=1\), prior weights proportional to \(q_i e^{x_i^2/2}\) ⇒ valid positive finite-support priors in U0 whose posterior weights equal \(q_i\).
- **U8-03:** U7 applied to these two realized posteriors ⇒ precision drifts zero and \(-3/4\).
- **U8-04:** Same \((y,\lambda,m,C)\), unequal derivatives ⇒ no single prior-independent function of that tuple represents this derivative across U0. Embedding these scalar atoms as \((x_i,0,\ldots,0)\) extends the counterexample to any fixed finite \(d\ge1\). Additional prior information or latent state is explicitly outside this impossibility claim.

**Issues**

1. **R1 — explicitly choose the smooth conditional-mean version**
   - **Status:** UNCLEAR
   - **Impact:** COSMETIC
   - **Severity:** MINOR
   - **Category:** REGULARITY_GAP / notation clarification
   - **Location:** `derivation.md:20–27`, `33–35`, and pointwise statements beginning at `67`.
   - **Why:** A conditional expectation alone denotes an almost-everywhere equivalence class. Pointwise derivatives require the smooth Bayes-ratio representative. Lines 33–35 already imply this intended choice, so I do not count this as a substantive proof gap.
   - **Counterexample to the unrestricted-version reading:** For \(X\sim N(0,1)\) and \(r=1\), \(m(y,1)=y/2\). Changing only its value at \(y=0\) to \(1\) gives another conditional-expectation version, but its derivative at zero is undefined.
   - **Downstream effect:** None under the intended posterior-ratio interpretation; arbitrary versions would invalidate pointwise derivative statements.
   - **Minimal fix:** Add: “Throughout, \(m\) and \(D\) denote their smooth Bayes-ratio versions,” preferably with
     \[
     m(y,r)=
     \frac{\int x\phi_r(y-x)\,\mu(dx)}
          {\int\phi_r(y-x)\,\mu(dx)}.
     \]

2. **R2 — distinguish density notation and state the vector domain explicitly**
   - **Status:** UNCLEAR
   - **Impact:** COSMETIC
   - **Severity:** MINOR
   - **Category:** notation / typed-domain bookkeeping
   - **Location:** `derivation.md:41`, `67`.
   - **Why:** \(p_r\) denotes the density of \(Y_r\), but \(p_t\) in U1 is not separately defined. “Every \(y,r>0\)” also treats the vector \(y\) as though it were a positive scalar.
   - **Counterexample:** None; both relevant observation densities are strictly positive, so the null-set issue does not change U1.
   - **Downstream effect:** Exposition only.
   - **Minimal fix:** Replace “\(p_t\)-almost-everywhere” with “almost everywhere under the law of \(Z_t\),” and write “every \(y\in\mathbb R^d\) and \(r>0\).”

**Key verification details**

The domination argument is sufficient. For any required finite derivative order, compact \(K\subset\mathbb R^d\), and \(0<a<b<\infty\), differentiation of the Gaussian kernel gives a polynomial in \(y-x\) times the kernel, with bounded powers of \(r^{-1}\). Thus, for suitable finite \(N,C\),
\[
\sup_{y\in K,\ r\in[a,b]}
(1+\|x\|)\,
|\partial_y^\alpha\partial_r^k\phi_r(y-x)|
\le C(1+\|x\|^N).
\]
U0’s finite-GMM prior makes this integrable. This covers the numerator’s extra \(X\), the heat equation, mixed derivatives, and spatial derivatives through third order. Positivity and continuity give a positive minimum of the denominator on each compact set, justifying quotient and logarithm derivatives there. For U7, a compact positive precision interval and fixed or compact \(y\) similarly give an integrable polynomial envelope after differentiating the unnormalized posterior weight.

The coordinate algebra yields
\[
\begin{aligned}
\partial_tD
&=-\frac{J_Dz}{t}
-\frac{1-t}{t}\Delta_zD
-\frac{2}{1-t}J_D(D-z/t)\\
&=-\frac{1-t}{t}\Delta_zD
-\frac{2}{1-t}J_DD
+\frac{1+t}{t(1-t)}J_Dz.
\end{aligned}
\]
Adding \(J_D(D-z)/(1-t)\) gives exactly U4’s material derivative.

For U5, \(V'=4t-2\), so
\[
\left(\frac tV\right)'
+\frac tV\frac{2t-1}{V}
=\frac{V-tV'+t(2t-1)}{V^2}
=\frac{1-t}{V^2}.
\]
Moreover, \(u=(V'/2V)z\), making \(z/\sqrt V\) constant along its trajectories. The two reported excess risks follow directly from
\[
\frac{\mathbb E Z_{1/2}^2}{(1/2)^2}
\left(\text{prediction-slope error}\right)^2
=2\left(\text{prediction-slope error}\right)^2.
\]

For U7,
\[
\operatorname{Cov}(X,\|X-y\|^2)
=\mathbb E\left[\xi\|\xi+(m-y)\|^2\right]
=\kappa+2C(m-y).
\]
No sign or factor of two is missing.

For U8,
\[
\begin{array}{c|ccc}
 & \mathbb E X & \mathbb E X^2 & \mathbb E X^3\\
q_A&0&1&0\\
q_B&-.4+.4&.2+.8&-.1+1.6
\end{array}
\]
and the exact fixed prior for \(q_B\) has weights
\[
\pi_-=\frac{4e^{1/8}}{4e^{1/8}+e^2},
\qquad
\pi_+=\frac{e^2}{4e^{1/8}+e^2}.
\]
At the prescribed observation, multiplying these by \(e^{-x_i^2/2}\) restores the weight ratio \(4:1\). These are fixed priors, not priors varied while differentiating \(\lambda\).

**Independent counterexample and degeneracy pass**

- **Point mass \(X=a\):** \(m=D=a\), covariance and all relevant derivatives vanish. Both PDEs and the precision identity reduce to zero. This confirms PSD, rather than positive definite, is the correct Jacobian claim. It also confirms that “need not vanish” is appropriately weaker than “never vanishes.”
- **Arbitrary singular Gaussian covariance \(\Sigma\succeq0\):**
  \[
  m=\mu+\Sigma(\Sigma+rI)^{-1}(y-\mu),\qquad
  C=r\Sigma(\Sigma+rI)^{-1}.
  \]
  No inverse of \(\Sigma\) is required. The Laplacian is zero, and both sides of the denoiser PDE equal
  \[
  -\Sigma(\Sigma+rI)^{-2}(y-\mu).
  \]
  No degeneracy counterexample was found.
- **Symmetric two-point prior \(X\in\{-a,a\}\):**
  \[
  m=a\tanh(\lambda ay),\quad
  C=a^2\operatorname{sech}^2(\lambda ay),\quad
  \kappa=-2mC.
  \]
  Thus \(\partial_\lambda m=Cy\), while the covariance-only expression would give \(C(y-m)\). The full U7 identity agrees exactly; omitting the third-moment term generally fails.
- **Identical mixture components or arbitrarily small positive weights:** These do not break positivity or fixed-prior local domination. The draft asserts no constants uniform in mixture parameters.
- **Extreme \(t\to0,1\) or \(r\to0\):** Coefficients or derivatives can grow. This does not contradict the stated interior-only scope. No endpoint-uniform step is used.
- **Heavy-tailed priors:** Outside the quantified prior class; they do not refute the statements audited here.

**Scope and remaining obligations**

The dependency graph is acyclic. U6’s FM-coordinate application also uses the coordinate transformation in U4; adding `U4` to that skeleton row would make the graph more explicit, although its current dependencies already include the underlying PDE.

The draft correctly preserves every requested non-claim: consistency is insufficient; no theorem novelty, architecture improvement, general hidden-state impossibility, or causal explanation of older covariance-model failures is established.

Two cosmetic clarifications remain. Separately, the actual benchmark manifests must establish that the three named implementation laws satisfy U0; their names and high-level descriptions in `protocol.md` are not a substitute for that application check. Numerical fixtures, source/data hashes, runtime results, and citation accuracy were outside this read-only mathematical review and have not been certified here.
