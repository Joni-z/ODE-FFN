# Conditional means after a fixed linear input projection

Let Y be an equally weighted eight-component Gaussian mixture, with component means mu_j=Q m_j, m_j the radius3 Ring8 means. The common covariance is Sigma=q I+(.04-q) Q Q^T, where q=0 or.04. Z is independent standard Gaussian and X=a Z+t Y, a=1-t. Time is observed and delta=max(a,.05). Parameters and the learned input matrix A are frozen; all subsequent hidden computation depends on X only through A X and time. No raw skip before that matrix is present. Let B have orthonormal rows spanning the row space of A and U=B X.

For t<1 the conditional component covariance of U is positive definite:

```text
D = a^2 I + t^2 B Sigma B^T
p_j(U,t) proportional to exp(-.5*(U-t B mu_j)^T D^-1*(U-t B mu_j))
mu_bar = sum_j p_j mu_j
E[Y|U,t] = mu_bar + t Sigma B^T D^-1*(U-t B mu_bar)
E[Z|U,t] = a B^T D^-1*(U-t B mu_bar).
```

These follow by conditioning the joint Gaussian of(Y,Z,U) within each component and averaging posterior weights. B's rank may be less than its width; no inverse of A or arbitrary ridge parameter is needed. Input bias can be subtracted and an independent time embedding changes no observation sigma-algebra.

The regression pair field is V=a/delta*(Y-Z). Conditioning on X gives the existing full oracle f=(E[Y|X,t]-X)/delta. Direct raw prediction minimizes squared error at E[V|U,t]. For clean prediction, delta*V+X=Y, so the optimal raw output is E[Y|U,t]. For shifted prediction, V+X/delta=Y/delta; its decoded optimum is the same as clean, although its raw scale and fitted input matrix can differ.

For a fixed input matrix let f_U denote these decoder-specific best fields. For any actual raw predictor g(U,t), write f_hat-f=(f_U-f)+(f_hat-f_U). The second term is measurable in(U,t), including for skip heads because the explicit X skip cancels. The first term has conditional expectation zero given(U,t), by the tower property. Hence the expected cross term vanishes and population oracle risk splits into information risk plus residual risk. Orthogonal tangent/normal projections preserve this argument. All moments exist: times are capped, Gaussian-mixture moments are finite and the native predictors have bounded raw output after final normalization.

Finite-bank empirical cross terms need not vanish. They must be measured rather than set to zero. Empirical information error is not a lower bound on every finite-bank actual error; the risk statement is a population identity. The finite banks also store rounded float32 inputs, so the known continuous-law conditional formula is evaluated at those rounded coordinates, as is the existing full oracle. No different observation law is silently substituted.

The remaining residual includes final-readout rank restrictions, nonlinear function approximation and optimization; its difference is not a causal effect of P. Independently trained input matrices also differ. Neither a small information contribution nor a large residual contribution establishes a successful new ODE method.

For efficient computation, set C=BQ, d0=a^2+t^2 q. Then D=d0 I+t^2(.04-q) C C^T. The formal implementation diagonalizes the two-column C. An independent implementation can instead solve the2-by2 matrix d0 I+t^2(.04-q) C^T C and use Woodbury. Full observation B=I reduces to the native oracle; rank-deficient observations are handled by extracting only nonzero input singular directions.
