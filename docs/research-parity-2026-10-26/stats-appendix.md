# Mathematical and statistical details

The saved protocol predates all new negative-input outcomes. The model implementation, checkpoints, runtime buffers, configuration, common population draws, and evaluation times are fixed. The added sigmoid-GLU is secondary; its historical SwiGLU128 comparator is verified to be the same current predictor, not a new independent arm.

## Finite antithetic decomposition

For a row x with an odd oracle v, define O=(f(x)-f(-x))/2 and E=(f(x)+f(-x))/2. Expanding the two squared errors gives, in each output coordinate,

```text
[(f(x)-v)^2 + (f(-x)+v)^2]/2 = (O-v)^2 + E^2.
```

At each time, X contains the 4096 physical input rows and 16 columns. Weighted least squares projects O-v onto X without an intercept. On the expanded antipodal panel, inputs and odd targets have zero paired means; this no-intercept fit is equivalent to fitting all pairs. The fitted part and residual are orthogonal in the weighted empirical inner product. Averaging outputs gives T=L+N+even exactly. Here N is a residual relative to a finite raw-linear span; it is not a proof of a global expressivity bound or an identified biological/physical mechanism.

Projecting O and v separately also defines residuals O_r and v_r. The reported alignment is <O_r,v_r>/<v_r,v_r>; the relative RMS is sqrt(<O_r,O_r>/<v_r,v_r>). These need not lie in [0,1]. At time 0.1 the oracle residual energy is only 0.000234, so normalized ratios there have a small denominator. At time 0.7 its energy is 1.252905; the corresponding comparison is illustrated in the report, and all times remain in the released table.

## The SiLU bound and what it does not establish

Write rho(z)=z*sigmoid(z)=z/2+g(z), where g is even. For p+=m+d and p-=m-d,

```text
(rho(p+)-rho(p-))/2 = d/2 + R(m,d)
R(m,d) = [g(m+d)-g(m-d)]/2.
```

For real z, |z|*sigmoid'(z)<=1/4. One proof uses sigmoid'(z)=1/[4*cosh(z/2)^2] and cosh(u)^2>=1+u^2>=2|u|. Thus rho' lies in [-1/4,5/4], |g'|<=3/4, and |rho''|<=2*(1/4)+1/4=3/4. Evenness of g first gives |R|<=3|m|/4. Since R(0,d)=0 and |partial R/partial m|<=3|d|/4, it also gives |R|<=3|m*d|/4. Combining them yields

```text
|R(m,d)| <= .75*|m|*min(|d|,1).
```

The experiment applies this coordinatewise to actual paired preactivations. It compares RMS(R) and the RMS upper bound to RMS(d/2), averaging units within a row. Large measured ratios reject the small-additional-response account. They do not prove that R supplies useful raw-input nonlinear features: both m and d may already depend nonlinearly on x, and a large local response can be poorly aligned with the final target.

For bias-free odd linear projections u and v, the odd part of sigmoid(u)*v is v/2, while the odd part of SiLU(u)*v is u*v*tanh(u/2)/2. Product statistics use these formulas at the observed odd projections with their even centers formally set to zero. This is a reference calculation, not an intervention on the full predictor, and no causal attribution or deployment gain is inferred from its energy.

## Resampling and the fixed decisions

The 256 row resamples use generator seed 0. Each resample preserves the same row weights across all models, output coordinates, and five times; every weighted projection is refitted. The original panel is index 0 and bootstrap samples occupy indices 1–256. Units and coordinates are not independent resampling units. All weights, projection coefficients and per-resample components are archived.

The registered ratio is the signed component excess divided by positive total ODE-minus-reference excess. Every saved primary denominator is positive. For a component to explain a majority, the lower 2.5th percentile must be at least 0.5 against all three SwiGLU references. To reject majority status, every upper 97.5th percentile must be below 0.5. These rules are unchanged after observation.

The nonlinear fractions' raw percentile ranges do not contain their full-panel estimates: 99.40% versus [98.80%,99.23%], 100.05% versus [99.45%,99.86%], and 99.14% versus [98.56%,98.94%]. The refitted empirical-projection statistic exhibits a resampling shift. We therefore present these as central resample sensitivity ranges, without claiming calibrated population confidence coverage or exact population fractions. No post-result interval correction, altered threshold, or new bootstrap run replaces the registered calculation. The descriptive finite-panel decomposition and its reproduction remain exact; the majority conclusion is far from the predeclared 50% cutoff in these resamples.

## Independent reconstruction and preserved failures

Actual hooks reproduce 20 old float32 primary panels and all 20 full positive population panels. The historical GLU reproduces five old float64 panels. Manual traversal explicitly computes both K2 solves, controllers, middle activations, generic gate products, residual updates and output projections, without calling the block/FFN forward methods. All 50 new plus/minus field tensors agree bitwise. The 1005 per-row activation-statistic arrays agree to at most 1.14e-13; weighted means agree to at most 5.69e-14.

The independent audit rotates both inputs and outputs into latent coordinates, solves each weighted projection through QR, and directly computes residual energies. All 1285 fits have rank 16. Maximum component-risk discrepancy is 8.55e-15; coefficient discrepancy is 1.51e-14. Ratio discrepancies are larger when oracle residual energy is tiny (up to 2.07e-10). All primary decisions and reported resample endpoints reproduce.

One positive-only preflight failed because recomputing a strided product after copying selected a different float64 vector kernel, changing at most 1.11e-16. The corrected extraction compares the live reconstructed product bitwise. The first frozen main then failed during metadata binding because a historical summary path was incorrect. Its terminal exit1 and files are retained in parity_20261026; no negative fields were evaluated in that attempt. The corrected source and sole completed main are in parity_20261026_v2. No model training was restarted or rerun. Source manifests, input bindings, receipts and runtime telemetry retain this distinction.

Final reports and plots are produced after the independent audit. They do not alter predictors, earlier artifacts or the submitted Report1. Main and audit processes have terminal exit0. The larger useful-architecture objective remains active and unachieved.
