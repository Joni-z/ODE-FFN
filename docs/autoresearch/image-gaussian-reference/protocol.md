# A train-only Gaussian image reference

Zhizhe Zhang · October 28, 2026. Register before fitting real statistics or scores.

## Question and role
How competitive is one explicitly linear Gaussian conditional-mean predictor
against the completed CIFAR-10 models? The cold-control result establishes
that adaptive clocks are unnecessary for one finite margin. It does not say
whether a Gaussian reference can attain that margin. This study adds ordinary
reference evidence; it cannot establish a new ODELayer successor or causally
explain the nonlinear networks. A negative result stops only this reference.

## Exactly two models, one train-only fit
Use the original 45000 training indices only, native float32 pixel
normalization promoted to float64, and full 3072-dimensional images. Estimate
mean and covariance with the maximum-likelihood denominator 45000, using a
fixed two-pass batch 512 accumulation. No validation/test pixels fit parameters.

The fixed active learned budget is 185744 from the completed constant model.
A spectral Gaussian stores d mean entries, d*r basis values, r eigenvalues and
one isotropic-tail variance: d+(d+1)*r+1. The largest admissible rank is 59,
storing 184380 scalars; rank 60 would exceed the budget. This count includes
redundant orthonormal-basis entries, and is not a claim of identical degrees
of freedom, precision bytes, training work or optimization to a neural model.

- `ppca59`: leading 59 training covariance eigenpairs; tail variance is the
  average remaining eigenvalue. No rank, residual-noise or eigenvalue-floor sweep.
- `isotropic`: the same training mean and trace(covariance)/3072 variance.
  It is an attribution reference, not a second tuned candidate.

Use float64 fitted statistics/conditioning for numerical clarity and return
float32 clean predictions. Stored bytes and scalar precision are reported.
The fitted covariance must be positive and satisfy the stated spectral checks;
stop on numerical failure rather than silently clipping or adding a variance.
No mixtures, learned FFN, optimizer steps, new seed or neural checkpoint changes.

## Prediction and unchanged evaluation
For this fitted Gaussian prior, not the true unknown image law, use
D(z,t)=mu+t*Sigma*[t²*Sigma+(1-t)²*I]^-1*(z-t*mu).
The [derivation](theory.md) specifies its low-rank action, time derivative and
boundaries. Retain the original 1024 paired validation states, all five 64-pair
time panels, Gaussian starts and 2048 original real Inception references.

Generate 1024 images per reference using the unchanged native CPU float32
Heun128 and denominator cap.05. No direct exact Gaussian endpoint is
substituted. No state clipping, alternative sampler or changed feature conversion.
Keep all 32 old model/budget endpoints, including old failed replacements;
new statistical fits are labelled as such, without a fictitious 20k training
budget. First eight common starts are fixed examples.

## Decisions before outcomes
For each Gaussian reference record the same 20k-reference screen: FID<=.95
of both Swi86/SiLU131, both fitting errors<=1.05 of the better corresponding
reference, and strict default SiLU64 ordering. Report continuous differences to
native K1, cold fixed P and matched-init SiLU, including every 5k/10k context row.
These are reference-competitiveness gates only. No outcome promotes a new ODE
method, proves what a neural model learned, or proves all Gaussians insufficient.
Do not select another rank, covariance form, fit objective or sampling setting.

## Verification and finite resources
Use the unchanged warm Mac mini image-metrics-v2@dc729a50 environment. One CPU
thread for statistics/sampling; existing MPS only for pinned feature extraction.
Available RAM 4.63GiB and disk 54.81GiB cover a sub 1GiB process and below .5GiB
expected new disk. Real fit ceiling 600s; independent fit audit 600s; sample
cohort 300s; independent samples 300s; feature/independent feature stages 600s each.
If an eligibility timing witness contradicts this estimate, revise the resource
plan before real fitting, preserving the failed readiness source and outputs.

Before formal fitting, require a committed runnable protocol/source and fresh
agent-follows-doc preflight: seeded dense/low-rank/independent posterior and
state/time derivatives, MLE/covariance identities, storage/dispatch cost,
physical save/load, actual-shape native-Heun fixtures and historical MPS/FID.
No package installation or environment change is selected.

The independent fit audit recomputes mean/covariance in batch 257, checks
normalization, all leading eigenvalues, orthogonality/eigen residual and tail
variance, without comparing arbitrary eigenvector signs. Independent posterior
calculation uses Woodbury solves, not the production spectral formula.
Validation/time-panel predictors agree within 3e-6 relative/absolute; independently
sampled float32 endpoints require max difference<3e-5 and RMS<3e-6. Formal raw
arrays are never replaced. Recompute every new/reference feature row bitwise
from the formal pixels and all 34 FIDs independently with SciPy gesvd within 1e-8.
These tolerances are frozen before fitted outcomes, not selected afterward.

Save exact commands, PID/create-time, primed resource logs and consumed tool
exits. Re-poll the same live process after an observation timeout. Preserve
all parent seals, datasets, environments, P histories, failed sources and
Report1. Stop after the reference/report/archive; no follow-on neural family
or changed benchmark is selected by this protocol.
